"""Compare deterministic, raw model, and final guarded response generation."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from restaurant_assistant.config import get_settings
from restaurant_assistant.data_loader import load_restaurants
from restaurant_assistant.dialogue_state import DialogueState
from restaurant_assistant.llm_generator import (
    DEFAULT_RESPONSE_MODEL,
    GroundedResponseGenerator,
    RawGenerationResult,
    validate_generated_response,
)
from restaurant_assistant.nlg import contains_json_or_debug_leakage
from restaurant_assistant.ranking import RankedRestaurant
from restaurant_assistant.response_prompt import ResponsePromptFields, parse_response_input

try:
    from .build_response_training_data import build_rows
except ImportError:
    from build_response_training_data import build_rows


DEFAULT_EVAL_FILE = ROOT / "data" / "evaluation" / "response_generation_eval.jsonl"
DEFAULT_JSON_REPORT = ROOT / "reports" / "response_generation_comparison.json"
DEFAULT_MD_REPORT = ROOT / "reports" / "response_generation_comparison.md"
DEFAULT_ADAPTER_PATH = ROOT / "models" / "response-generator-lora"

METRIC_KEYS = (
    "deterministic_baseline_response",
    "raw_pretrained_model_response",
    "raw_trained_lora_response",
    "final_guarded_response",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate guarded response generation.")
    parser.add_argument("--sample-data", action="store_true", help="Use bundled sample restaurants.")
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--markdown-report", type=Path, default=DEFAULT_MD_REPORT)
    parser.add_argument("--response-model-name", default=DEFAULT_RESPONSE_MODEL)
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument(
        "--run-llm",
        action="store_true",
        help="Actually load local Transformers models. By default, LLM columns are marked as not run.",
    )
    return parser


def _load_rows(path: Path, restaurants: list[dict[str, Any]]) -> list[dict[str, str]]:
    if path.exists() and path.read_text(encoding="utf-8").strip():
        rows = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    return build_rows(restaurants)[::4]


def _normalise(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _coerce_people(value: Any) -> int | None:
    try:
        people = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return people if people > 0 else None


def _find_restaurant(
    *,
    name: str,
    evidence_records: list[dict[str, Any]],
    restaurants: list[dict[str, Any]],
) -> dict[str, Any] | None:
    wanted = _normalise(name)
    if not wanted:
        return evidence_records[0] if evidence_records else None
    for record in [*evidence_records, *restaurants]:
        if _normalise(record.get("name")) == wanted:
            return record
    return evidence_records[0] if evidence_records else None


def _state_from_prompt(
    fields: ResponsePromptFields,
    restaurants: list[dict[str, Any]],
) -> DialogueState:
    evidence_records = fields.evidence_records
    state_values = fields.state
    restaurant = _find_restaurant(
        name=state_values.get("restaurant", ""),
        evidence_records=evidence_records,
        restaurants=restaurants,
    )
    booking_status = state_values.get("booking_status") or state_values.get("status") or "none"
    selected = restaurant if restaurant else (evidence_records[0] if evidence_records else None)
    return DialogueState(
        food=state_values.get("food") or (selected or {}).get("food"),
        area=state_values.get("area") or (selected or {}).get("area"),
        pricerange=state_values.get("pricerange") or (selected or {}).get("pricerange"),
        day=state_values.get("day"),
        time=state_values.get("time"),
        people=_coerce_people(state_values.get("people")),
        selected_restaurant=selected,
        booking_restaurant=selected if fields.intent in {"book", "reschedule", "cancel", "booking_info", "booking_list"} else None,
        booking_status=booking_status,
        booking_reference=state_values.get("booking_reference"),
    )


def _ranked(records: list[dict[str, Any]]) -> list[RankedRestaurant]:
    ranked: list[RankedRestaurant] = []
    for record in records:
        ranked.append(
            RankedRestaurant(
                record=record,
                score=1.0,
                matched_constraints=["food", "area", "pricerange"],
                missing_unmatched_constraints=[],
                explanation="response generation evaluation evidence",
                similarity=1.0,
            )
        )
    return ranked


def _validation_payload(
    text: str,
    *,
    evidence_records: list[dict[str, Any]],
    known_restaurant_records: list[dict[str, Any]],
) -> dict[str, Any]:
    validation = validate_generated_response(
        text,
        evidence_records=evidence_records,
        known_restaurant_records=known_restaurant_records,
    )
    return {
        "ok": validation.ok,
        "rejection_reason": validation.reason,
    }


def _baseline_result(
    text: str,
    *,
    evidence_records: list[dict[str, Any]],
    known_restaurant_records: list[dict[str, Any]],
) -> dict[str, Any]:
    validation = _validation_payload(
        text,
        evidence_records=evidence_records,
        known_restaurant_records=known_restaurant_records,
    )
    return {
        "text": text,
        "mode": "deterministic_baseline",
        "attempted": False,
        "accepted": False,
        "rejected": False,
        "fallback": False,
        "skipped": False,
        "latency_seconds": 0.0,
        "validation": validation,
        "final_validation": validation,
        "rejection_reason": None,
    }


def _skipped_model_result(mode: str, reason: str) -> dict[str, Any]:
    return {
        "text": "",
        "mode": mode,
        "attempted": False,
        "accepted": False,
        "rejected": False,
        "fallback": False,
        "skipped": True,
        "latency_seconds": 0.0,
        "validation": {"ok": None, "rejection_reason": reason},
        "final_validation": {"ok": None, "rejection_reason": reason},
        "rejection_reason": reason,
    }


def _raw_model_result(
    raw: RawGenerationResult,
    *,
    evidence_records: list[dict[str, Any]],
    known_restaurant_records: list[dict[str, Any]],
) -> dict[str, Any]:
    validation = _validation_payload(
        raw.text,
        evidence_records=evidence_records,
        known_restaurant_records=known_restaurant_records,
    )
    accepted = bool(raw.attempted and validation["ok"])
    rejected = bool(raw.attempted and not accepted)
    reason = raw.error or validation["rejection_reason"]
    return {
        "text": raw.text,
        "mode": raw.mode,
        "attempted": raw.attempted,
        "accepted": accepted,
        "rejected": rejected,
        "fallback": False,
        "skipped": False,
        "latency_seconds": raw.latency_seconds,
        "validation": validation,
        "final_validation": validation,
        "rejection_reason": reason if rejected else None,
    }


def _final_guarded_result(
    *,
    baseline_text: str,
    candidate: dict[str, Any] | None,
    run_llm: bool,
    evidence_records: list[dict[str, Any]],
    known_restaurant_records: list[dict[str, Any]],
) -> dict[str, Any]:
    if run_llm and candidate and candidate.get("attempted") and candidate.get("accepted"):
        text = candidate["text"]
        mode = f"accepted_{candidate['mode']}"
        fallback = False
    else:
        text = baseline_text
        fallback = bool(run_llm and candidate and candidate.get("attempted") and not candidate.get("accepted"))
        mode = "deterministic_baseline_fallback" if fallback else "deterministic_baseline"

    final_validation = _validation_payload(
        text,
        evidence_records=evidence_records,
        known_restaurant_records=known_restaurant_records,
    )
    pre_validation = candidate.get("validation") if candidate else {"ok": None, "rejection_reason": "not_attempted"}
    return {
        "text": text,
        "text_before_fallback": candidate.get("text", "") if candidate else "",
        "mode": mode,
        "attempted": bool(candidate and candidate.get("attempted")),
        "accepted": bool(candidate and candidate.get("accepted")),
        "rejected": bool(candidate and candidate.get("rejected")),
        "fallback": fallback,
        "skipped": False,
        "latency_seconds": float(candidate.get("latency_seconds", 0.0)) if candidate else 0.0,
        "validation": pre_validation,
        "final_validation": final_validation,
        "rejection_reason": candidate.get("rejection_reason") if candidate and candidate.get("rejected") else None,
    }


def _guard_candidate(
    *,
    pretrained: dict[str, Any],
    trained: dict[str, Any],
    adapter_available: bool,
) -> dict[str, Any] | None:
    if adapter_available and not trained.get("skipped"):
        return trained
    if trained.get("attempted"):
        return trained
    if pretrained.get("attempted"):
        return pretrained
    return None


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _metrics(cases: list[dict[str, Any]], key: str) -> dict[str, Any]:
    results = [case[key] for case in cases]
    count = len(results)
    skipped = sum(int(result.get("skipped")) for result in results)
    active_count = count - skipped
    attempted = sum(int(result.get("attempted")) for result in results)
    accepted = sum(int(result.get("accepted")) for result in results)
    rejected = sum(int(result.get("rejected")) for result in results)
    fallback = sum(int(result.get("fallback")) for result in results)
    before_validations = [result.get("validation", {}) for result in results if result.get("attempted")]
    after_validations = [
        result.get("final_validation") or result.get("validation", {})
        for result in results
        if not result.get("skipped")
    ]
    before_texts = [
        result.get("text_before_fallback", result.get("text", ""))
        for result in results
        if result.get("attempted")
    ]
    latencies = [float(result.get("latency_seconds", 0.0)) for result in results if not result.get("skipped")]
    rejection_reasons = Counter(
        str(result.get("rejection_reason") or result.get("validation", {}).get("rejection_reason"))
        for result in results
        if result.get("rejected")
    )
    before_denominator = len(before_validations)
    after_denominator = len(after_validations)
    return {
        "status": "skipped" if skipped == count and count else "completed",
        "case_count": count,
        "model_attempt_count": attempted,
        "accepted_model_output_count": accepted,
        "rejected_model_output_count": rejected,
        "fallback_count": fallback,
        "fallback_rate": _rate(fallback, active_count),
        "groundedness_before_fallback": _rate(
            sum(int(validation.get("ok") is True) for validation in before_validations),
            before_denominator,
        ),
        "groundedness_after_fallback": _rate(
            sum(int(validation.get("ok") is True) for validation in after_validations),
            after_denominator,
        ),
        "json_debug_leakage_before_fallback": _rate(
            sum(int(contains_json_or_debug_leakage(text)) for text in before_texts),
            before_denominator,
        ),
        "unsupported_claim_rate_before_fallback": _rate(
            sum(int(validation.get("rejection_reason") == "unsupported_claim") for validation in before_validations),
            before_denominator,
        ),
        "average_latency_seconds": round(statistics.mean(latencies), 6) if latencies else 0.0,
        "response_latency_seconds": {
            "mean": round(statistics.mean(latencies), 6) if latencies else 0.0,
            "median": round(statistics.median(latencies), 6) if latencies else 0.0,
            "max": round(max(latencies), 6) if latencies else 0.0,
        },
        "rejection_reason_counts": dict(sorted(rejection_reasons.items())),
        "skipped_cases": skipped,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    settings = get_settings()
    restaurants = load_restaurants(settings, use_sample=args.sample_data)
    rows = _load_rows(args.eval_file, restaurants)
    adapter_available = args.adapter_path.exists()
    cases: list[dict[str, Any]] = []

    pretrained_generator = GroundedResponseGenerator(
        enable_llm=args.run_llm,
        model_name=args.response_model_name,
        known_restaurants=restaurants,
    )
    trained_generator = GroundedResponseGenerator(
        enable_llm=args.run_llm and adapter_available,
        model_name=str(args.adapter_path),
        pretrained_fallback=False,
        known_restaurants=restaurants,
    )

    for row in rows:
        input_text = str(row.get("input") or "")
        reference_output = str(row.get("output") or "")
        fields = parse_response_input(input_text)
        evidence_records = fields.evidence_records
        known_restaurant_records = [*restaurants, *evidence_records]
        state = _state_from_prompt(fields, restaurants)
        ranked = _ranked(evidence_records)

        baseline = _baseline_result(
            reference_output,
            evidence_records=evidence_records,
            known_restaurant_records=known_restaurant_records,
        )

        if args.run_llm:
            pretrained_raw = pretrained_generator.generate_raw(
                fields.user,
                state,
                ranked,
                intent=fields.intent,
                missing_slots=fields.missing_slots,
            )
            pretrained = _raw_model_result(
                pretrained_raw,
                evidence_records=evidence_records,
                known_restaurant_records=known_restaurant_records,
            )
        else:
            pretrained = _skipped_model_result("not_run", "run_llm_false")

        if args.run_llm and adapter_available:
            trained_raw = trained_generator.generate_raw(
                fields.user,
                state,
                ranked,
                intent=fields.intent,
                missing_slots=fields.missing_slots,
            )
            trained = _raw_model_result(
                trained_raw,
                evidence_records=evidence_records,
                known_restaurant_records=known_restaurant_records,
            )
        else:
            trained = _skipped_model_result(
                "adapter_missing" if not adapter_available else "not_run",
                "adapter_missing" if not adapter_available else "run_llm_false",
            )

        candidate = _guard_candidate(
            pretrained=pretrained,
            trained=trained,
            adapter_available=adapter_available,
        )
        final_guarded = _final_guarded_result(
            baseline_text=reference_output,
            candidate=candidate,
            run_llm=args.run_llm,
            evidence_records=evidence_records,
            known_restaurant_records=known_restaurant_records,
        )

        cases.append(
            {
                "intent": fields.intent,
                "user": fields.user,
                "state": fields.state,
                "food": state.food,
                "area": state.area,
                "pricerange": state.pricerange,
                "day": state.day,
                "time": state.time,
                "people": state.people,
                "booking_status": state.booking_status,
                "booking_reference": state.booking_reference,
                "missing_slots": fields.missing_slots,
                "restaurant_evidence": evidence_records,
                "reference_output": reference_output,
                "deterministic_baseline_response": baseline,
                "raw_pretrained_model_response": pretrained,
                "raw_trained_lora_response": trained,
                "final_guarded_response": final_guarded,
            }
        )

    metrics = {key: _metrics(cases, key) for key in METRIC_KEYS}
    return {
        "run_llm": args.run_llm,
        "response_model_name": args.response_model_name,
        "adapter_path": str(args.adapter_path),
        "adapter_available": adapter_available,
        "device_selection": "cuda_if_available_else_cpu",
        "headline": "final_guarded_response",
        "metrics": metrics,
        "cases": cases,
    }


def _format_metric(value: float | int | None, *, places: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{places}f}"


def _markdown_report(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    lines = [
        "# Response Generation Comparison",
        "",
        "Headline result: `final_guarded_response`, because this is what the user sees after validation and deterministic fallback.",
        "",
        "| Mode | Cases | Attempts | Accepted | Rejected | Fallback | Grounded before | Grounded after | JSON/debug before | Unsupported before | Avg latency (s) | Rejection reasons | Skipped |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for key in METRIC_KEYS:
        item = metrics[key]
        reasons = ", ".join(f"{name}: {count}" for name, count in item["rejection_reason_counts"].items()) or "-"
        lines.append(
            "| "
            + " | ".join(
                [
                    key,
                    str(item["case_count"]),
                    str(item["model_attempt_count"]),
                    str(item["accepted_model_output_count"]),
                    str(item["rejected_model_output_count"]),
                    str(item["fallback_count"]),
                    _format_metric(item["groundedness_before_fallback"]),
                    _format_metric(item["groundedness_after_fallback"]),
                    _format_metric(item["json_debug_leakage_before_fallback"]),
                    _format_metric(item["unsupported_claim_rate_before_fallback"]),
                    _format_metric(item["average_latency_seconds"], places=6),
                    reasons,
                    str(item["skipped_cases"]),
                ]
            )
            + " |"
        )
    if not payload.get("run_llm"):
        lines.extend(
            [
                "",
                "Local LLM loading was not requested. Re-run with `--run-llm` to populate raw pretrained and trained-adapter model attempts.",
            ]
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = evaluate(args)
    args.json_report.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_report.parent.mkdir(parents=True, exist_ok=True)
    args.json_report.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    args.markdown_report.write_text(_markdown_report(payload), encoding="utf-8")
    print(f"Saved response generation JSON report to {args.json_report}")
    print(f"Saved response generation Markdown report to {args.markdown_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
