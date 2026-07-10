"""Compare baseline, optional LLM, and final guarded response generation."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
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
    validate_generated_response,
)
from restaurant_assistant.nlg import contains_json_or_debug_leakage
from restaurant_assistant.ranking import RankedRestaurant

try:
    from .build_response_training_data import build_rows
except ImportError:
    from build_response_training_data import build_rows


DEFAULT_EVAL_FILE = ROOT / "data" / "evaluation" / "response_generation_eval.jsonl"
DEFAULT_JSON_REPORT = ROOT / "reports" / "response_generation_comparison.json"
DEFAULT_MD_REPORT = ROOT / "reports" / "response_generation_comparison.md"
DEFAULT_ADAPTER_PATH = ROOT / "models" / "response-generator-lora"


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


def _first_restaurant_from_output(output: str, restaurants: list[dict[str, Any]]) -> dict[str, Any] | None:
    lowered = output.casefold()
    for record in restaurants:
        name = str(record.get("name") or "").casefold()
        if name and name in lowered:
            return record
    return None


def _intent_from_input(input_text: str) -> str:
    for line in input_text.splitlines():
        if line.startswith("Intent:"):
            return line.split(":", 1)[1].strip()
    return "search"


def _user_from_input(input_text: str) -> str:
    for line in input_text.splitlines():
        if line.startswith("User:"):
            return line.split(":", 1)[1].strip()
    return ""


def _missing_slots_from_input(input_text: str) -> list[str]:
    for line in input_text.splitlines():
        if line.startswith("Missing slots:"):
            raw = line.split(":", 1)[1].strip()
            try:
                parsed = json.loads(raw.replace("'", '"'))
                return [str(item) for item in parsed] if isinstance(parsed, list) else []
            except json.JSONDecodeError:
                return []
    return []


def _evidence_records_from_input(input_text: str) -> list[dict[str, str]]:
    for line in input_text.splitlines():
        if not line.startswith("Evidence:"):
            continue
        raw = line.split(":", 1)[1].strip()
        records: list[dict[str, str]] = []
        for chunk in raw.split("|"):
            record: dict[str, str] = {}
            for part in chunk.split(";"):
                if "=" not in part:
                    continue
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key and value:
                    record[key] = value
            if record:
                records.append(record)
        return records
    return []


def _state_for_record(record: dict[str, Any] | None) -> DialogueState:
    if not record:
        return DialogueState()
    return DialogueState(
        food=record.get("food"),
        area=record.get("area"),
        pricerange=record.get("pricerange"),
        selected_restaurant=record,
    )


def _ranked(record: dict[str, Any] | None) -> list[RankedRestaurant]:
    if not record:
        return []
    return [
        RankedRestaurant(
            record=record,
            score=1.0,
            matched_constraints=["food", "area", "pricerange"],
            missing_unmatched_constraints=[],
            explanation="response generation evaluation evidence",
            similarity=1.0,
        )
    ]


def _clarity_pass(text: str) -> bool:
    stripped = " ".join(str(text or "").split())
    if not stripped:
        return False
    if len(stripped.split()) > 120:
        return False
    return stripped[-1] in ".!?"


def _exact_evidence_preserved(text: str, record: dict[str, Any] | None) -> bool:
    if not record:
        return True
    lowered = text.casefold()
    contact_values = [
        str(record.get("phone") or "").casefold(),
        str(record.get("postcode") or "").casefold(),
        str(record.get("address") or "").casefold(),
    ]
    mentioned_any = any(value and value in lowered for value in contact_values)
    if not mentioned_any:
        return True
    return all(value in lowered for value in contact_values if value and value in lowered)


def _metrics(cases: list[dict[str, Any]], key: str) -> dict[str, Any]:
    latencies = [case[key].get("latency_seconds", 0.0) for case in cases]
    count = len(cases)
    grounded = 0
    leakage = 0
    unsupported = 0
    nonempty = 0
    clarity = 0
    exact = 0
    fallback = 0
    skipped = 0
    for case in cases:
        result = case[key]
        text = result.get("text", "")
        validation = validate_generated_response(
            text,
            evidence_records=case.get("evidence_records", []),
            known_restaurant_records=case.get("known_restaurants", []),
        )
        grounded += int(validation.ok)
        leakage += int(contains_json_or_debug_leakage(text))
        unsupported += int(validation.reason == "unsupported_claim")
        nonempty += int(bool(str(text).strip()))
        clarity += int(_clarity_pass(text))
        exact += int(_exact_evidence_preserved(text, case.get("record")))
        fallback += int(bool(result.get("fallback")))
        skipped += int(bool(result.get("skipped")))
    denominator = max(count, 1)
    return {
        "case_count": count,
        "groundedness_rate": round(grounded / denominator, 4),
        "json_debug_leakage_rate": round(leakage / denominator, 4),
        "unsupported_claim_rate": round(unsupported / denominator, 4),
        "response_nonempty_rate": round(nonempty / denominator, 4),
        "fallback_rate": round(fallback / denominator, 4),
        "average_latency_seconds": round(statistics.mean(latencies), 6) if latencies else 0.0,
        "simple_clarity_check_pass_rate": round(clarity / denominator, 4),
        "exact_evidence_preservation_rate": round(exact / denominator, 4),
        "skipped_cases": skipped,
    }


def _result(text: str, *, latency: float = 0.0, fallback: bool = False, skipped: bool = False, mode: str = "") -> dict[str, Any]:
    return {
        "text": text,
        "latency_seconds": round(latency, 6),
        "fallback": fallback,
        "skipped": skipped,
        "mode": mode,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    settings = get_settings()
    restaurants = load_restaurants(settings, use_sample=args.sample_data)
    rows = _load_rows(args.eval_file, restaurants)
    cases: list[dict[str, Any]] = []

    pretrained_generator = GroundedResponseGenerator(
        enable_llm=args.run_llm,
        model_name=args.response_model_name,
        known_restaurants=restaurants,
    )
    trained_generator = GroundedResponseGenerator(
        enable_llm=args.run_llm and args.adapter_path.exists(),
        model_name=str(args.adapter_path),
        known_restaurants=restaurants,
    )
    final_generator = GroundedResponseGenerator(
        enable_llm=args.run_llm,
        model_name=str(args.adapter_path) if args.adapter_path.exists() else args.response_model_name,
        known_restaurants=restaurants,
    )

    for row in rows:
        input_text = str(row.get("input") or "")
        baseline_text = str(row.get("output") or "")
        evidence_records = _evidence_records_from_input(input_text)
        record = evidence_records[0] if evidence_records else _first_restaurant_from_output(baseline_text, restaurants)
        intent = _intent_from_input(input_text)
        user = _user_from_input(input_text)
        missing_slots = _missing_slots_from_input(input_text)
        state = _state_for_record(record)
        ranked = _ranked(record)
        case: dict[str, Any] = {
            "intent": intent,
            "user": user,
            "record": record,
            "evidence_records": evidence_records or ([record] if record else []),
            "known_restaurants": [*restaurants, *evidence_records],
            "baseline_template": _result(baseline_text, mode="baseline_template"),
        }

        if args.run_llm:
            start = time.perf_counter()
            pretrained = pretrained_generator.generate(
                user,
                state,
                ranked,
                intent=intent,
                missing_slots=missing_slots,
                baseline_text=baseline_text,
            )
            case["pretrained_flan_t5_base_response"] = _result(
                pretrained.text,
                latency=time.perf_counter() - start,
                fallback=not pretrained.used_llm,
                mode=pretrained.final_response_mode,
            )
        else:
            case["pretrained_flan_t5_base_response"] = _result(
                "",
                skipped=True,
                mode="not_run",
            )

        if args.run_llm and args.adapter_path.exists():
            start = time.perf_counter()
            trained = trained_generator.generate(
                user,
                state,
                ranked,
                intent=intent,
                missing_slots=missing_slots,
                baseline_text=baseline_text,
            )
            case["trained_lora_response"] = _result(
                trained.text,
                latency=time.perf_counter() - start,
                fallback=not trained.used_llm,
                mode=trained.final_response_mode,
            )
        else:
            case["trained_lora_response"] = _result(
                "",
                skipped=True,
                mode="adapter_missing" if not args.adapter_path.exists() else "not_run",
            )

        if args.run_llm:
            start = time.perf_counter()
            final = final_generator.generate(
                user,
                state,
                ranked,
                intent=intent,
                missing_slots=missing_slots,
                baseline_text=baseline_text,
            )
            case["final_guarded_response"] = _result(
                final.text,
                latency=time.perf_counter() - start,
                fallback=not final.used_llm,
                mode=final.final_response_mode,
            )
        else:
            case["final_guarded_response"] = _result(
                baseline_text,
                mode="baseline_template",
            )
        cases.append(case)

    metrics = {
        "baseline_template": _metrics(cases, "baseline_template"),
        "pretrained_flan_t5_base_response": _metrics(cases, "pretrained_flan_t5_base_response"),
        "trained_lora_response": _metrics(cases, "trained_lora_response"),
        "final_guarded_response": _metrics(cases, "final_guarded_response"),
    }
    return {
        "run_llm": args.run_llm,
        "response_model_name": args.response_model_name,
        "adapter_path": str(args.adapter_path),
        "headline": "final_guarded_response",
        "metrics": metrics,
        "cases": cases,
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    lines = [
        "# Response Generation Comparison",
        "",
        "Headline result: `final_guarded_response`, because this is what the user sees.",
        "",
        "| Mode | Cases | Groundedness | JSON/debug leakage | Unsupported claims | Nonempty | Fallback | Avg latency (s) | Clarity | Evidence preservation | Skipped |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in [
        "baseline_template",
        "pretrained_flan_t5_base_response",
        "trained_lora_response",
        "final_guarded_response",
    ]:
        item = metrics[key]
        lines.append(
            "| "
            + " | ".join(
                [
                    key,
                    str(item["case_count"]),
                    f"{item['groundedness_rate']:.4f}",
                    f"{item['json_debug_leakage_rate']:.4f}",
                    f"{item['unsupported_claim_rate']:.4f}",
                    f"{item['response_nonempty_rate']:.4f}",
                    f"{item['fallback_rate']:.4f}",
                    f"{item['average_latency_seconds']:.6f}",
                    f"{item['simple_clarity_check_pass_rate']:.4f}",
                    f"{item['exact_evidence_preservation_rate']:.4f}",
                    str(item["skipped_cases"]),
                ]
            )
            + " |"
        )
    if not payload.get("run_llm"):
        lines.extend(
            [
                "",
                "Local LLM loading was not requested. Re-run with `--run-llm` to populate the pretrained and trained adapter columns.",
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
