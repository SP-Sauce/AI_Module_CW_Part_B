"""Evaluate slot extraction, retrieval and end-to-end dialogue behavior."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from restaurant_assistant.assistant import RestaurantAssistant
from restaurant_assistant.config import get_settings
from restaurant_assistant.data_loader import load_restaurants
from restaurant_assistant.dialogue_state import DialogueState
from restaurant_assistant.ranking import rank_candidates
from restaurant_assistant.retrieval import RestaurantRetriever
from restaurant_assistant.slot_extraction import OptionalLLMSlotExtractor, RuleBasedSlotExtractor


SLOT_FIXTURE = ROOT / "tests" / "fixtures" / "slot_cases.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the restaurant assistant.")
    parser.add_argument("--sample-data", action="store_true", help="Use bundled sample restaurants.")
    parser.add_argument("--enable-llm", action="store_true", help="Evaluate with LLM extraction and generation enabled.")
    parser.add_argument("--model-name", default=None, help="Transformers model name for grounded response generation.")
    parser.add_argument("--slot-model-name", default=None, help="Transformers model name or adapter path for LLM slot extraction.")
    parser.add_argument(
        "--slot-fixture",
        type=Path,
        default=SLOT_FIXTURE,
        help="Labelled slot fixture as JSON array or JSONL records.",
    )
    return parser


def load_slot_cases(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return []
    if raw.lstrip().startswith("["):
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {path}")
        return [case for case in data if isinstance(case, dict)]

    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        case = json.loads(line)
        if not isinstance(case, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        cases.append(case)
    return cases


def _canonical_slot_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _canonical_slot_value(item)) for key, item in value.items()))
    if isinstance(value, list):
        return tuple(_canonical_slot_value(item) for item in value)
    return value


def _slot_pairs(slots: dict[str, Any]) -> set[tuple[str, Any]]:
    return {(key, _canonical_slot_value(value)) for key, value in slots.items()}


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _has_json_or_parse_error(errors: list[str]) -> bool:
    return any("json" in error.lower() or "parse" in error.lower() for error in errors)


def _raw_output_preview(raw_output: str | None, limit: int = 300) -> str | None:
    if raw_output is None:
        return None
    preview = " ".join(raw_output.split())
    return preview if len(preview) <= limit else preview[: limit - 3] + "..."


def evaluate_slots(*, enable_llm: bool, slot_model_name: str, slot_fixture: Path) -> dict[str, Any]:
    slot_cases = load_slot_cases(slot_fixture)
    if not slot_cases:
        raise ValueError(f"No slot evaluation cases found in {slot_fixture}")
    extractor = OptionalLLMSlotExtractor(slot_model_name) if enable_llm else RuleBasedSlotExtractor()
    intent_correct = 0
    exact_slot_objects = 0
    expected_slot_total = 0
    expected_slot_correct = 0
    true_positive_slots = 0
    false_positive_slots = 0
    false_negative_slots = 0
    invalid_json_or_parse_errors = 0
    llm_used = 0
    llm_attempted = 0
    llm_parse_success = 0
    fallback_used = 0
    latencies = []
    details = []
    for case in slot_cases:
        start = time.perf_counter()
        result = extractor.extract(case["text"])
        latency = time.perf_counter() - start
        latencies.append(latency)
        llm_used += int(result.used_llm)
        llm_attempted += int(result.llm_attempted)
        llm_parse_success += int(result.llm_parse_success)
        fallback_used += int(enable_llm and not result.used_llm)
        intent_correct += int(result.intent == case["intent"])
        expected_slots = case.get("slots", {})
        predicted_slots = result.slots
        exact_slot_objects += int(predicted_slots == expected_slots)
        expected_pairs = _slot_pairs(expected_slots)
        predicted_pairs = _slot_pairs(predicted_slots)
        matched_pairs = predicted_pairs & expected_pairs
        expected_slot_total += len(expected_pairs)
        expected_slot_correct += len(matched_pairs)
        true_positive_slots += len(matched_pairs)
        false_positive_slots += len(predicted_pairs - expected_pairs)
        false_negative_slots += len(expected_pairs - predicted_pairs)
        invalid_json_or_parse_errors += int(_has_json_or_parse_error(result.errors))
        details.append(
            {
                "text": case["text"],
                "expected": case,
                "predicted": {
                    "intent": result.intent,
                    "slots": result.slots,
                    "used_llm": result.used_llm,
                    "llm_attempted": result.llm_attempted,
                    "llm_parse_success": result.llm_parse_success,
                    "llm_raw_output": _raw_output_preview(result.llm_raw_output),
                    "errors": result.errors,
                },
                "latency_seconds": round(latency, 6),
            }
        )
    slot_precision = _safe_divide(true_positive_slots, true_positive_slots + false_positive_slots)
    slot_recall = _safe_divide(true_positive_slots, true_positive_slots + false_negative_slots)
    slot_f1 = (
        round(2 * slot_precision * slot_recall / (slot_precision + slot_recall), 4)
        if slot_precision + slot_recall
        else 0.0
    )
    return {
        "fixture": str(slot_fixture),
        "case_count": len(slot_cases),
        "intent_accuracy": round(intent_correct / len(slot_cases), 4),
        "slot_accuracy": _safe_divide(expected_slot_correct, expected_slot_total),
        "exact_slot_object_accuracy": round(exact_slot_objects / len(slot_cases), 4),
        "slot_precision": slot_precision,
        "slot_recall": slot_recall,
        "slot_f1": slot_f1,
        "invalid_json_or_parse_error_count": invalid_json_or_parse_errors,
        "mean_slot_latency_seconds": round(statistics.mean(latencies), 6),
        "llm_enabled": enable_llm,
        "llm_used_cases": llm_used,
        "llm_attempted_cases": llm_attempted,
        "llm_parse_success_cases": llm_parse_success,
        "llm_parse_failed_cases": llm_attempted - llm_parse_success,
        "fallback_used_cases": fallback_used,
        "slot_model_name": slot_model_name if enable_llm else None,
        "cases": details,
    }


def evaluate_retrieval(restaurants: list[dict[str, Any]]) -> dict[str, float]:
    retriever = RestaurantRetriever().fit(restaurants)
    reciprocal_ranks = []
    recall_at_1 = 0
    recall_at_3 = 0
    for record in restaurants:
        query = f"{record.get('pricerange', '')} {record.get('food', '')} restaurant in {record.get('area', '')}"
        state = DialogueState(food=record.get("food"), area=record.get("area"), pricerange=record.get("pricerange"))
        results = retriever.search(query, state, top_k=3)
        names = [item.record.get("name") for item in results]
        target = record.get("name")
        if names and names[0] == target:
            recall_at_1 += 1
        if target in names:
            recall_at_3 += 1
            reciprocal_ranks.append(1 / (names.index(target) + 1))
        else:
            reciprocal_ranks.append(0.0)
    total = max(len(restaurants), 1)
    return {
        "recall_at_1": round(recall_at_1 / total, 4),
        "recall_at_3": round(recall_at_3 / total, 4),
        "mrr": round(sum(reciprocal_ranks) / total, 4),
    }


def _query_from_record(record: dict[str, Any]) -> str:
    return f"I need a {record.get('pricerange')} {record.get('food')} restaurant in the {record.get('area')}"


def evaluate_end_to_end(use_sample: bool, restaurants: list[dict[str, Any]], settings: Any, *, enable_llm: bool) -> dict[str, Any]:
    target = next(
        (
            record
            for record in restaurants
            if record.get("food") and record.get("area") and record.get("pricerange")
        ),
        restaurants[0],
    )
    scripts = [
        {
            "name": "search_book_reschedule_cancel",
            "turns": [
                _query_from_record(target),
                "Can you book it for Friday at 7pm for 2 people?",
                "Move it to Saturday",
                "Cancel it",
            ],
            "success_terms": ["your reference", "updated booking", "cancelled booking"],
        },
        {
            "name": "clarify_missing_search",
            "turns": ["I need a restaurant"],
            "success_terms": ["Please tell me"],
        },
    ]
    successes = 0
    latencies = []
    transcript = []
    for script in scripts:
        assistant = RestaurantAssistant(settings=settings, use_sample=use_sample, enable_llm=enable_llm)
        responses = []
        llm_slot_turns = 0
        generation_modes = []
        for turn in script["turns"]:
            start = time.perf_counter()
            result = assistant.process(turn, debug=True)
            latencies.append(time.perf_counter() - start)
            responses.append(result.response)
            llm_slot_turns += int(result.debug.get("slot_extraction_used_llm") is True)
            generation_modes.append(result.debug.get("generation_mode"))
        combined = " ".join(responses).lower()
        success = all(term.lower() in combined for term in script["success_terms"])
        successes += int(success)
        transcript.append(
            {
                "name": script["name"],
                "success": success,
                "llm_slot_turns": llm_slot_turns,
                "generation_modes": generation_modes,
                "responses": responses,
            }
        )
    return {
        "task_success_rate": round(successes / len(scripts), 4),
        "latency_seconds_mean": round(statistics.mean(latencies), 4),
        "latency_seconds_max": round(max(latencies), 4),
        "transcript": transcript,
    }


def evaluate_response_safety(restaurants: list[dict[str, Any]], settings: Any, *, enable_llm: bool) -> dict[str, Any]:
    assistant = RestaurantAssistant(restaurants=restaurants, settings=settings, enable_llm=enable_llm)
    target = next(
        (
            record
            for record in restaurants
            if record.get("food") and record.get("area") and record.get("pricerange")
        ),
        restaurants[0],
    )
    result = assistant.process(_query_from_record(target), debug=True)
    forbidden = ["live availability", "payment", "verified halal", "verified vegetarian"]
    lower = result.response.lower()
    return {
        "fallback_grounded_check": str(target.get("name", "")).lower() in lower,
        "forbidden_claims_present": [term for term in forbidden if term in lower],
        "slot_extraction_used_llm": result.debug.get("slot_extraction_used_llm"),
        "generation_mode": result.debug.get("generation_mode"),
        "optional_metrics": "BLEU/ROUGE/BERTScore hooks not installed for MVP",
    }


def run_evaluation(
    *,
    sample_data: bool,
    enable_llm: bool,
    model_name: str | None = None,
    slot_model_name: str | None = None,
    slot_fixture: Path = SLOT_FIXTURE,
) -> dict[str, Any]:
    settings = get_settings()
    if model_name:
        settings = replace(settings, model_name=model_name)
    if slot_model_name:
        settings = replace(settings, slot_model_name=slot_model_name)
    if enable_llm:
        settings = replace(settings, enable_llm=True)
    restaurants = load_restaurants(settings, use_sample=sample_data)
    return {
        "slot_extraction": evaluate_slots(
            enable_llm=settings.enable_llm,
            slot_model_name=settings.slot_model_name,
            slot_fixture=slot_fixture,
        ),
        "retrieval": evaluate_retrieval(restaurants),
        "response_generation": evaluate_response_safety(restaurants, settings, enable_llm=settings.enable_llm),
        "end_to_end": evaluate_end_to_end(sample_data, restaurants, settings, enable_llm=settings.enable_llm),
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output = run_evaluation(
        sample_data=args.sample_data,
        enable_llm=args.enable_llm,
        model_name=args.model_name,
        slot_model_name=args.slot_model_name,
        slot_fixture=args.slot_fixture,
    )
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
