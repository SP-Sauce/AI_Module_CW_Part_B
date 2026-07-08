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
    return parser


def evaluate_slots(*, enable_llm: bool, slot_model_name: str) -> dict[str, Any]:
    with SLOT_FIXTURE.open("r", encoding="utf-8") as file:
        slot_cases = json.load(file)
    extractor = OptionalLLMSlotExtractor(slot_model_name) if enable_llm else RuleBasedSlotExtractor()
    intent_correct = 0
    slot_total = 0
    slot_correct = 0
    llm_used = 0
    details = []
    for case in slot_cases:
        result = extractor.extract(case["text"])
        llm_used += int(result.used_llm)
        intent_correct += int(result.intent == case["intent"])
        for key, expected in case["slots"].items():
            slot_total += 1
            slot_correct += int(result.slots.get(key) == expected)
        details.append(
            {
                "text": case["text"],
                "expected": case,
                "predicted": {
                    "intent": result.intent,
                    "slots": result.slots,
                    "used_llm": result.used_llm,
                    "errors": result.errors,
                },
            }
        )
    return {
        "intent_accuracy": round(intent_correct / len(slot_cases), 4),
        "slot_accuracy": round(slot_correct / slot_total, 4),
        "llm_enabled": enable_llm,
        "llm_used_cases": llm_used,
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


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    if args.model_name:
        settings = replace(settings, model_name=args.model_name)
    if args.slot_model_name:
        settings = replace(settings, slot_model_name=args.slot_model_name)
    if args.enable_llm:
        settings = replace(settings, enable_llm=True)
    restaurants = load_restaurants(settings, use_sample=args.sample_data)
    output = {
        "slot_extraction": evaluate_slots(enable_llm=settings.enable_llm, slot_model_name=settings.slot_model_name),
        "retrieval": evaluate_retrieval(restaurants),
        "response_generation": evaluate_response_safety(restaurants, settings, enable_llm=settings.enable_llm),
        "end_to_end": evaluate_end_to_end(args.sample_data, restaurants, settings, enable_llm=settings.enable_llm),
    }
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
