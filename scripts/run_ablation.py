"""Simple ablation comparison for state and retrieval components."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from restaurant_assistant.assistant import RestaurantAssistant
from restaurant_assistant.config import get_settings
from restaurant_assistant.data_loader import load_restaurants
from restaurant_assistant.dialogue_state import DialogueState
from restaurant_assistant.retrieval import RestaurantRetriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run simple ablation comparisons.")
    parser.add_argument("--sample-data", action="store_true", help="Use bundled sample restaurants.")
    parser.add_argument("--enable-llm", action="store_true", help="Run the final system with LLM extraction/generation enabled.")
    parser.add_argument("--model-name", default=None, help="Transformers model name for grounded response generation.")
    parser.add_argument("--slot-model-name", default=None, help="Transformers model name or adapter path for LLM slot extraction.")
    return parser


def scenario_from_target(record: dict[str, Any]) -> list[str]:
    query = f"I need a {record.get('pricerange')} {record.get('food')} restaurant in the {record.get('area')}"
    return [query, "Can you book it for Friday at 7pm for 2 people?"]


def final_system(use_sample: bool, settings: Any, scenario: list[str], *, enable_llm: bool) -> dict[str, Any]:
    assistant = RestaurantAssistant(settings=settings, use_sample=use_sample, enable_llm=enable_llm)
    results = [assistant.process(turn, debug=True) for turn in scenario]
    return {
        "success": "your reference" in results[-1].response.lower(),
        "llm_slot_turns": sum(int(result.debug.get("slot_extraction_used_llm") is True) for result in results),
        "generation_modes": [result.debug.get("generation_mode") for result in results],
        "responses": [result.response for result in results],
    }


def retrieval_only(restaurants: list[dict[str, Any]], target: dict[str, Any], scenario: list[str]) -> dict[str, Any]:
    retriever = RestaurantRetriever().fit(restaurants)
    state = DialogueState(food=target.get("food"), area=target.get("area"), pricerange=target.get("pricerange"))
    results = retriever.search(scenario[0], state, top_k=1)
    success = bool(results and results[0].record.get("name") == target.get("name"))
    return {"success": success, "top_result": results[0].record.get("name") if results else None}


def no_state_tracking(use_sample: bool, settings: Any, scenario: list[str]) -> dict[str, Any]:
    responses = []
    for turn in scenario:
        assistant = RestaurantAssistant(settings=settings, use_sample=use_sample, enable_llm=False)
        responses.append(assistant.process(turn).response)
    return {"success": "your reference" in responses[-1].lower(), "responses": responses}


def llm_only_placeholder() -> dict[str, Any]:
    return {
        "success": False,
        "note": "No retrieval evidence or state is available, so the safe MVP does not allow LLM-only booking claims.",
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
    target = next(
        (
            record
            for record in restaurants
            if record.get("food") and record.get("area") and record.get("pricerange")
        ),
        restaurants[0],
    )
    scenario = scenario_from_target(target)
    rows = {
        "scenario": scenario,
        "target_restaurant": target.get("name"),
        "final_system": final_system(args.sample_data, settings, scenario, enable_llm=settings.enable_llm),
        "retrieval_only": retrieval_only(restaurants, target, scenario),
        "no_state_tracking": no_state_tracking(args.sample_data, settings, scenario),
        "llm_only_baseline": llm_only_placeholder(),
    }
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
