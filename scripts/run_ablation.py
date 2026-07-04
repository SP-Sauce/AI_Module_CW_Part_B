"""Simple ablation comparison for state and retrieval components."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from restaurant_assistant.assistant import RestaurantAssistant
from restaurant_assistant.data_loader import load_restaurants
from restaurant_assistant.dialogue_state import DialogueState
from restaurant_assistant.retrieval import RestaurantRetriever


SCENARIO = [
    "I need a cheap Italian restaurant in the south",
    "Can you book it for Friday at 7pm for 2 people?",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run simple ablation comparisons.")
    parser.add_argument("--sample-data", action="store_true", help="Use bundled sample restaurants.")
    return parser


def final_system(use_sample: bool) -> dict[str, Any]:
    assistant = RestaurantAssistant(use_sample=use_sample)
    responses = [assistant.process(turn).response for turn in SCENARIO]
    return {"success": "simulated reference" in responses[-1].lower(), "responses": responses}


def retrieval_only(restaurants: list[dict[str, Any]]) -> dict[str, Any]:
    retriever = RestaurantRetriever().fit(restaurants)
    state = DialogueState(food="italian", area="south", pricerange="cheap")
    results = retriever.search(SCENARIO[0], state, top_k=1)
    success = bool(results and results[0].record.get("food_norm") == "italian")
    return {"success": success, "top_result": results[0].record.get("name") if results else None}


def no_state_tracking(use_sample: bool) -> dict[str, Any]:
    responses = []
    for turn in SCENARIO:
        assistant = RestaurantAssistant(use_sample=use_sample)
        responses.append(assistant.process(turn).response)
    return {"success": "simulated reference" in responses[-1].lower(), "responses": responses}


def llm_only_placeholder() -> dict[str, Any]:
    return {
        "success": False,
        "note": "No retrieval evidence or state is available, so the safe MVP does not allow LLM-only booking claims.",
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    restaurants = load_restaurants(use_sample=args.sample_data)
    rows = {
        "final_system": final_system(args.sample_data),
        "retrieval_only": retrieval_only(restaurants),
        "no_state_tracking": no_state_tracking(args.sample_data),
        "llm_only_baseline": llm_only_placeholder(),
    }
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

