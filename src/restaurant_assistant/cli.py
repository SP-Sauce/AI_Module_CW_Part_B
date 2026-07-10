"""Command line chat interface."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

from restaurant_assistant.config import get_settings
from restaurant_assistant.assistant import RestaurantAssistant


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MultiWOZ restaurant assistant CLI.")
    parser.add_argument("--sample-data", action="store_true", help="Use bundled sample restaurants.")
    parser.add_argument("--debug", action="store_true", help="Print extraction, state, retrieval and ranking details.")
    parser.add_argument("--enable-llm", action="store_true", help="Enable the trained LLM slot extractor.")
    parser.add_argument("--enable-response-llm", action="store_true", help="Enable optional guarded LLM response generation.")
    parser.add_argument(
        "--response-model-name",
        default=None,
        help="Transformers model name or LoRA adapter path for optional response generation.",
    )
    parser.add_argument("--model-name", default=None, help="Deprecated alias for --response-model-name.")
    parser.add_argument("--slot-model-name", default=None, help="Transformers model name or adapter path for LLM slot extraction.")
    parser.add_argument("--slot-num-beams", type=int, default=None, help="Beam count for slot-model JSON generation.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    response_model_name = args.response_model_name or args.model_name
    if response_model_name:
        settings = replace(
            settings,
            model_name=response_model_name,
            response_model_name=response_model_name,
        )
    if args.slot_model_name:
        settings = replace(settings, slot_model_name=args.slot_model_name)
    if args.slot_num_beams is not None:
        settings = replace(settings, slot_num_beams=args.slot_num_beams)
    if args.enable_llm:
        settings = replace(settings, enable_llm=True)
    if args.enable_response_llm:
        settings = replace(settings, enable_response_llm=True)
    assistant = RestaurantAssistant(settings=settings, use_sample=args.sample_data, enable_llm=settings.enable_llm)

    print("MultiWOZ restaurant assistant. Type 'exit' to quit.")
    while True:
        try:
            user_message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_message.lower() in {"exit", "quit"}:
            break
        if not user_message:
            continue
        result = assistant.process(user_message, debug=args.debug)
        print(f"Assistant: {result.response}")
        if args.debug:
            print(json.dumps(result.debug, indent=2, default=str))


if __name__ == "__main__":
    main()
