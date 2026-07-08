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
    parser.add_argument("--enable-llm", action="store_true", help="Attempt Hugging Face Transformers generation.")
    parser.add_argument("--model-name", default=None, help="Transformers model name for optional LLM mode.")
    parser.add_argument("--slot-model-name", default=None, help="Transformers model name or adapter path for LLM slot extraction.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    if args.model_name:
        settings = replace(settings, model_name=args.model_name)
    if args.slot_model_name:
        settings = replace(settings, slot_model_name=args.slot_model_name)
    if args.enable_llm:
        settings = replace(settings, enable_llm=True)
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
