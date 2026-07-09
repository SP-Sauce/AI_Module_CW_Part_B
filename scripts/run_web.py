"""Entrypoint for the browser-based restaurant assistant."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from restaurant_assistant.config import get_settings
from restaurant_assistant.web_app import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MultiWOZ restaurant assistant web app.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind.")
    parser.add_argument("--sample-data", action="store_true", help="Use bundled sample restaurants.")
    parser.add_argument("--enable-llm", action="store_true", help="Attempt Hugging Face Transformers generation.")
    parser.add_argument("--model-name", default=None, help="Transformers model name for grounded response generation.")
    parser.add_argument("--slot-model-name", default=None, help="Transformers model name or adapter path for LLM slot extraction.")
    parser.add_argument("--slot-num-beams", type=int, default=None, help="Beam count for slot-model JSON generation.")
    parser.add_argument("--debug", action="store_true", help="Run Flask in debug mode.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    if args.model_name:
        settings = replace(settings, model_name=args.model_name)
    if args.slot_model_name:
        settings = replace(settings, slot_model_name=args.slot_model_name)
    if args.slot_num_beams is not None:
        settings = replace(settings, slot_num_beams=args.slot_num_beams)
    if args.enable_llm:
        settings = replace(settings, enable_llm=True)
    app = create_app(
        settings=settings,
        use_sample=args.sample_data,
        enable_llm=settings.enable_llm,
        debug_turns=args.debug,
    )
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
