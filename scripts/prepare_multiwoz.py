"""Prepare cleaned MultiWOZ restaurant records."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from restaurant_assistant.config import get_settings
from restaurant_assistant.data_loader import load_sample_restaurants
from restaurant_assistant.preprocessing import load_multiwoz_restaurant_db, preprocess_restaurants, save_jsonl


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Create cleaned restaurant records from MultiWOZ.")
    parser.add_argument("--multiwoz-path", type=Path, default=settings.raw_multiwoz_path)
    parser.add_argument("--output", type=Path, default=settings.processed_restaurant_path)
    parser.add_argument("--sample-if-missing", action="store_true", help="Use bundled sample data if MultiWOZ is absent.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    try:
        raw_records = load_multiwoz_restaurant_db(args.multiwoz_path)
        source = f"MultiWOZ restaurant_db.json under {args.multiwoz_path}"
    except FileNotFoundError:
        if not args.sample_if_missing:
            raise
        raw_records = load_sample_restaurants(settings)
        source = "bundled sample restaurants"
    cleaned = preprocess_restaurants(raw_records)
    save_jsonl(cleaned, args.output)
    print(f"Saved {len(cleaned)} restaurant records to {args.output}")
    print(f"Source: {source}")


if __name__ == "__main__":
    main()

