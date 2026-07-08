"""Check for text leakage between slot-extraction train and evaluation data."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_FILE = ROOT / "data" / "training" / "slot_instruction_examples.jsonl"
DEFAULT_EVAL_FILE = ROOT / "data" / "evaluation" / "slot_eval_cases.jsonl"


def normalize_text(text: str) -> str:
    """Normalize user text for exact train/eval leakage checks."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load either a JSON array file or newline-delimited JSON records."""

    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return []
    if raw.lstrip().startswith("["):
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {path}")
        return [record for record in data if isinstance(record, dict)]

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        records.append(record)
    return records


def normalized_text_index(path: Path) -> dict[str, list[str]]:
    """Return normalized text -> original texts for one fixture file."""

    index: dict[str, list[str]] = {}
    for line_number, record in enumerate(load_records(path), start=1):
        text = str(record.get("text") or record.get("input") or "").strip()
        if not text:
            raise ValueError(f"Missing text/input at {path}:{line_number}")
        index.setdefault(normalize_text(text), []).append(text)
    return index


def find_leaked_texts(train_file: Path, eval_file: Path) -> list[dict[str, str]]:
    """Find texts that are exact duplicates after normalization."""

    train_index = normalized_text_index(train_file)
    eval_index = normalized_text_index(eval_file)
    leaked: list[dict[str, str]] = []
    for normalized, eval_texts in eval_index.items():
        train_texts = train_index.get(normalized)
        if not train_texts:
            continue
        leaked.append(
            {
                "normalized_text": normalized,
                "train_text": train_texts[0],
                "eval_text": eval_texts[0],
            }
        )
    return leaked


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check slot train/evaluation text leakage.")
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    leaked = find_leaked_texts(args.train_file, args.eval_file)
    if leaked:
        print("Found train/evaluation text leakage:", file=sys.stderr)
        for item in leaked:
            print(
                f"- {item['normalized_text']!r}\n"
                f"  train: {item['train_text']}\n"
                f"  eval:  {item['eval_text']}",
                file=sys.stderr,
            )
        return 1
    print(f"No normalized text leakage between {args.train_file} and {args.eval_file}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
