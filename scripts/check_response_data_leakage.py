"""Check response-generation JSONL splits for leakage and schema issues."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_FILE = ROOT / "data" / "training" / "response_generation_examples.jsonl"
DEFAULT_EVAL_FILE = ROOT / "data" / "evaluation" / "response_generation_eval.jsonl"
DEFAULT_CHALLENGE_FILE = ROOT / "data" / "evaluation" / "response_generation_challenge.jsonl"
DEFAULT_REPORT_FILE = ROOT / "reports" / "response_generation_dataset_report.json"
REQUIRED_FIELDS = {"instruction", "input", "output"}
BOOKING_REF_RE = re.compile(r"\bBK-[A-Z0-9]{6}\b")


@dataclass
class SplitRows:
    name: str
    path: Path
    rows: list[dict[str, str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check response-generation dataset leakage.")
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--challenge-file", type=Path, default=DEFAULT_CHALLENGE_FILE)
    parser.add_argument("--dataset-report", type=Path, default=DEFAULT_REPORT_FILE)
    return parser


def normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).casefold())).strip()


def row_key(row: dict[str, str]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def user_text(input_text: str) -> str:
    for line in input_text.splitlines():
        if line.startswith("User:"):
            return line.split(":", 1)[1].strip()
    return ""


def evidence_restaurant_ids(input_text: str) -> set[str]:
    ids: set[str] = set()
    for line in input_text.splitlines():
        if not line.startswith("Evidence:"):
            continue
        raw = line.split(":", 1)[1].strip()
        for chunk in raw.split("|"):
            fields: dict[str, str] = {}
            for part in chunk.split(";"):
                if "=" not in part:
                    continue
                key, value = part.split("=", 1)
                fields[key.strip()] = value.strip()
            if fields.get("name"):
                ids.add(
                    normalise_text(
                        "|".join(
                            [
                                fields.get("name", ""),
                                fields.get("area", ""),
                                fields.get("food", ""),
                                fields.get("pricerange", ""),
                            ]
                        )
                    )
                )
    return ids


def duplicate_count(values: list[str]) -> int:
    counts = Counter(values)
    return sum(count - 1 for count in counts.values() if count > 1)


def load_split(name: str, path: Path) -> SplitRows:
    split = SplitRows(name=name, path=path)
    if not path.exists():
        split.errors.append(f"{name}: file not found: {path}")
        return split
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                split.errors.append(f"{name}:{line_number}: invalid JSON: {exc}")
                continue
            if not isinstance(raw, dict):
                split.errors.append(f"{name}:{line_number}: row is not a JSON object")
                continue
            if set(raw) != REQUIRED_FIELDS:
                split.errors.append(f"{name}:{line_number}: expected fields {sorted(REQUIRED_FIELDS)}, got {sorted(raw)}")
                continue
            row = {key: str(raw.get(key) or "").strip() for key in REQUIRED_FIELDS}
            for key in REQUIRED_FIELDS:
                if not row[key]:
                    split.errors.append(f"{name}:{line_number}: empty {key}")
            split.rows.append({"instruction": row["instruction"], "input": row["input"], "output": row["output"]})
    return split


def _overlap(first: list[str], second: list[str]) -> int:
    return len(set(first) & set(second))


def _load_disjoint_status(report_path: Path) -> bool | None:
    if not report_path.exists():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    value = report.get("restaurant_level_disjoint_split_achieved")
    return value if isinstance(value, bool) else None


def check_splits(
    *,
    train_file: Path = DEFAULT_TRAIN_FILE,
    eval_file: Path = DEFAULT_EVAL_FILE,
    challenge_file: Path = DEFAULT_CHALLENGE_FILE,
    dataset_report: Path = DEFAULT_REPORT_FILE,
) -> dict[str, Any]:
    splits = {
        "train": load_split("train", train_file),
        "eval": load_split("eval", eval_file),
        "challenge": load_split("challenge", challenge_file),
    }
    errors: list[str] = []
    limitations: list[str] = []
    for split in splits.values():
        errors.extend(split.errors)

    metrics: dict[str, Any] = {"splits": {}, "overlaps": {}}
    for name, split in splits.items():
        rows = split.rows
        row_keys = [row_key(row) for row in rows]
        input_values = [row["input"] for row in rows]
        input_output_pairs = [row["input"] + "\n" + row["output"] for row in rows]
        users = [normalise_text(user_text(row["input"])) for row in rows]
        missing_booking_refs = []
        for index, row in enumerate(rows, start=1):
            output_refs = set(BOOKING_REF_RE.findall(row["output"]))
            input_refs = set(BOOKING_REF_RE.findall(row["input"]))
            missing = sorted(output_refs - input_refs)
            if missing:
                missing_booking_refs.append({"line": index, "missing_references": missing})
        metrics["splits"][name] = {
            "row_count": len(rows),
            "duplicate_rows": duplicate_count(row_keys),
            "duplicate_inputs": duplicate_count(input_values),
            "duplicate_input_output_pairs": duplicate_count(input_output_pairs),
            "duplicate_normalised_user_messages": duplicate_count(users),
            "booking_reference_grounding_failures": len(missing_booking_refs),
            "restaurant_ids": sorted(set().union(*(evidence_restaurant_ids(row["input"]) for row in rows)) if rows else set()),
        }
        if metrics["splits"][name]["duplicate_rows"]:
            errors.append(f"{name}: duplicate rows detected")
        if metrics["splits"][name]["duplicate_input_output_pairs"]:
            errors.append(f"{name}: duplicate input/output pairs detected")
        if metrics["splits"][name]["booking_reference_grounding_failures"]:
            errors.append(f"{name}: booking references found in output but missing from input")

    pairs = [("train", "eval"), ("train", "challenge"), ("eval", "challenge")]
    for first, second in pairs:
        first_rows = splits[first].rows
        second_rows = splits[second].rows
        metrics["overlaps"][f"{first}_{second}_normalised_user_overlap"] = _overlap(
            [normalise_text(user_text(row["input"])) for row in first_rows],
            [normalise_text(user_text(row["input"])) for row in second_rows],
        )
        metrics["overlaps"][f"{first}_{second}_input_overlap"] = _overlap(
            [row["input"] for row in first_rows],
            [row["input"] for row in second_rows],
        )
        metrics["overlaps"][f"{first}_{second}_row_overlap"] = _overlap(
            [row_key(row) for row in first_rows],
            [row_key(row) for row in second_rows],
        )
        first_restaurants = set(metrics["splits"][first]["restaurant_ids"])
        second_restaurants = set(metrics["splits"][second]["restaurant_ids"])
        metrics["overlaps"][f"{first}_{second}_restaurant_overlap"] = len(first_restaurants & second_restaurants)

    for key, value in metrics["overlaps"].items():
        if key.endswith(("normalised_user_overlap", "input_overlap", "row_overlap")) and value:
            errors.append(f"{key}: {value}")

    disjoint_status = _load_disjoint_status(dataset_report)
    restaurant_overlap_total = sum(
        value for key, value in metrics["overlaps"].items() if key.endswith("_restaurant_overlap")
    )
    if restaurant_overlap_total:
        message = f"restaurant identifier overlap total: {restaurant_overlap_total}"
        if disjoint_status is False:
            limitations.append(message)
        else:
            errors.append(message)

    metrics["error_count"] = len(errors)
    metrics["errors"] = errors
    metrics["limitations"] = limitations
    metrics["restaurant_level_disjoint_split_achieved"] = disjoint_status
    return metrics


def print_summary(metrics: dict[str, Any]) -> None:
    print("Response data leakage check")
    for split, values in metrics["splits"].items():
        print(
            f"- {split}: rows={values['row_count']}, duplicate_rows={values['duplicate_rows']}, "
            f"duplicate_input_output_pairs={values['duplicate_input_output_pairs']}, "
            f"booking_ref_failures={values['booking_reference_grounding_failures']}"
        )
    print("- overlaps:")
    for key, value in sorted(metrics["overlaps"].items()):
        print(f"  {key}: {value}")
    if metrics["limitations"]:
        print("- limitations:")
        for limitation in metrics["limitations"]:
            print(f"  {limitation}")
    if metrics["errors"]:
        print("- errors:")
        for error in metrics["errors"]:
            print(f"  {error}")
    else:
        print("No prohibited response-data leakage or schema errors found.")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = check_splits(
        train_file=args.train_file,
        eval_file=args.eval_file,
        challenge_file=args.challenge_file,
        dataset_report=args.dataset_report,
    )
    print_summary(metrics)
    return 1 if metrics["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
