"""Print raw LLM slot-extraction outputs for a small labelled fixture sample."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from restaurant_assistant.slot_extraction import OptionalLLMSlotExtractor


DEFAULT_SLOT_FIXTURE = ROOT / "data" / "evaluation" / "slot_eval_cases.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect raw LLM slot-extraction outputs.")
    parser.add_argument("--slot-model-name", default="google/flan-t5-small")
    parser.add_argument("--slot-fixture", type=Path, default=DEFAULT_SLOT_FIXTURE)
    parser.add_argument("--limit", type=int, default=10)
    return parser


def load_cases(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    if raw.lstrip().startswith("["):
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected a JSON array in {path}")
        return [case for case in parsed if isinstance(case, dict)]

    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        case = json.loads(line)
        if not isinstance(case, dict):
            raise ValueError(f"Expected a JSON object at {path}:{line_number}")
        cases.append(case)
    return cases


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    cases = load_cases(args.slot_fixture)[: args.limit]
    extractor = OptionalLLMSlotExtractor(args.slot_model_name)
    for index, case in enumerate(cases, start=1):
        result = extractor.extract(str(case.get("text", "")))
        print(f"\n--- Case {index} ---")
        print(f"Input: {case.get('text', '')}")
        print(
            "Expected: "
            + json.dumps(
                {"intent": case.get("intent"), "slots": case.get("slots", {})},
                ensure_ascii=False,
            )
        )
        print(f"LLM attempted: {result.llm_attempted}")
        print(f"Raw LLM output: {result.llm_raw_output if result.llm_raw_output is not None else '<none>'}")
        print(f"Strict parse success: {result.llm_parse_success}")
        print(f"Repair success: {result.llm_repair_success}")
        print(f"Repair strategy: {result.llm_repair_strategy}")
        print(f"Weak repair: {result.llm_repair_weak}")
        print(f"Repaired intent: {result.llm_repaired_intent}")
        print(f"Intent trusted: {result.llm_intent_trusted}")
        print(f"Slots trusted: {result.llm_slots_trusted}")
        print(f"Trusted LLM slots: {json.dumps(result.llm_trusted_slots, ensure_ascii=False)}")
        print(f"Meaningful LLM slot contribution: {result.llm_meaningful_slot_contribution}")
        if result.llm_repaired_output is not None:
            print(f"Repaired JSON: {result.llm_repaired_output}")
        if result.llm_parse_success or result.llm_repair_success:
            print(
                "Parsed: "
                + json.dumps({"intent": result.intent, "slots": result.slots}, ensure_ascii=False)
            )
        else:
            print("Parsed: <unavailable; rule-based fallback used>")
            print(
                "Fallback: "
                + json.dumps({"intent": result.intent, "slots": result.slots}, ensure_ascii=False)
            )
        print(f"Errors: {json.dumps(result.errors, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
