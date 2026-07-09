"""Run baseline, base-LLM and adapter slot-evaluation configurations."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluate import ROOT, run_evaluation


DEFAULT_SLOT_FIXTURE = ROOT / "data" / "evaluation" / "slot_eval_cases.jsonl"
DEFAULT_ADAPTER_PATH = ROOT / "models" / "slot-extractor-qlora"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "evaluation"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the coursework evaluation matrix.")
    parser.add_argument("--sample-data", action="store_true", help="Use bundled sample restaurants.")
    parser.add_argument("--slot-fixture", type=Path, default=DEFAULT_SLOT_FIXTURE)
    parser.add_argument("--model-name", default=None, help="Optional response-generation model override.")
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def _metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _command_for_config(
    *,
    sample_data: bool,
    slot_fixture: Path,
    enable_llm: bool,
    slot_model_name: str | None,
    model_name: str | None,
) -> str:
    parts = ["python", "scripts/evaluate.py"]
    if sample_data:
        parts.append("--sample-data")
    parts.extend(["--slot-fixture", str(slot_fixture)])
    if enable_llm:
        parts.append("--enable-llm")
    if model_name:
        parts.extend(["--model-name", model_name])
    if slot_model_name:
        parts.extend(["--slot-model-name", slot_model_name])
    return " ".join(parts)


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Evaluation Matrix",
        "",
        "| Configuration | Status | Intent Accuracy | Exact Slot Accuracy | Slot Precision | Slot Recall | Slot F1 | Raw Parse Errors | LLM Attempted | Strict Parse Success | Repair Success | Weak Repair | Intent Trusted | Slots Trusted | Meaningful LLM Slots | Valid or Repaired | Strict Parse Failed | Unrepaired Failure | Fallback Used | Mean Slot Latency (s) | Slot Model |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        slot_metrics = row.get("metrics", {}).get("slot_extraction", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    row["name"],
                    row["status"],
                    _metric(slot_metrics.get("intent_accuracy")),
                    _metric(slot_metrics.get("exact_slot_object_accuracy")),
                    _metric(slot_metrics.get("slot_precision")),
                    _metric(slot_metrics.get("slot_recall")),
                    _metric(slot_metrics.get("slot_f1")),
                    _metric(slot_metrics.get("invalid_json_or_parse_error_count")),
                    _metric(slot_metrics.get("llm_attempted_cases")),
                    _metric(slot_metrics.get("llm_parse_success_cases")),
                    _metric(slot_metrics.get("llm_repair_success_cases")),
                    _metric(slot_metrics.get("llm_repair_weak_cases")),
                    _metric(slot_metrics.get("llm_intent_trusted_cases")),
                    _metric(slot_metrics.get("llm_slots_trusted_cases")),
                    _metric(slot_metrics.get("meaningful_llm_slot_contribution_cases")),
                    _metric(slot_metrics.get("llm_valid_or_repaired_cases")),
                    _metric(slot_metrics.get("llm_parse_failed_cases")),
                    _metric(slot_metrics.get("llm_unrepaired_failure_cases")),
                    _metric(slot_metrics.get("fallback_used_cases")),
                    _metric(slot_metrics.get("mean_slot_latency_seconds")),
                    str(slot_metrics.get("slot_model_name") or "-"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Commands run:",
            "",
        ]
    )
    for row in rows:
        lines.append(f"- `{row['name']}`: `{row['command']}`")
    parse_failures: list[tuple[str, dict[str, Any]]] = []
    for row in rows:
        cases = row.get("metrics", {}).get("slot_extraction", {}).get("cases", [])
        for case in cases:
            predicted = case.get("predicted", {})
            if predicted.get("llm_attempted") and not predicted.get("llm_parse_success"):
                parse_failures.append((row["name"], case))
    if parse_failures:
        lines.extend(["", "## LLM parse failures", ""])
        for configuration, case in parse_failures:
            predicted = case.get("predicted", {})
            raw_output = str(predicted.get("llm_raw_output") or "<no output>")
            repaired_output = str(predicted.get("llm_repaired_output") or "<not repaired>")
            errors = "; ".join(str(error) for error in predicted.get("errors", []))
            lines.append(
                f"- `{configuration}` - input: `{case.get('text', '')}`; "
                f"raw output: `{raw_output}`; repaired output: `{repaired_output}`; "
                f"repair success: `{bool(predicted.get('llm_repair_success'))}`; "
                f"weak repair: `{bool(predicted.get('llm_repair_weak'))}`; "
                f"intent trusted: `{bool(predicted.get('llm_intent_trusted'))}`; errors: `{errors}`"
            )
    return "\n".join(lines) + "\n"


def run_matrix(args: argparse.Namespace) -> list[dict[str, Any]]:
    configs = [
        {
            "name": "baseline_rule_based",
            "enable_llm": False,
            "slot_model_name": None,
        },
        {
            "name": "base_llm",
            "enable_llm": True,
            "slot_model_name": "google/flan-t5-small",
        },
        {
            "name": "qlora_adapter",
            "enable_llm": True,
            "slot_model_name": str(args.adapter_path),
        },
    ]
    rows: list[dict[str, Any]] = []
    for config in configs:
        command = _command_for_config(
            sample_data=args.sample_data,
            slot_fixture=args.slot_fixture,
            enable_llm=config["enable_llm"],
            slot_model_name=config["slot_model_name"],
            model_name=args.model_name,
        )
        if config["name"] == "qlora_adapter" and not args.adapter_path.exists():
            rows.append(
                {
                    "name": config["name"],
                    "status": "skipped",
                    "reason": f"Adapter folder not found: {args.adapter_path}",
                    "command": command,
                    "metrics": {},
                }
            )
            continue
        try:
            metrics = run_evaluation(
                sample_data=args.sample_data,
                enable_llm=config["enable_llm"],
                model_name=args.model_name,
                slot_model_name=config["slot_model_name"],
                slot_fixture=args.slot_fixture,
            )
            rows.append(
                {
                    "name": config["name"],
                    "status": "completed",
                    "command": command,
                    "metrics": metrics,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "name": config["name"],
                    "status": "failed",
                    "error": str(exc),
                    "command": command,
                    "metrics": {},
                }
            )
    return rows


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = run_matrix(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_data": args.sample_data,
        "slot_fixture": str(args.slot_fixture),
        "results": rows,
    }
    json_path = args.output_dir / "evaluation_matrix.json"
    markdown_path = args.output_dir / "evaluation_matrix.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    markdown_path.write_text(_markdown_table(rows), encoding="utf-8")
    print(f"Saved evaluation matrix JSON to {json_path}")
    print(f"Saved evaluation matrix Markdown to {markdown_path}")


if __name__ == "__main__":
    main()
