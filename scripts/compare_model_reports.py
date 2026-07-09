"""Create a Markdown comparison table from evaluation JSON reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_REPORT_DIR = Path("reports")
DEFAULT_OUTPUT = DEFAULT_REPORT_DIR / "model_comparison.md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare restaurant assistant model evaluation reports.")
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        help="Report spec as label=path.json. May be repeated. If omitted, reports/*.json is used.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _load_report_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label.strip(), Path(path.strip())
    path = Path(spec)
    return path.stem, path


def _metric(payload: dict[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def discover_reports(report_specs: list[str]) -> list[tuple[str, Path]]:
    if report_specs:
        return [_load_report_spec(spec) for spec in report_specs]
    if not DEFAULT_REPORT_DIR.exists():
        return []
    return [(path.stem, path) for path in sorted(DEFAULT_REPORT_DIR.glob("*.json"))]


def load_rows(report_specs: list[str]) -> list[dict[str, Any]]:
    rows = []
    for label, path in discover_reports(report_specs):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model": label,
                "path": str(path),
                "strict_json_parse_success": _metric(payload, "slot_extraction.strict_json_parse_success_rate"),
                "raw_parse_errors": _metric(payload, "slot_extraction.raw_parse_error_count"),
                "intent_accuracy": _first_present(
                    _metric(payload, "slot_extraction.final_intent_accuracy"),
                    _metric(payload, "slot_extraction.intent_accuracy"),
                ),
                "slot_precision": _first_present(
                    _metric(payload, "slot_extraction.final_slot_precision"),
                    _metric(payload, "slot_extraction.slot_precision"),
                ),
                "slot_recall": _first_present(
                    _metric(payload, "slot_extraction.final_slot_recall"),
                    _metric(payload, "slot_extraction.slot_recall"),
                ),
                "slot_f1": _first_present(
                    _metric(payload, "slot_extraction.final_slot_f1"),
                    _metric(payload, "slot_extraction.slot_f1"),
                ),
                "fallback_usage": _metric(payload, "slot_extraction.fallback_used_cases"),
                "task_success_rate": _first_present(
                    _metric(payload, "system_metrics.task_success_rate"),
                    _metric(payload, "end_to_end.task_success_rate"),
                ),
                "json_leakage_rate": _first_present(
                    _metric(payload, "system_metrics.json_leakage_rate"),
                    _metric(payload, "end_to_end.json_leakage_rate"),
                ),
                "groundedness_pass_rate": _first_present(
                    _metric(payload, "system_metrics.groundedness_pass_rate"),
                    _metric(payload, "end_to_end.groundedness_pass_rate"),
                ),
                "average_latency_ms": _first_present(
                    _metric(payload, "system_metrics.average_latency_ms"),
                    _metric(payload, "end_to_end.average_latency_ms"),
                ),
                "p95_latency_ms": _first_present(
                    _metric(payload, "system_metrics.p95_latency_ms"),
                    _metric(payload, "end_to_end.p95_latency_ms"),
                ),
            }
        )
    return rows


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Model Comparison",
        "",
        "| Model | Strict JSON Parse Success | Raw Parse Errors | Intent Accuracy | Slot Precision | Slot Recall | Slot F1 | Fallback Used | Task Success | JSON Leakage | Groundedness | Avg Latency ms | P95 Latency ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    _fmt(row["strict_json_parse_success"]),
                    _fmt(row["raw_parse_errors"]),
                    _fmt(row["intent_accuracy"]),
                    _fmt(row["slot_precision"]),
                    _fmt(row["slot_recall"]),
                    _fmt(row["slot_f1"]),
                    _fmt(row["fallback_usage"]),
                    _fmt(row["task_success_rate"]),
                    _fmt(row["json_leakage_rate"]),
                    _fmt(row["groundedness_pass_rate"]),
                    _fmt(row["average_latency_ms"]),
                    _fmt(row["p95_latency_ms"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- Strict JSON metrics are computed from raw slot-model output before repair or rule fallback.",
            "- Final intent/slot metrics include validation, repair and fallback, so they should be read separately from raw model quality.",
            "- JSON leakage and groundedness are system-level assistant response checks after ResponsePlan/NLG safety handling.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = load_rows(args.report)
    if not rows:
        raise SystemExit("No evaluation reports found. Pass --report label=path.json or create reports/*.json.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(rows), encoding="utf-8")
    print(f"Wrote comparison report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
