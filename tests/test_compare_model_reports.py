import json

from scripts.compare_model_reports import load_rows, render_markdown


def test_compare_model_reports_renders_requested_metrics(tmp_path):
    report = tmp_path / "strict.json"
    report.write_text(
        json.dumps(
            {
                "slot_extraction": {
                    "strict_json_parse_success_rate": 0.8,
                    "raw_parse_error_count": 2,
                    "final_intent_accuracy": 0.9,
                    "final_slot_precision": 0.91,
                    "final_slot_recall": 0.92,
                    "final_slot_f1": 0.915,
                    "fallback_used_cases": 3,
                },
                "system_metrics": {
                    "task_success_rate": 1.0,
                    "json_leakage_rate": 0.0,
                    "groundedness_pass_rate": 1.0,
                    "average_latency_ms": 12.0,
                    "p95_latency_ms": 25.0,
                },
            }
        ),
        encoding="utf-8",
    )

    rows = load_rows([f"strict={report}"])
    markdown = render_markdown(rows)

    assert rows[0]["strict_json_parse_success"] == 0.8
    assert "Strict JSON Parse Success" in markdown
    assert "Raw Parse Errors" in markdown
    assert "| strict | 0.8000 | 2 |" in markdown
