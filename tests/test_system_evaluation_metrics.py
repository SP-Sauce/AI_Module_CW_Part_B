from scripts.evaluate import system_metric_summary


def test_system_metric_summary_exposes_required_metrics():
    summary = system_metric_summary(
        {
            "slot_extraction": {
                "intent_accuracy": 0.9,
                "slot_precision": 0.8,
                "slot_recall": 0.7,
                "slot_f1": 0.7467,
            },
            "retrieval": {"recall_at_3": 0.95},
            "end_to_end": {
                "task_success_rate": 1.0,
                "json_leakage_rate": 0.0,
                "groundedness_pass_rate": 1.0,
                "average_latency_ms": 12.5,
                "p95_latency_ms": 22.0,
            },
            "response_generation": {"groundedness_pass": True},
        }
    )

    assert summary == {
        "intent_accuracy": 0.9,
        "slot_precision": 0.8,
        "slot_recall": 0.7,
        "slot_f1": 0.7467,
        "retrieval_recall_at_3": 0.95,
        "task_success_rate": 1.0,
        "json_leakage_rate": 0.0,
        "groundedness_pass_rate": 1.0,
        "average_latency_ms": 12.5,
        "p95_latency_ms": 22.0,
    }
