# Model Comparison

| Model | Strict JSON Parse Success | Raw Parse Errors | Intent Accuracy | Slot Precision | Slot Recall | Slot F1 | Fallback Used | Task Success | JSON Leakage | Groundedness | Avg Latency ms | P95 Latency ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| strict_eval | 0.0000 | 50 | 1.0000 | 0.7468 | 1.0000 | 0.8550 | 32 | 1.0000 | 0.0000 | 1.0000 | 2328.6660 | 4753.1500 |
| strict_challenge | 0.0000 | 80 | 0.6625 | 0.7600 | 0.8962 | 0.8225 | 46 | 1.0000 | 0.0000 | 1.0000 | 2316.9520 | 4745.8320 |

Notes:

- Strict JSON metrics are computed from raw slot-model output before repair or rule fallback.
- Final intent/slot metrics include validation, repair and fallback, so they should be read separately from raw model quality.
- JSON leakage and groundedness are system-level assistant response checks after ResponsePlan/NLG safety handling.
