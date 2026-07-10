# Response Generation Comparison

Headline result: `final_guarded_response`, because this is what the user sees after validation and deterministic fallback.

| Mode | Cases | Attempts | Accepted | Rejected | Fallback | Grounded before | Grounded after | JSON/debug before | Unsupported before | Avg latency (s) | Rejection reasons | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| deterministic_baseline_response | 100 | 0 | 0 | 0 | 0 | N/A | 1.0000 | N/A | N/A | 0.000000 | - | 0 |
| raw_pretrained_model_response | 100 | 100 | 53 | 47 | 0 | 0.5300 | 0.5300 | 0.0000 | 0.0500 | 0.979736 | invented_address: 42, unsupported_claim: 5 | 0 |
| raw_trained_lora_response | 100 | 100 | 92 | 8 | 0 | 0.9200 | 0.9200 | 0.0000 | 0.0300 | 0.697509 | invented_phone: 1, invented_restaurant_name: 2, prompt_label_leakage: 2, unsupported_claim: 3 | 0 |
| final_guarded_response | 100 | 100 | 92 | 8 | 8 | 0.9200 | 1.0000 | 0.0000 | 0.0300 | 0.697509 | invented_phone: 1, invented_restaurant_name: 2, prompt_label_leakage: 2, unsupported_claim: 3 | 0 |
