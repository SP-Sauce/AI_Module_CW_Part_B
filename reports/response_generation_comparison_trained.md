# Response Generation Comparison

Headline result: `final_guarded_response`, because this is what the user sees after validation and deterministic fallback.

| Mode | Cases | Attempts | Accepted | Rejected | Fallback | Grounded before | Grounded after | JSON/debug before | Unsupported before | Avg latency (s) | Rejection reasons | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| deterministic_baseline_response | 160 | 0 | 0 | 0 | 0 | N/A | 1.0000 | N/A | N/A | 0.000000 | - | 0 |
| raw_pretrained_model_response | 160 | 160 | 82 | 78 | 0 | 0.5125 | 0.5125 | 0.0000 | 0.0625 | 0.874311 | invented_address: 66, invented_restaurant_name: 2, unsupported_claim: 10 | 0 |
| raw_trained_lora_response | 160 | 160 | 151 | 9 | 0 | 0.9437 | 0.9437 | 0.0000 | 0.0250 | 0.661876 | invented_restaurant_name: 5, unsupported_claim: 4 | 0 |
| final_guarded_response | 160 | 160 | 151 | 9 | 9 | 0.9437 | 1.0000 | 0.0000 | 0.0250 | 0.661876 | invented_restaurant_name: 5, unsupported_claim: 4 | 0 |
