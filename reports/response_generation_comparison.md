# Response Generation Comparison

Headline result: `final_guarded_response`, because this is what the user sees after validation and deterministic fallback.

| Mode | Cases | Attempts | Accepted | Rejected | Fallback | Grounded before | Grounded after | JSON/debug before | Unsupported before | Avg latency (s) | Rejection reasons | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| deterministic_baseline_response | 160 | 0 | 0 | 0 | 0 | N/A | 1.0000 | N/A | N/A | 0.000000 | - | 0 |
| raw_pretrained_model_response | 160 | 0 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | 0.000000 | - | 160 |
| raw_trained_lora_response | 160 | 0 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | 0.000000 | - | 160 |
| final_guarded_response | 160 | 0 | 0 | 0 | 0 | N/A | 1.0000 | N/A | N/A | 0.000000 | - | 0 |

Local LLM loading was not requested. Re-run with `--run-llm` to populate raw pretrained and trained-adapter model attempts.
