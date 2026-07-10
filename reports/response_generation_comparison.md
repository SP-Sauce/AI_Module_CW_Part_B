# Response Generation Comparison

Headline result: `final_guarded_response`, because this is what the user sees.

| Mode | Cases | Groundedness | JSON/debug leakage | Unsupported claims | Nonempty | Fallback | Avg latency (s) | Clarity | Evidence preservation | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_template | 6 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.000000 | 1.0000 | 1.0000 | 0 |
| pretrained_flan_t5_base_response | 6 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000000 | 0.0000 | 1.0000 | 6 |
| trained_lora_response | 6 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000000 | 0.0000 | 1.0000 | 6 |
| final_guarded_response | 6 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.000000 | 1.0000 | 1.0000 | 0 |

Local LLM loading was not requested. Re-run with `--run-llm` to populate the pretrained and trained adapter columns.
