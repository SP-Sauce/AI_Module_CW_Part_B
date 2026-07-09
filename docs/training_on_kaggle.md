# Training The Strict Slot Extractor On Kaggle

This project uses a layered assistant architecture. The slot extractor predicts
intent and slots, but the final assistant remains protected by slot validation,
repair/fallback, retrieval grounding, `ResponsePlan`, and controlled NLG. The
strict LoRA model improves raw JSON validity; it does not replace those system
guardrails.

## Build Strict Data

```powershell
python scripts/build_slot_training_data.py
```

This writes:

- `data/training/slot_train_strict.jsonl`
- `data/training/slot_dev_strict.jsonl`

Each row has an `input` and a canonical compact JSON `target`. The builder uses
a fixed seed, validates every target with `json.loads()`, and rejects exact
normalized text overlap with both evaluation fixtures:

- `data/evaluation/slot_eval_cases.jsonl`
- `data/evaluation/slot_challenge_cases.jsonl`

## Train On Kaggle

Open `notebooks/kaggle_train_strict_slot_extractor.md` and run the cells in a
Kaggle Notebook with GPU enabled. The recommended model is:

```text
google/flan-t5-base
```

The output adapter is saved to:

```text
models/slot-extractor-lora-strict
```

The trainer also writes:

```text
models/slot-extractor-lora-strict/training_metadata.json
```

That metadata includes model name, dataset sizes, seed, LoRA settings, train and
eval loss, strict raw JSON success on dev, raw parse errors, strict intent
accuracy, strict slot precision/recall/F1, timestamp and CUDA/GPU details.

## Use The Model Locally

After downloading and unzipping the Kaggle output into the repo root:

```powershell
python scripts/run_web.py --enable-llm --slot-model-name models/slot-extractor-lora-strict --debug
```

Greedy decoding is the default for structured extraction:

```powershell
python scripts/run_web.py --enable-llm --slot-model-name models/slot-extractor-lora-strict --slot-num-beams 1
```

To try beam search:

```powershell
python scripts/run_web.py --enable-llm --slot-model-name models/slot-extractor-lora-strict --slot-num-beams 4
```

## Evaluate

```powershell
python scripts/evaluate.py --sample-data --slot-fixture data/evaluation/slot_eval_cases.jsonl --enable-llm --slot-model-name models/slot-extractor-lora-strict --report-path reports/evaluation_lora_strict_eval.json
python scripts/evaluate.py --sample-data --slot-fixture data/evaluation/slot_challenge_cases.jsonl --enable-llm --slot-model-name models/slot-extractor-lora-strict --report-path reports/evaluation_lora_strict_challenge.json
```

Then compare reports:

```powershell
python scripts/compare_model_reports.py --report strict_eval=reports/evaluation_lora_strict_eval.json --report strict_challenge=reports/evaluation_lora_strict_challenge.json
```

## Interpreting Metrics

Raw LLM metrics are computed before repair or fallback:

- `strict_json_parse_success_rate`
- `raw_parse_error_count`
- `strict_intent_accuracy`
- `strict_slot_precision`
- `strict_slot_recall`
- `strict_slot_f1`

Final metrics include validation, repair and fallback:

- `repaired_json_success_rate`
- `fallback_used_cases`
- `final_intent_accuracy`
- `final_slot_precision`
- `final_slot_recall`
- `final_slot_f1`

System-level metrics check the complete assistant:

- `task_success_rate`
- `json_leakage_rate`
- `groundedness_pass_rate`
- `average_latency_ms`
- `p95_latency_ms`

The honest report story should be: the old raw LLM often failed strict JSON
parsing; strict LoRA training improves raw JSON validity; validation, fallback,
retrieval grounding and controlled NLG keep the final assistant robust.
