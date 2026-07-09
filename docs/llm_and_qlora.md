# LLM And QLoRA Path

## Runtime LLM Use

The assistant can run as an LLM-assisted, retrieval-grounded task-oriented
dialogue system. When `--enable-llm` is used, the pipeline becomes:

```text
user message
-> LLM JSON intent and slot extraction
-> rule validation and safety fallback
-> dialogue state tracking
-> TF-IDF retrieval and preference-fit ranking
-> ResponsePlan construction
-> controlled NLG response with leakage checks
-> booking record update
```

The LLM is used in two places:

- language understanding: intent and slot extraction through
  `OptionalLLMSlotExtractor`;
- response generation: concise grounded replies through
  `GroundedResponseGenerator`.

The deterministic parts remain deliberately in the loop. Slot validation keeps
the LLM within supported MultiWOZ values, retrieval prevents invented restaurant
details, and booking code only creates local proof-of-concept records.

The QLoRA path fine-tunes only the language-understanding slot extractor. It
does not fine-tune retrieval, response validation, dialogue state, ranking or
booking behavior. Those components remain deterministic guardrails before and
after adapter training.

## Commands

Run the CLI with LLM extraction and generation:

```powershell
python scripts/run_chat.py --enable-llm --debug
```

Run the web demo with separate models for generation and slot extraction:

```powershell
python scripts/run_web.py --enable-llm --model-name google/flan-t5-small --slot-model-name google/flan-t5-small
```

Evaluate the LLM path:

```powershell
python scripts/evaluate.py --sample-data --slot-fixture data/evaluation/slot_eval_cases.jsonl --enable-llm --slot-model-name google/flan-t5-small
```

The slot evaluation output includes `llm_attempted_cases`,
`llm_parse_success_cases`, `llm_parse_failed_cases` and
`fallback_used_cases`, plus raw-output previews in case details. It also
separates strict raw model metrics from final repaired/fallback metrics:
`strict_json_parse_success_rate`, `raw_parse_error_count`,
`strict_intent_accuracy`, `strict_slot_precision`, `strict_slot_recall`,
`strict_slot_f1`, then `final_intent_accuracy`, `final_slot_precision`,
`final_slot_recall` and `final_slot_f1`.

Inspect raw model output before running the full evaluation matrix:

```powershell
python scripts/debug_llm_slot_outputs.py --slot-model-name google/flan-t5-small --limit 5
```

## Strict Kaggle LoRA Fine-Tuning

The current recommended stronger extractor is the Kaggle LoRA path:

```powershell
python scripts/build_slot_training_data.py
python scripts/train_strict_slot_extractor_lora.py --base-model google/flan-t5-base --train-file data/training/slot_train_strict.jsonl --dev-file data/training/slot_dev_strict.jsonl --output-dir models/slot-extractor-lora-strict --max-steps 800
python scripts/evaluate.py --sample-data --slot-fixture data/evaluation/slot_eval_cases.jsonl --enable-llm --slot-model-name models/slot-extractor-lora-strict --report-path reports/evaluation_lora_strict_eval.json
```

Use `notebooks/kaggle_train_strict_slot_extractor.md` for exact Kaggle cells.
This path trains the model to emit exactly one minified JSON object using the
same prompt used at runtime. It avoids copying exact evaluation fixture text and
records strict raw JSON dev metrics in
`models/slot-extractor-lora-strict/training_metadata.json`.

The NLG safety boundary remains active after training. The stronger extractor
should reduce raw parse failures; it is not used to hide them.

## Legacy QLoRA Fine-Tuning

The optional QLoRA script fine-tunes the JSON slot extractor, not the whole
booking system. `google/flan-t5-small` is retained as the lightweight prompted
baseline and optional small-adapter experiment:

```powershell
pip install -r requirements-qlora.txt
Remove-Item -Recurse -Force models/slot-extractor-qlora -ErrorAction SilentlyContinue
python scripts/train_qlora_slot_extractor.py --base-model google/flan-t5-small --eval-file data/evaluation/slot_eval_cases.jsonl --metrics-output outputs/evaluation/qlora_training_metadata.json
```

The training and inference paths share a strict prompt ending in `User: ...`
and `JSON:`. Delete and retrain any adapter created before this prompt-format
fix; an old adapter must not be evaluated against the new inference prompt.

By default it writes an adapter to:

```text
models/slot-extractor-qlora
```

Use the trained adapter for language understanding:

```powershell
python scripts/run_chat.py --enable-llm --slot-model-name models/slot-extractor-qlora
python scripts/evaluate.py --sample-data --slot-fixture data/evaluation/slot_eval_cases.jsonl --enable-llm --slot-model-name models/slot-extractor-qlora
```

The stronger older GPU experiment adapts `google/flan-t5-base` separately,
so it does not overwrite the small-model adapter:

```powershell
python scripts/augment_slot_training_data.py
python scripts/check_data_leakage.py --train-file data/training/slot_instruction_examples_augmented.jsonl --eval-file data/evaluation/slot_eval_cases.jsonl
python scripts/train_qlora_slot_extractor.py --base-model google/flan-t5-base --train-file data/training/slot_instruction_examples_augmented.jsonl --eval-file data/evaluation/slot_eval_cases.jsonl --output-dir models/slot-extractor-qlora-base --metrics-output outputs/evaluation/qlora_base_training_metadata.json --batch-size 1 --max-steps 600
python scripts/evaluate.py --sample-data --slot-fixture data/evaluation/slot_eval_cases.jsonl --enable-llm --slot-model-name models/slot-extractor-qlora-base
```

`augment_slot_training_data.py` uses deterministic templates to balance intent
coverage and reinforce complete compact JSON with empty, single-slot and
multi-slot objects. The original instruction file is preserved. Generation
fails if normalized training text overlaps the independent hold-out evaluation
fixture, and the explicit leakage command checks the generated file again.

The matrix reports this adapter as `qlora_adapter_base` when
`models/slot-extractor-qlora-base` exists. Larger models are not assumed to be
structurally reliable: strict parsing, constrained repair, weak-repair
detection, slot validation, trusted-intent/slot checks and rule-based fallback
remain active for both adapter sizes.

4-bit QLoRA normally requires a CUDA-capable Linux, WSL, Kaggle or Colab environment.
On CPU-only machines, use `--no-4bit` for a small LoRA smoke test, or keep the
base LLM path and describe QLoRA as implemented optional fine-tuning.

Local smoke test:

```powershell
python scripts/train_qlora_slot_extractor.py --base-model google/flan-t5-small --no-4bit --max-steps 5 --output-dir models/slot-extractor-smoke
```

Report-ready comparison:

```powershell
python scripts/run_evaluation_matrix.py --sample-data
```

The matrix writes `outputs/evaluation/evaluation_matrix.json` and
`outputs/evaluation/evaluation_matrix.md`. Missing small or base adapter
directories are skipped instead of failing the whole evaluation.

## Report Wording

Recommended wording:

> The final prototype is an LLM-assisted retrieval-grounded dialogue system.
> An LLM performs JSON intent/slot extraction and grounded response generation,
> while deterministic validation, retrieval and dialogue state tracking prevent
> hallucinated restaurants, unsupported booking claims and unsafe state changes.
