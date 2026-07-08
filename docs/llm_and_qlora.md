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
-> grounded LLM response generation
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

The evaluation output includes `slot_extraction_used_llm` and
`generation_mode`, so the report can show whether LLM components were active.

## QLoRA Fine-Tuning

The optional QLoRA script fine-tunes the JSON slot extractor, not the whole
booking system:

```powershell
pip install -r requirements-qlora.txt
python scripts/train_qlora_slot_extractor.py --base-model google/flan-t5-small --eval-file data/evaluation/slot_eval_cases.jsonl --metrics-output outputs/evaluation/qlora_training_metadata.json
```

By default it writes an adapter to:

```text
models/slot-extractor-qlora
```

Use the trained adapter for language understanding:

```powershell
python scripts/run_chat.py --enable-llm --slot-model-name models/slot-extractor-qlora
python scripts/evaluate.py --sample-data --slot-fixture data/evaluation/slot_eval_cases.jsonl --enable-llm --slot-model-name models/slot-extractor-qlora
```

4-bit QLoRA normally requires a CUDA-capable Linux, WSL or Colab environment.
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
`outputs/evaluation/evaluation_matrix.md`. If the adapter directory does not
exist, the QLoRA row is skipped instead of failing the whole evaluation.

## Report Wording

Recommended wording:

> The final prototype is an LLM-assisted retrieval-grounded dialogue system.
> An LLM performs JSON intent/slot extraction and grounded response generation,
> while deterministic validation, retrieval and dialogue state tracking prevent
> hallucinated restaurants, unsupported booking claims and unsafe state changes.
