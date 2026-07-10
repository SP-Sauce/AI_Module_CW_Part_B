# Retrieval-Grounded LLM Dialogue System for Restaurant Search and Booking Records

This repository contains a Part B implementation for the 6CM606 Frontiers in
Artificial Intelligence and Data Science coursework. It builds a CPU-feasible
MultiWOZ restaurant assistant that can search restaurant records, maintain
session state and create proof-of-concept booking records.

## Overview

The system follows the Part A design direction of dialogue state tracking plus
controlled response generation, narrowed to the restaurant domain because the
EDA showed restaurant is the strongest MultiWOZ service for this prototype:
4,728 dialogues and 68,234 service-turn records.

Pipeline:

```text
User message
-> intent and slot extraction
-> dialogue state tracking
-> restaurant retrieval
-> preference-fit ranking
-> ResponsePlan construction
-> controlled NaturalLanguageGenerator response
-> booking-state update
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The default tests and demo work without downloading an LLM. With `--enable-llm`,
the assistant uses Hugging Face Transformers for JSON intent/slot extraction.
Optional response generation is separate: use `--enable-response-llm` with
`--response-model-name` when you want a guarded response model. Validated rules,
retrieval and booking safeguards remain active as guardrails.

For local LLM mode, install the optional backend dependencies as well:

```powershell
pip install -r requirements-llm.txt
```

If an older PyTorch is already installed, upgrade it first or use the PyTorch
CPU wheel index:

```powershell
python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu
```

## Dataset

The full raw MultiWOZ dataset is not committed. Clone or place the lecturer
provided MultiWOZ repository locally, then run:

```powershell
python scripts/prepare_multiwoz.py --multiwoz-path C:\path\to\multiwoz
```

For a local workspace where the MultiWOZ clone is one folder above this
project, use:

```powershell
python scripts/prepare_multiwoz.py --multiwoz-path "..\multiwoz"
```

If the full dataset is unavailable, the project still runs with:

```powershell
python scripts/prepare_multiwoz.py --sample-if-missing
```

Tests use `data/samples/sample_restaurants.json`.

## Run The Web App

```powershell
python scripts/run_web.py
```

Then open:

```text
http://127.0.0.1:5000
```

For the local demo overseer/admin view, open:

```text
http://127.0.0.1:5000/admin
```

Use the same login page for normal users and admin. Demo credentials:

```text
user1 / pass123  (existing demo history)
user2 / pass123  (empty account)
admin / pass123  (admin dashboard)
```

The admin dashboard shows all local sessions, saved conversations, booking
records and prototype metrics such as intent distribution, booking conversion,
average turns per session, clarification counts and latency for newly saved
turns. It also lets the local demo overseer close a session while preserving
the transcript and booking evidence for review. The raw dashboard payload is
also available at:

```text
http://127.0.0.1:5000/api/admin
```

If you want to use the bundled sample data instead of the processed MultiWOZ
restaurant records:

```powershell
python scripts/run_web.py --sample-data
```

The browser app now uses local username/password accounts. Register or log in
first, then the app creates user-owned conversation sessions and stores chat
turns plus booking records in a local SQLite database at
`data/runtime/bookings.sqlite3`. Set `BOOKING_DB_PATH` to change the database
location.

Account history is available from the chat sidebar or:

```text
http://127.0.0.1:5000/history
```

Use **Copy history** in the web app to copy the current session transcript and
booking summary. **New session** switches to a separate empty conversation; it
does not delete earlier session records from SQLite. **Exit conversation**
deletes the active chat session and its booking records, then starts a fresh
session with the opening greeting.

Booking references are scoped to the logged-in user and active session in the
public chat. The admin dashboard can show all local account/session records for
demo oversight, but a normal chat session cannot open booking details from
another account or session id.

## Run The Terminal Chatbot

```powershell
python scripts/run_chat.py --sample-data --debug
```

Example flow:

```text
You: I need a cheap Italian restaurant in the centre.
Assistant: I found pizza hut city centre...

You: Can you book it for Friday at 7 for 2 people?
Assistant: I have created a booking record...

You: Move it to Saturday.
Assistant: I have updated booking...

You: Cancel it.
Assistant: I have cancelled booking...
```

Enable attempted Transformers slot extraction:

```powershell
python scripts/run_chat.py --sample-data --enable-llm
```

Use separate slot and response models:

```powershell
python scripts/run_chat.py --enable-llm --enable-response-llm --slot-model-name models/slot-extractor-lora-strict --response-model-name google/flan-t5-base
```

## Run Tests

```powershell
pytest
```

All tests run against the sample dataset and do not require internet access.

## Run Evaluation

```powershell
python scripts/evaluate.py --sample-data
python scripts/evaluate.py --sample-data --slot-fixture data/evaluation/slot_eval_cases.jsonl
python scripts/run_ablation.py --sample-data
```

The first command keeps the original small fixture for backwards compatibility.
The second command evaluates the fresh hold-out slot set in
`data/evaluation/slot_eval_cases.jsonl`. Before using the hold-out set in the
report, check for train/evaluation leakage:

```powershell
python scripts/check_data_leakage.py
```

The evaluation reports raw LLM JSON validity separately from final
repair/fallback quality. Raw slot-model metrics include strict JSON parse
success, raw parse errors, strict intent accuracy and strict slot
precision/recall/F1. Final metrics include repaired JSON success, fallback use,
final intent accuracy and final slot precision/recall/F1. System-level metrics
include task success, JSON leakage rate, groundedness pass rate and latency. The
ablation script compares the final stateful retrieval system with simpler
baselines.

Evaluate the LLM path:

```powershell
python scripts/evaluate.py --sample-data --enable-llm --slot-model-name google/flan-t5-small
python scripts/run_ablation.py --enable-llm
```

The JSON output distinguishes LLM attempts, successful JSON parses, parse
failures and rule-based fallbacks. Per-case details include a bounded preview
of the raw slot-model output.

Inspect a few raw outputs before running the full matrix:

```powershell
python scripts/debug_llm_slot_outputs.py --slot-model-name google/flan-t5-small --limit 5
```

Run the report-ready experiment matrix:

```powershell
python scripts/run_evaluation_matrix.py --sample-data
```

Optionally run the same configurations against the separate 80-case challenge
fixture:

```powershell
python scripts/run_evaluation_matrix.py --sample-data --slot-fixture data/evaluation/slot_eval_cases.jsonl --challenge-fixture data/evaluation/slot_challenge_cases.jsonl
```

Challenge results are written to
`outputs/evaluation/challenge_evaluation_matrix.json` and
`outputs/evaluation/challenge_evaluation_matrix.md`; the main 50-case report
filenames remain unchanged.

This runs:

- `baseline_rule_based`: deterministic rule extractor, no `--enable-llm`;
- `base_llm`: lightweight `google/flan-t5-small` prompted extraction;
- `qlora_adapter`: optional FLAN-T5-small adapter at
  `models/slot-extractor-qlora`;
- `qlora_adapter_base`: optional stronger FLAN-T5-base adapter at
  `models/slot-extractor-qlora-base`.

Missing adapter folders are marked as skipped. Outputs are written to
`outputs/evaluation/evaluation_matrix.json` and
`outputs/evaluation/evaluation_matrix.md`.

## Safety And Limitations

- Booking records are proof-of-concept and session-only, with references such as `BK-AB12CD`.
- Relative booking days such as `today`, `tomorrow`, `day after tomorrow`
  and `the day after` are resolved from the local prompt timestamp using the
  configured timezone, defaulting to `Europe/London`.
- The assistant does not claim live restaurant availability.
- The assistant does not process payments.
- The assistant does not use external review websites.
- The assistant does not claim verified halal, vegetarian or allergy
  certification unless a field is explicitly present in the data.
- The assistant must not invent phone numbers, addresses, postcodes or food
  types. Responses are grounded in loaded restaurant records.
- The web app stores local session ids, chat turns and booking records in
  SQLite. The terminal chatbot keeps state only for the current process.

## Strict Slot Extractor Training On Kaggle

For the stronger extractor, build a strict training/dev split:

```powershell
python scripts/build_slot_training_data.py
```

This creates:

```text
data/training/slot_train_strict.jsonl
data/training/slot_dev_strict.jsonl
```

Targets are canonical minified JSON strings generated with sorted keys and
validated before saving. The builder checks for exact normalized text overlap
with `data/evaluation/slot_eval_cases.jsonl` and
`data/evaluation/slot_challenge_cases.jsonl`.

Train on Kaggle GPU with the cells in:

```text
notebooks/kaggle_train_strict_slot_extractor.md
```

The recommended full run is:

```powershell
python scripts/train_strict_slot_extractor_lora.py --base-model google/flan-t5-base --train-file data/training/slot_train_strict.jsonl --dev-file data/training/slot_dev_strict.jsonl --output-dir models/slot-extractor-lora-strict --max-steps 800
```

After downloading the Kaggle output, use it locally with:

```powershell
python scripts/run_web.py --enable-llm --slot-model-name models/slot-extractor-lora-strict --debug
```

That command enables the trained slot extractor only. To try guarded response
generation as well, add `--enable-response-llm --response-model-name
models/response-generator-lora` or use the default `google/flan-t5-base`
response model.

## Response Generation Dataset

The optional response-generator LoRA uses deterministic synthetic supervision.
The examples are generated from trusted MultiWOZ restaurant records, controlled
paraphrase pools and grounded response templates. They are not manually written
human annotations, and template-generated data may favour the system's target
response style. Automatic safety checks help catch leakage and unsupported
claims, but they do not replace human evaluation of trained model outputs.

Generate the expanded default response datasets:

```powershell
python scripts/build_response_training_data.py `
  --train-count 800 `
  --eval-count 160 `
  --challenge-count 100 `
  --seed 6062026
```

This writes:

```text
data/training/response_generation_examples.jsonl
data/evaluation/response_generation_eval.jsonl
data/evaluation/response_generation_challenge.jsonl
reports/response_generation_dataset_report.json
reports/response_generation_dataset_report.md
```

Use the bundled sample restaurants for a smaller CPU-only validation run:

```powershell
python scripts/build_response_training_data.py `
  --sample-data `
  --train-count 80 `
  --eval-count 20 `
  --challenge-count 20 `
  --seed 6062026
```

The builder splits restaurant records before generating examples. With enough
processed records, train, standard evaluation and challenge restaurant records
are disjoint. With the small sample dataset, the report clearly states any
restaurant-diversity limitation. In all cases the builder avoids exact input
overlap, exact normalised user-message overlap and duplicate rows.

Run the response-data leakage checker:

```powershell
python scripts/check_response_data_leakage.py
```

Codex did not execute response-model training or response-model evaluation.
Those steps are performed separately in Kaggle.

### Kaggle Response LoRA Commands

Every command in this subsection is **NOT EXECUTED BY CODEX - RUN MANUALLY IN KAGGLE**.

Install dependencies:

```python
!pip install -q -U -r requirements-qlora.txt
```

Confirm GPU:

```python
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

Generate the full response datasets, if not already generated locally:

```python
!python scripts/build_response_training_data.py \
  --train-count 800 \
  --eval-count 160 \
  --challenge-count 100 \
  --seed 6062026
```

Run the response-data leakage checker:

```python
!python scripts/check_response_data_leakage.py
```

Remove any previous response adapter:

```python
!rm -rf models/response-generator-lora
!mkdir -p outputs/evaluation reports
```

Train the response LoRA adapter:

```python
!python scripts/train_lora_response_generator.py \
  --base-model google/flan-t5-base \
  --train-file data/training/response_generation_examples.jsonl \
  --eval-file data/evaluation/response_generation_eval.jsonl \
  --output-dir models/response-generator-lora \
  --metadata-path outputs/evaluation/response_lora_training_metadata.json \
  --max-steps 300 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-4 \
  --eval-steps 50 \
  --save-steps 50 \
  --seed 6062026
```

Evaluate the pretrained and trained response models on the standard set:

```python
!python scripts/evaluate_response_generation.py \
  --run-llm \
  --response-model-name google/flan-t5-base \
  --adapter-path models/response-generator-lora \
  --eval-file data/evaluation/response_generation_eval.jsonl \
  --json-report reports/response_generation_comparison_trained.json \
  --markdown-report reports/response_generation_comparison_trained.md
```

Evaluate the pretrained and trained response models on the challenge set:

```python
!python scripts/evaluate_response_generation.py \
  --run-llm \
  --response-model-name google/flan-t5-base \
  --adapter-path models/response-generator-lora \
  --eval-file data/evaluation/response_generation_challenge.jsonl \
  --json-report reports/response_generation_challenge_trained.json \
  --markdown-report reports/response_generation_challenge_trained.md
```

Inspect the reports:

```python
!cat reports/response_generation_comparison_trained.md
!cat reports/response_generation_challenge_trained.md
```

Archive the response adapter and evidence:

```python
!zip -r response_generator_lora_artifacts.zip \
  models/response-generator-lora \
  outputs/evaluation/response_lora_training_metadata.json \
  reports/response_generation_comparison_trained.json \
  reports/response_generation_comparison_trained.md \
  reports/response_generation_challenge_trained.json \
  reports/response_generation_challenge_trained.md \
  reports/response_generation_dataset_report.json \
  reports/response_generation_dataset_report.md
```

Evaluate and save reports:

```powershell
python scripts/evaluate.py --sample-data --slot-fixture data/evaluation/slot_eval_cases.jsonl --enable-llm --slot-model-name models/slot-extractor-lora-strict --report-path reports/evaluation_lora_strict_eval.json
python scripts/evaluate.py --sample-data --slot-fixture data/evaluation/slot_challenge_cases.jsonl --enable-llm --slot-model-name models/slot-extractor-lora-strict --report-path reports/evaluation_lora_strict_challenge.json
python scripts/compare_model_reports.py --report strict_eval=reports/evaluation_lora_strict_eval.json --report strict_challenge=reports/evaluation_lora_strict_challenge.json
```

The NLG safety boundary remains in place after slot extraction. The model is
trained to improve raw structured output honestly, but malformed JSON, weak
repairs and fallback use are still reported rather than hidden.

## Optional Legacy QLoRA Fine-Tuning

The repository includes an optional LoRA/QLoRA training path for the LLM slot
extractor:

```powershell
pip install -r requirements-qlora.txt
Remove-Item -Recurse -Force models/slot-extractor-qlora -ErrorAction SilentlyContinue
python scripts/train_qlora_slot_extractor.py --base-model google/flan-t5-small
```

The adapter prompt now uses a strict `User: ...` followed by `JSON:` answer
marker. Any adapter trained with the earlier prompt format is incompatible and
must be deleted and retrained before adapter evaluation.

For a short local CPU smoke test of the LoRA path, disable 4-bit loading and
use a tiny number of steps:

```powershell
python scripts/train_qlora_slot_extractor.py --base-model google/flan-t5-small --no-4bit --max-steps 5 --output-dir models/slot-extractor-smoke
```

For the legacy GPU coursework run:

```powershell
python scripts/augment_slot_training_data.py
python scripts/check_data_leakage.py --train-file data/training/slot_instruction_examples_augmented.jsonl --eval-file data/evaluation/slot_eval_cases.jsonl
python scripts/train_qlora_slot_extractor.py --base-model google/flan-t5-base --train-file data/training/slot_instruction_examples_augmented.jsonl --eval-file data/evaluation/slot_eval_cases.jsonl --output-dir models/slot-extractor-qlora-base --metrics-output outputs/evaluation/qlora_base_training_metadata.json --batch-size 1 --max-steps 600
python scripts/run_evaluation_matrix.py --sample-data
```

`notebooks/train_qlora_colab.ipynb` is kept as an older workflow reference.
The current strict extractor workflow is Kaggle-first and is documented in
`notebooks/kaggle_train_strict_slot_extractor.md`. FLAN-T5-small remains the
lightweight baseline and optional small-adapter comparison.

The augmentation script deterministically expands the original instruction set
with balanced paraphrase templates, multi-slot targets and repeated complete
empty-slot objects. It writes
`data/training/slot_instruction_examples_augmented.jsonl` without changing the
original training file. Every target is validated as compact JSON, and
normalized user text is checked against the separate hold-out file
`data/evaluation/slot_eval_cases.jsonl` to prevent evaluation leakage.

The default adapter output is ignored under `models/`. The fine-tuned adapter
changes only the slot extractor. Retrieval, slot validation, dialogue state and
booking logic remain deterministic guardrails. Both model sizes still pass
through JSON repair, weak-repair detection, trusted-intent/slot checks and
rule-based fallback because a larger model can still emit malformed structured
output. See `docs/llm_and_qlora.md` for the LLM pipeline and recommended report
wording.
