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
-> grounded LLM or template response
-> booking-state update
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The default tests and demo work without downloading an LLM. With `--enable-llm`,
the assistant uses Hugging Face Transformers for JSON intent/slot extraction and
grounded response generation when a supported model is available. Validated
rules, retrieval and booking safeguards remain active as guardrails.

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

For this local workspace, the MultiWOZ clone is currently one folder above this
project:

```powershell
python scripts/prepare_multiwoz.py --multiwoz-path "C:\Users\salih\OneDrive\Documents\Work\ai_module\Implementation\multiwoz"
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

Enable attempted Transformers generation:

```powershell
python scripts/run_chat.py --sample-data --enable-llm
```

Use separate models or a fine-tuned adapter for slot extraction:

```powershell
python scripts/run_chat.py --enable-llm --model-name google/flan-t5-small --slot-model-name google/flan-t5-small
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

The evaluation reports intent accuracy, exact slot-object accuracy, slot
precision, recall and F1, parse-error counts, slot latency, retrieval metrics,
response safety and end-to-end task success. The ablation script compares the
final stateful retrieval system with simpler baselines.

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

This runs:

- `baseline_rule_based`: deterministic rule extractor, no `--enable-llm`;
- `base_llm`: `--enable-llm --slot-model-name google/flan-t5-small`;
- `qlora_adapter`: `--enable-llm --slot-model-name models/slot-extractor-qlora`.

If the adapter folder is missing, the QLoRA row is marked as skipped. Outputs
are written to `outputs/evaluation/evaluation_matrix.json` and
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

## Optional QLoRA Fine-Tuning

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

For the Colab/GPU coursework run:

```powershell
python scripts/train_qlora_slot_extractor.py --base-model google/flan-t5-small --eval-file data/evaluation/slot_eval_cases.jsonl --metrics-output outputs/evaluation/qlora_training_metadata.json
python scripts/run_evaluation_matrix.py --sample-data
```

`notebooks/train_qlora_colab.ipynb` contains the Colab workflow: clone the repo,
install requirements, check the GPU, train the adapter, run the matrix and
optionally copy `outputs/evaluation` to Google Drive.

The default adapter output is ignored under `models/`. The fine-tuned adapter
changes only the slot extractor. Retrieval, slot validation, dialogue state and
booking logic remain deterministic guardrails. See `docs/llm_and_qlora.md` for
the LLM pipeline and recommended report wording.
