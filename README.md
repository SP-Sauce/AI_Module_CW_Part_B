# Retrieval-Grounded LLM Dialogue System

This repository is the Part B prototype for a customer-support dialogue system.
It focuses on the algorithmic and system-design aspects required by the brief:
understanding customer queries, producing grounded responses and maintaining
coherent multi-turn context. The runnable domain is MultiWOZ restaurant search
and proof-of-concept booking records.

The implementation is intentionally local and reproducible. It does not depend
on external databases, payment systems, live restaurant availability or
deployment infrastructure. A small Flask app is included only for the final demo.

## Brief Alignment

| Brief requirement | Implementation evidence |
| --- | --- |
| Interpret and process customer queries using LLM-based techniques | Rule baseline plus optional FLAN-T5/LoRA intent-slot extractor with strict JSON validation, repair metrics and rule fallback. |
| Generate contextually appropriate, accurate and clear responses | Retrieval-grounded `ResponsePlan` plus deterministic NLG; optional guarded response model path uses the same shared prompt for training, runtime and evaluation. |
| Manage conversational context | `DialogueState` tracks food, area, price, selected restaurant, booking day/time/people, booking references and history across turns. |
| Working AI component with setup and execution instructions | CLI chatbot, web demo, data builders, evaluation scripts and tests are included below. |
| Baseline performance metrics | Committed reports in `reports/` cover slot extraction, task success, latency, response groundedness and leakage checks. |
| Reproducibility and testing | Fixed seeds, checked JSONL data, leakage checkers, `pyproject.toml` test config and a full pytest suite. |

## What Is Implemented

- Intent and slot extraction for restaurant search, list, restaurant info,
  booking, reschedule, cancellation, booking lookup, greetings and unsupported
  requests.
- Optional LLM slot extraction through Transformers, with strict minified JSON
  prompts and post-generation validation.
- Dialogue state tracking for ongoing conversations.
- TF-IDF restaurant retrieval and transparent preference-fit ranking.
- Grounded response planning and deterministic natural-language generation.
- Optional response-generator LoRA workflow with shared prompt construction
  across dataset generation, training, runtime inference and evaluation.
- Safety guardrails for unsupported domains, live availability, payments,
  reviews, halal/allergy claims, JSON/debug leakage and invented restaurant
  evidence.
- Proof-of-concept local booking records with generated references such as
  `BK-AB12CD`.
- CLI demo, local Flask demo, account-scoped chat history and local admin
  dashboard for presentation.

## Repository Map

```text
src/restaurant_assistant/   Core assistant, state, retrieval, NLG, web app
scripts/                    Data preparation, training helpers, evaluation
data/training/              Slot and response training JSONL files
data/evaluation/            Hold-out and challenge fixtures
data/processed/             Processed MultiWOZ restaurant records
data/samples/               Tiny sample dataset for tests and smoke runs
reports/                    Committed metrics and dataset reports
models/                     Local trained adapters kept for reproducibility
notebooks/                  Kaggle/Colab training notes retained for now
docs/                       Architecture, evaluation and training notes
tests/                      Unit and integration tests
```

## Installation

The default rule-baseline system and tests do not require a model download.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional local LLM inference:

```powershell
pip install -r requirements-llm.txt
```

Optional LoRA/QLoRA training dependencies:

```powershell
pip install -r requirements-qlora.txt
```

If PyTorch needs to be installed or repaired on CPU:

```powershell
python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu
```

## Data Preparation

The full raw MultiWOZ dataset is not committed. Place or clone the lecturer
provided MultiWOZ repository locally, then run:

```powershell
python scripts/prepare_multiwoz.py --multiwoz-path C:\path\to\multiwoz
```

For a workspace where the MultiWOZ clone is one folder above this project:

```powershell
python scripts/prepare_multiwoz.py --multiwoz-path "..\multiwoz"
```

If the full dataset is unavailable, use the sample records:

```powershell
python scripts/prepare_multiwoz.py --sample-if-missing
```

Tests and smoke demos can always use `--sample-data`.

## Run The System

Terminal chatbot:

```powershell
python scripts/run_chat.py --sample-data --debug
```

Web demo:

```powershell
python scripts/run_web.py --sample-data
```

Open `http://127.0.0.1:5000`.

Demo users:

```text
user1 / pass123
user2 / pass123
admin / pass123
```

The admin account opens the local dashboard with transcripts, booking records,
intent distribution, fallback/clarification counts and latency summaries.

Use the trained strict slot adapter, if present:

```powershell
python scripts/run_web.py --enable-llm --slot-model-name models/slot-extractor-lora-strict --debug
```

Enable optional response-model generation as a separate experiment:

```powershell
python scripts/run_web.py --enable-llm --enable-response-llm --slot-model-name models/slot-extractor-lora-strict --response-model-name google/flan-t5-base
```

## Evaluation And Metrics

Run tests:

```powershell
python -m pytest
```

Run the default evaluation:

```powershell
python scripts/evaluate.py --sample-data --slot-fixture data/evaluation/slot_eval_cases.jsonl
```

Run the trained strict slot adapter evaluation, if the adapter is available:

```powershell
python scripts/evaluate.py --sample-data --slot-fixture data/evaluation/slot_eval_cases.jsonl --enable-llm --slot-model-name models/slot-extractor-lora-strict --report-path reports/evaluation_lora_strict_eval.json
python scripts/evaluate.py --sample-data --slot-fixture data/evaluation/slot_challenge_cases.jsonl --enable-llm --slot-model-name models/slot-extractor-lora-strict --report-path reports/evaluation_lora_strict_challenge.json
python scripts/compare_model_reports.py --report strict_eval=reports/evaluation_lora_strict_eval.json --report strict_challenge=reports/evaluation_lora_strict_challenge.json
```

Current committed headline metrics:

| Report | Intent accuracy | Slot F1 | Retrieval R@3 | Task success | JSON leakage | Groundedness | Avg latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Strict eval | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 2824 ms |
| Strict challenge | 0.7250 | 0.9327 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 2846 ms |

See:

- `reports/evaluation_matrix.md`
- `reports/challenge_evaluation_matrix.md`
- `reports/response_generation_comparison_trained.md`
- `reports/response_generation_challenge_trained.md`
- `reports/evaluation_lora_strict_eval.json`
- `reports/evaluation_lora_strict_challenge.json`
- `docs/evaluation_plan.md`

The reports separate raw model behaviour from final guarded system behaviour.
Fallback success is not presented as raw LLM accuracy.

## Response Generation Dataset

The response-generator data is synthetic but grounded in processed MultiWOZ
restaurant records. It is useful for optional response LoRA training and for
demonstrating prompt/evaluation hygiene.

Regenerate the committed split:

```powershell
python scripts/build_response_training_data.py `
  --train-count 800 `
  --eval-count 160 `
  --challenge-count 100 `
  --seed 6062026
```

Validate leakage:

```powershell
python scripts/check_response_data_leakage.py
```

Current split counts:

| Split | Count |
| --- | ---: |
| Train | 800 |
| Standard eval | 160 |
| Challenge | 100 |

The shared prompt builder is `src/restaurant_assistant/response_prompt.py`.
The gold/reference response is never included in the model prompt. The
deterministic baseline response remains available only for comparison rows and
safety fallback after model rejection.

Evaluate response generation without loading models:

```powershell
python scripts/evaluate_response_generation.py
```

The current no-model response report has 160 cases, deterministic and final
guarded groundedness of `1.0000`, and raw model columns marked as skipped.

## Training Notes

Strict slot-extractor training is documented in:

```text
notebooks/kaggle_train_strict_slot_extractor.md
docs/training_on_kaggle.md
```

Optional response LoRA training is documented in:

```text
docs/llm_and_qlora.md
```

Training outputs should be written under `models/` for adapters and `outputs/`
for disposable run metadata. Copy only report-ready evidence into `reports/`.

## Safety And Scope Boundaries

- Restaurant domain only; hotel, train, taxi, attraction and payment requests
  are rejected or redirected.
- Booking records are local proof-of-concept records, not live reservations.
- The assistant does not claim live availability, reviews, ratings, payment
  processing, verified halal status or allergy safety.
- Responses must be grounded in loaded restaurant evidence and must not invent
  addresses, phone numbers, postcodes, food types or restaurant names.
- The local web app stores SQLite demo accounts, sessions, chat turns and
  booking records. It is for demonstration and evaluation, not deployment.
