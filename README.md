# Retrieval-Grounded LLM Dialogue System for Restaurant Support

## Project status

This repository contains the **completed Part B prototype** for the 6CM606
*Frontiers in Artificial Intelligence and Data Science* coursework.

The repository includes:

- the runnable restaurant-support application;
- trained slot-extraction and response-generation LoRA adapters;
- fixed training, standard-evaluation and challenge-evaluation datasets;
- final evaluation matrices and trained-model comparison reports;
- unit, integration, persistence, safety and leakage checks;
- training metadata recording seeds, hyperparameters, losses and hardware.

---

## 1. Project overview

The system is a retrieval-grounded, task-oriented dialogue assistant for the
**MultiWOZ restaurant domain**. It supports:

- restaurant search by cuisine, area and price range;
- clarification when important preferences are missing;
- retrieval of grounded restaurant details such as address, postcode and phone;
- multi-turn conversational context;
- local proof-of-concept booking creation;
- booking lookup, rescheduling and cancellation;
- explicit handling of unsupported requests.

The implementation follows the Part A design decision to combine
knowledge-grounded retrieval with lightweight dialogue-state tracking.

---

## 2. Main technical contribution

The project does not allow a language model to control the entire dialogue
pipeline. Instead, it combines trained LoRA adapters with deterministic system
controls:

```text
User message
    ↓
Intent and slot extraction
    ↓
Strict parsing, constrained repair and rule fallback
    ↓
DialogueState update
    ↓
Exact restaurant constraints
    ↓
TF-IDF retrieval and preference-fit ranking
    ↓
ResponsePlan containing approved evidence
    ↓
LoRA or deterministic response generation
    ↓
Evidence and safety validation
    ↓
Safe fallback when validation fails
    ↓
Local booking-state and SQLite persistence
```

This design separates:

- **neural language understanding and generation**;
- **deterministic state transitions and booking logic**;
- **retrieved restaurant evidence**;
- **user-visible response validation**.

The final reported figures therefore distinguish raw model behaviour from the
performance of the protected hybrid system.

---

## 3. Assessment alignment

| Assessment requirement | Repository evidence |
| --- | --- |
| Interpret and process customer queries using LLM-based techniques | FLAN-T5-base LoRA slot extractor with strict parsing, repair metrics and deterministic fallback |
| Generate accurate and contextually appropriate responses | Retrieval-grounded `ResponsePlan`, trained response LoRA, evidence validation and deterministic fallback |
| Manage conversational context | `DialogueState` tracks preferences, selected restaurant, booking details, references and conversation history |
| Provide a working AI component | CLI application, Flask web application, local accounts, SQLite persistence and administrator dashboard |
| Compare against baselines | Standard and challenge matrices compare rules, an unadapted base model and the selected LoRA hybrid |
| Evaluate with appropriate metrics | Intent accuracy, exact-slot accuracy, precision, recall, F1, groundedness, rejection, fallback and latency |
| Demonstrate reproducibility | Fixed seed, committed datasets, training metadata, model adapters, evaluation scripts and leakage checks |
| Address safety and limitations | Unsupported-claim checks, evidence validation, scoped persistence, limitations and future-work discussion |

---

## 4. Implemented components

- Intent classification for search, list, information, booking, rescheduling,
  cancellation, booking lookup, greetings and unsupported requests.
- Slot extraction for cuisine, area, price range, booking day, time, party size
  and booking reference.
- Optional trained LoRA slot extraction through Transformers.
- Strict JSON parsing, constrained repair and deterministic rule fallback.
- Explicit multi-turn dialogue-state tracking.
- Exact filtering followed by TF-IDF restaurant retrieval.
- Transparent preference-fit ranking.
- Structured `ResponsePlan` generation using approved public evidence fields.
- Completed and evaluated response-generator LoRA.
- Deterministic natural-language generation as a protected fallback.
- Validation against invented restaurant names, addresses, postcodes and phone
  numbers.
- Rejection of unsupported claims about live availability, payments, reviews,
  halal certification and allergy safety.
- Local account-scoped chat sessions and SQLite booking records.
- Administrator dashboard for transcripts, bookings and operational metrics.
- Unit, integration, web, persistence, model, safety and leakage tests.

---

## 5. Repository structure

```text
src/restaurant_assistant/   Core assistant, state, retrieval, NLG and web app
scripts/                    Data preparation, evaluation and training scripts
data/training/              Fixed slot and response training data
data/evaluation/            Standard and challenge hold-out fixtures
data/processed/             Processed MultiWOZ restaurant records
data/samples/               Small smoke-test dataset
models/                     Completed slot and response LoRA adapters
reports/                    Final metrics, matrices, metadata and integrity reports
tests/                      Unit and integration test suite
```

---

## 6. Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install the local LLM runtime dependencies when using the trained adapters:

```powershell
pip install -r requirements-llm.txt
```

The deterministic baseline and most tests can run without loading a model.

---

## 7. Run the completed system

### 7.1 Web application with processed restaurant data

```powershell
python scripts/run_web.py
```

Open:

```text
http://127.0.0.1:5000
```

### 7.2 Trained slot LoRA with protected deterministic responses

This is the recommended stable demonstration configuration:

```powershell
python scripts/run_web.py `
  --enable-llm `
  --slot-model-name models/slot-extractor-lora-strict `
  --debug
```

### 7.3 Full trained slot and response LoRA configuration

```powershell
python scripts/run_web.py `
  --enable-llm `
  --slot-model-name models/slot-extractor-lora-strict `
  --enable-response-llm `
  --response-model-name models/response-generator-lora
```

The response validator and deterministic fallback remain active when the
response LoRA is enabled.

### 7.4 Terminal application

```powershell
python scripts/run_chat.py `
  --enable-llm `
  --slot-model-name models/slot-extractor-lora-strict `
  --debug
```

### 7.5 Smoke-test fallback

Use the bundled sample records only when the processed dataset is unavailable:

```powershell
python scripts/run_web.py --sample-data
```

### Local demo-only accounts

```text
user1 / pass123
user2 / pass123
admin / pass123
```

---

## 8. Testing and integrity checks

Run the complete test suite:

```powershell
python -m pytest
```

Check slot-training and evaluation leakage:

```powershell
python scripts/check_data_leakage.py
```

Check response-dataset integrity and cross-split leakage:

```powershell
python scripts/check_response_data_leakage.py
```

The test suite covers core dialogue behaviour, retrieval, response validation,
web flows, account scoping, SQLite persistence and booking-state changes.

---

## 9. Evaluation

### 9.1 Slot-extraction evaluation

The final matrices compare three configurations on the same labelled fixtures:

- deterministic rules;
- unadapted `google/flan-t5-small` with protected fallback;
- selected FLAN-T5-base LoRA with repair and protected fallback.

Final evidence:

```text
reports/evaluation_matrix.md
reports/challenge_evaluation_matrix.md
reports/evaluation_matrix.json
reports/challenge_evaluation_matrix.json
```

### Standard 50-case fixture

| Configuration | Intent accuracy | Exact-slot accuracy | Slot F1 | Fallbacks | Mean slot latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rules | 1.0000 | 1.0000 | 1.0000 | 0/50 | 0.0013 s |
| Base LLM hybrid | 1.0000 | 1.0000 | 1.0000 | 50/50 | 0.5711 s |
| Selected LoRA hybrid | 1.0000 | 1.0000 | 1.0000 | 44/50 | 1.6330 s |

### Independent 80-case challenge fixture

| Configuration | Intent accuracy | Exact-slot accuracy | Slot F1 | Fallbacks | Mean slot latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rules | 0.7000 | 0.9000 | 0.9327 | 0/80 | 0.0020 s |
| Base LLM hybrid | 0.7000 | 0.9000 | 0.9327 | 80/80 | 0.1971 s |
| Selected LoRA hybrid | 0.7250 | 0.9000 | 0.9327 | 61/80 | 1.6058 s |

### Critical interpretation

The selected slot LoRA produced a small challenge-intent improvement from
`0.7000` to `0.7250`, but it did not improve final challenge slot F1 over the
rules baseline.

The selected adapter produced **zero strict-valid JSON outputs** on both
fixtures. Constrained repair recovered recognisable content, but deterministic
fallback remained necessary in `44/50` standard cases and `61/80` challenge
cases.

Therefore, the final values above are **protected hybrid-system results**, not
standalone LoRA accuracy.

---

## 10. Response-generation evaluation

The completed response LoRA is compared against the unadapted FLAN-T5-base model
and the final guarded response path.

Final evidence:

```text
reports/response_generation_comparison_trained.md
reports/response_generation_challenge_trained.md
reports/response_generation_comparison_trained.json
reports/response_generation_challenge_trained.json
```

| Dataset and mode | Accepted | Rejected | Raw groundedness | Fallbacks | Final groundedness | Mean latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard — pretrained | 82/160 | 78 | 0.5125 | 0 | 0.5125 | 0.8743 s |
| Standard — response LoRA | 151/160 | 9 | 0.9437 | 0 | 0.9437 | 0.6619 s |
| Standard — guarded final | 151/160 | 9 | 0.9437 | 9 | 1.0000 | 0.6619 s |
| Challenge — pretrained | 53/100 | 47 | 0.5300 | 0 | 0.5300 | 0.9797 s |
| Challenge — response LoRA | 92/100 | 8 | 0.9200 | 0 | 0.9200 | 0.6975 s |
| Challenge — guarded final | 92/100 | 8 | 0.9200 | 8 | 1.0000 | 0.6975 s |

### Interpretation

The response LoRA produced the clearest model-level improvement:

- standard groundedness increased from `0.5125` to `0.9437`;
- challenge groundedness increased from `0.5300` to `0.9200`;
- standard acceptance increased from `82/160` to `151/160`;
- challenge acceptance increased from `53/100` to `92/100`;
- recorded mean latency decreased in both evaluations.

The remaining rejected responses were replaced by deterministic grounded
fallbacks. Final groundedness of `1.0000` is therefore a protected-system result,
not raw response-model accuracy.

---

## 11. Scripted end-to-end system checks

System-level checks are reported separately from the 50-case and 80-case slot
fixtures.

The scripted flows cover:

- restaurant search;
- clarification;
- booking creation;
- rescheduling;
- cancellation.

The committed strict evaluation reports record:

| Metric | Standard scripted flow | Challenge scripted flow |
| --- | ---: | ---: |
| Retrieval Recall@3 | 1.0000 | 1.0000 |
| Task-success rate | 1.0000 | 1.0000 |
| JSON/debug leakage rate | 0.0000 | 0.0000 |
| Final groundedness pass rate | 1.0000 | 1.0000 |
| Mean end-to-end latency | 2.824 s | 2.846 s |

These figures describe the scripted system flows and should not be interpreted
as measurements over every slot-fixture case.

Evidence:

```text
reports/evaluation_lora_strict_eval.json
reports/evaluation_lora_strict_challenge.json
```

---

## 12. Completed training evidence

The trained adapters are already included. 

### Slot LoRA

| Item | Recorded value |
| --- | --- |
| Base model | `google/flan-t5-base` |
| Training examples | 1,870 |
| Development examples | 330 |
| Seed | `6062026` |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | `q`, `v` |
| Maximum steps | 800 |
| Evaluation loss | 0.0216 |
| Hardware | 2 × Tesla T4 |
| Four-bit loading | No |

Evidence:

```text
models/slot-extractor-lora-strict/training_metadata.json
```

### Response LoRA

| Item | Recorded value |
| --- | --- |
| Base model | `google/flan-t5-base` |
| Training examples | 800 |
| Standard evaluation examples | 160 |
| Challenge examples | 100 |
| Seed | `6062026` |
| Maximum steps | 300 |
| Batch size | 1 |
| Gradient accumulation | 8 |
| Training runtime | 936.3 seconds |
| Training loss | 1.5217 |
| Evaluation loss | 0.5894 |
| Hardware | 2 × Tesla T4 |
| Four-bit loading | No |

Evidence:

```text
reports/response_lora_training_metadata.json
models/response-generator-lora/
```

The completed runs used **LoRA**, not QLoRA. 
QLoRA remains a possible lower-memory alternative, but it is not
presented as the final completed configuration.

---

## 13. Dataset and leakage controls

The response-generation data are synthetic but grounded in trusted processed
MultiWOZ restaurant records.

| Split | Cases |
| --- | ---: |
| Training | 800 |
| Standard evaluation | 160 |
| Challenge evaluation | 100 |

The dataset generation process used:

- seed `6062026`;
- 110 processed restaurant records;
- restaurant-level train/evaluation separation;
- zero cross-split input overlap;
- zero cross-split user-message overlap;
- duplicate checks;
- booking-reference grounding checks;
- response-safety validation.

Evidence:

```text
reports/response_generation_dataset_report.md
reports/response_generation_dataset_report.json
```

Synthetic supervision may favour the target response style, so automatic
results do not replace independent human evaluation.

---

## 14. Safety and scope boundaries

The application is intentionally restricted to the restaurant domain.

It does not:

- contact restaurants;
- confirm live availability;
- process payments;
- use external ratings or review services;
- claim verified halal status;
- provide allergy-safety guarantees;
- invent restaurant evidence.

The response validator checks for:

- invented restaurant names;
- unsupported addresses, postcodes and phone numbers;
- JSON or debug leakage;
- prompt-label leakage;
- unsupported availability, payment, review, halal and allergy claims.

The web application stores local demo accounts, sessions, transcripts and
booking records in SQLite. It is a coursework demonstration environment rather
than a production deployment.
