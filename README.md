# Retrieval-Grounded LLM Dialogue System for Restaurant Search and Simulated Booking

This repository contains a Part B implementation for the 6CM606 Frontiers in
Artificial Intelligence and Data Science coursework. It builds a CPU-feasible
MultiWOZ restaurant assistant that can search restaurant records, maintain
session state and create clearly simulated booking confirmations.

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
-> simulated booking-state update
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The default tests and demo work without downloading an LLM. LLM generation mode
uses Hugging Face Transformers if a supported backend/model is available. The
safe fallback generator is used automatically if model loading fails.

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

## Run The Chatbot

```powershell
python scripts/run_chat.py --sample-data --debug
```

Example flow:

```text
You: I need a cheap Italian restaurant in the centre.
Assistant: I found pizza hut city centre...

You: Can you book it for Friday at 7 for 2 people?
Assistant: I have created a simulated booking...

You: Move it to Saturday.
Assistant: I have updated the simulated booking...

You: Cancel it.
Assistant: I have cancelled the simulated booking...
```

Enable attempted Transformers generation:

```powershell
python scripts/run_chat.py --sample-data --enable-llm
```

## Run Tests

```powershell
pytest
```

All tests run against the sample dataset and do not require internet access.

## Run Evaluation

```powershell
python scripts/evaluate.py --sample-data
python scripts/run_ablation.py --sample-data
```

The evaluation reports slot metrics, retrieval metrics, task success and
latency. The ablation script compares the final stateful retrieval system with
simpler baselines.

## Safety And Limitations

- Bookings are simulated only and use references such as `SIM-AB12CD`.
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
- State is session-only and is reset when the process exits.

## Optional Stretch Work

PEFT or QLoRA fine-tuning can be discussed as a future extension, but it is not
required for the core prototype. The assessed MVP is designed around retrieval,
state tracking and grounded response generation.
