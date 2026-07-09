# Evaluation Plan

## Slot Extraction

The evaluation script can use either the original small fixture in
`tests/fixtures/slot_cases.json` or the hold-out file in
`data/evaluation/slot_eval_cases.jsonl`. The hold-out file contains fresh
natural restaurant-assistant turns that are not duplicated from the training
JSONL.

Final slot metrics:

- intent accuracy;
- exact slot-object accuracy;
- micro slot precision, recall and F1 over key/value slot pairs;
- invalid JSON or parse-error count where an LLM extractor is active;
- mean slot-extraction latency;
- LLM enabled flag, LLM-used case count and slot model name.

The main report should compare the same hold-out fixture across:

- `baseline_rule_based`: deterministic rules only;
- `base_llm`: lightweight `google/flan-t5-small` prompted JSON extraction;
- `qlora_adapter`: the optional `models/slot-extractor-qlora` adapter when it
  has been trained from FLAN-T5-small;
- `qlora_adapter_base`: the stronger optional FLAN-T5-base adapter at
  `models/slot-extractor-qlora-base`.

## Retrieval

Synthetic queries are generated from restaurant records, for example:

```text
cheap italian restaurant in the south
```

The expected target is the source record. Metrics are:

- Recall@1
- Recall@3
- Mean Reciprocal Rank

These metrics show whether the TF-IDF retrieval stage can recover relevant
restaurant records from grounded evidence.

## Response Generation

The default evaluation checks safety properties of the grounded generator:

- generated replies are grounded in retrieved records;
- missing information is handled with clarification;
- booking responses avoid claims about live restaurant availability;
- unavailable details are not invented.

In LLM mode, the response-generation output includes the active generation
mode, such as `transformers:google/flan-t5-small`, or `template` if model
loading fails and the safe fallback is used.

Optional BLEU, ROUGE or BERTScore hooks can be added if the dependencies are
installed, but they are not required for the runnable MVP.

## End-to-End Task Success

Scripted conversations test common demo flows:

- restaurant search from food, area and price;
- clarification for underspecified search;
- booking after a selected restaurant exists;
- rescheduling;
- cancellation.

## Web And Persistence Checks

The test suite covers the Flask API flow and SQLite persistence:

- users can register and log in before using the chat app;
- a browser session receives a generated session id;
- chat turns are stored against that account-owned session;
- booking creation, rescheduling and cancellation update the SQLite booking
  record for the same session;
- account history is scoped so one user cannot see another user's sessions or
  bookings through the normal chat APIs.

The local admin dashboard supports the final demo and report by showing:

- all local session transcripts;
- booking records and status counts;
- intent distribution for newly saved turns;
- booking conversion as a task-success proxy;
- average turns per session and latency for newly saved turns;
- clarification, limitation and fallback counts for failure analysis.

The script reports task success rate and latency per turn.

## Hold-Out Evaluation Design

`data/evaluation/slot_eval_cases.jsonl` uses the same schema as the training
file:

```json
{"text":"...","intent":"...","slots":{}}
```

It covers search, list, booking, reschedule, correction, cancellation, booking
lookup, booking lists, restaurant details, filter help, cuisine help, dish
preference, distance questions, table view requests, greetings, thanks and
unsupported requests. The examples include typos and natural phrasing so the
evaluation is not just a copy of the rule patterns or the training examples.

## Leakage Prevention

Before reporting hold-out results, run:

```powershell
python scripts/check_data_leakage.py
```

The checker compares normalized `text` fields between
`data/training/slot_instruction_examples.jsonl` and
`data/evaluation/slot_eval_cases.jsonl`. It exits with a non-zero status if an
evaluation turn is an exact duplicate after lowercasing, punctuation removal and
whitespace normalization.

## Ablation

`scripts/run_ablation.py` compares:

- final system: state tracking plus retrieval plus grounded generation;
- retrieval-only baseline;
- no-state-tracking baseline;
- LLM-only placeholder baseline where practical.

The aim is not to claim production performance. It is to show that state
tracking and retrieval each add measurable value over simpler baselines.

## QLoRA Adapter

If a slot-extraction adapter is trained with
`scripts/train_qlora_slot_extractor.py`, rerun:

```powershell
Remove-Item -Recurse -Force models/slot-extractor-qlora -ErrorAction SilentlyContinue
python scripts/train_qlora_slot_extractor.py --base-model google/flan-t5-small
python scripts/evaluate.py --sample-data --slot-fixture data/evaluation/slot_eval_cases.jsonl --enable-llm --slot-model-name models/slot-extractor-qlora
```

Adapters trained before the strict `User: ...` / `JSON:` prompt-format fix must
be deleted and retrained. Training and inference now use the same prompt.

The report should compare the base LLM extractor, the QLoRA adapter where
available, and the rule fallback on the same labelled fixture.

The base `google/flan-t5-small` model did not reliably follow the structured
output instruction and often generated labels such as `location` or `food`
instead of JSON. After fine-tuning, the QLoRA adapter learned recognisable
intent/slot fragments, but some outputs still omitted structural braces, for
example `"intent":"list","slots":"area":"centre"`. The runtime applies a
narrow repair step that accepts only `intent`, `slots` and allowed slot keys,
then runs the normal deterministic slot validation. Rule-based fallback still
prevents a malformed or unrepaired model response from causing task failure.

Final evaluation keeps these outcomes separate: raw strict-JSON parse failures,
successful constrained repairs, valid-or-repaired outputs, unrepaired failures
and rule fallback usage are all reported. Raw and repaired output previews are
retained in case details rather than presenting fallback accuracy as LLM
accuracy.

The FLAN-T5-base adapter is evaluated through the same validation, repair,
weak-repair detection, trusted-intent/slot checks and rule fallback as the
small model. Model size alone is not treated as evidence that structured output
is reliable.

Use the matrix command for reproducible JSON and Markdown outputs:

```powershell
python scripts/run_evaluation_matrix.py --sample-data
```

## Limitations

The hold-out set is intentionally small enough for coursework inspection, so it
does not estimate production accuracy. Base LLM results may fall back to the
rule extractor if local model dependencies or model files are unavailable. QLoRA
results should be labelled as skipped unless `models/slot-extractor-qlora`
exists in the local or Colab runtime. Booking evaluation remains
proof-of-concept: it verifies state changes and references, not real restaurant
availability.
