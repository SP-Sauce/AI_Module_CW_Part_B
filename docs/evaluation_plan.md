# Evaluation Plan

## Slot Extraction

The evaluation script uses a small labelled fixture covering intent, food, area,
price range, day, time and number of people. It reports exact slot accuracy and
intent accuracy. This checks whether the system can interpret customer queries
before state tracking or retrieval.

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

The default evaluation checks safety properties of the fallback generator:

- generated replies are grounded in retrieved records;
- missing information is handled with clarification;
- booking responses avoid claims about live restaurant availability;
- unavailable details are not invented.

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

## Ablation

`scripts/run_ablation.py` compares:

- final system: state tracking plus retrieval plus grounded generation;
- retrieval-only baseline;
- no-state-tracking baseline;
- LLM-only placeholder baseline where practical.

The aim is not to claim production performance. It is to show that state
tracking and retrieval each add measurable value over simpler baselines.
