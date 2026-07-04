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
- simulated booking language is used;
- unavailable details are not invented.

Optional BLEU, ROUGE or BERTScore hooks can be added if the dependencies are
installed, but they are not required for the runnable MVP.

## End-to-End Task Success

Scripted conversations test common demo flows:

- restaurant search from food, area and price;
- clarification for underspecified search;
- simulated booking after a selected restaurant exists;
- simulated rescheduling;
- simulated cancellation.

The script reports task success rate and latency per turn.

## Ablation

`scripts/run_ablation.py` compares:

- final system: state tracking plus retrieval plus grounded generation;
- retrieval-only baseline;
- no-state-tracking baseline;
- LLM-only placeholder baseline where practical.

The aim is not to claim production performance. It is to show that state
tracking and retrieval each add measurable value over simpler baselines.

