# Processed Data

`scripts/prepare_multiwoz.py` writes cleaned restaurant records here by default:

```text
data/processed/restaurants.jsonl
```

Generated records preserve display fields such as `name`, `address`, `postcode`
and `phone`, while also adding normalized fields for matching.

