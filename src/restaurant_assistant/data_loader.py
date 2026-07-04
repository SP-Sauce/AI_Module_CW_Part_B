"""Data loading utilities for processed and sample restaurant records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from restaurant_assistant.config import Settings, get_settings
from restaurant_assistant.preprocessing import preprocess_restaurants


RestaurantRecord = dict[str, Any]


def load_records_from_file(path: Path) -> list[RestaurantRecord]:
    """Load restaurant records from JSON, JSONL or CSV."""

    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list of records in {path}")
        return preprocess_restaurants(data)
    if suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    records.append(json.loads(line))
        return preprocess_restaurants(records)
    if suffix == ".csv":
        return preprocess_restaurants(pd.read_csv(path).to_dict(orient="records"))
    raise ValueError(f"Unsupported restaurant data format: {path.suffix}")


def load_sample_restaurants(settings: Settings | None = None) -> list[RestaurantRecord]:
    active_settings = settings or get_settings()
    return load_records_from_file(active_settings.sample_data_path)


def load_restaurants(
    settings: Settings | None = None,
    *,
    use_sample: bool = False,
    processed_path: Path | None = None,
) -> list[RestaurantRecord]:
    """Load processed data when available, otherwise fall back to samples."""

    active_settings = settings or get_settings()
    if use_sample:
        return load_sample_restaurants(active_settings)
    candidate = processed_path or active_settings.processed_restaurant_path
    if candidate.exists():
        return load_records_from_file(candidate)
    return load_sample_restaurants(active_settings)

