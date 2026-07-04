"""Load and normalize MultiWOZ restaurant database records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


EMPTY_VALUES = {"", "?", "none", "nan", "not mentioned", "dontcare", "don't care"}

AREA_ALIASES = {
    "center": "centre",
    "city": "centre",
    "city center": "centre",
    "city centre": "centre",
    "centre": "centre",
    "north": "north",
    "south": "south",
    "east": "east",
    "west": "west",
}

PRICE_ALIASES = {
    "cheap": "cheap",
    "inexpensive": "cheap",
    "budget": "cheap",
    "low cost": "cheap",
    "moderate": "moderate",
    "moderately": "moderate",
    "midrange": "moderate",
    "mid range": "moderate",
    "mid-range": "moderate",
    "expensive": "expensive",
    "upmarket": "expensive",
    "upscale": "expensive",
}


def clean_text(value: Any) -> str:
    """Return a stripped display string, converting missing markers to empty."""

    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in EMPTY_VALUES:
        return ""
    return " ".join(text.split())


def normalize_text(value: Any) -> str:
    """Normalize text for matching."""

    return clean_text(value).lower()


def normalize_area(value: Any) -> str:
    text = normalize_text(value)
    return AREA_ALIASES.get(text, text)


def normalize_price(value: Any) -> str:
    text = normalize_text(value)
    return PRICE_ALIASES.get(text, text)


def normalize_food(value: Any) -> str:
    return normalize_text(value)


def normalize_time(value: str) -> str:
    """Normalize a time string to HH:MM where possible."""

    text = normalize_text(value).replace(".", ":")
    if not text:
        return ""
    suffix = ""
    if text.endswith("am") or text.endswith("pm"):
        suffix = text[-2:]
        text = text[:-2].strip()
    if ":" in text:
        hour_text, minute_text = text.split(":", 1)
    else:
        hour_text, minute_text = text, "00"
    try:
        hour = int(hour_text)
        minute = int(minute_text[:2])
    except ValueError:
        return value
    if suffix == "pm" and hour < 12:
        hour += 12
    if suffix == "am" and hour == 12:
        hour = 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return value
    return f"{hour:02d}:{minute:02d}"


def normalize_record(record: dict[str, Any], source_id: str | None = None) -> dict[str, Any]:
    """Normalize a raw MultiWOZ restaurant record while preserving display fields."""

    normalized = {
        "source_id": clean_text(record.get("source_id") or record.get("id") or source_id or ""),
        "name": clean_text(record.get("name")),
        "food": clean_text(record.get("food")),
        "area": clean_text(record.get("area")),
        "pricerange": clean_text(record.get("pricerange") or record.get("price")),
        "address": clean_text(record.get("address") or record.get("addr")),
        "postcode": clean_text(record.get("postcode") or record.get("post")),
        "phone": clean_text(record.get("phone")),
        "type": clean_text(record.get("type") or "restaurant"),
    }
    normalized["name_norm"] = normalize_text(normalized["name"])
    normalized["food_norm"] = normalize_food(normalized["food"])
    normalized["area_norm"] = normalize_area(normalized["area"])
    normalized["pricerange_norm"] = normalize_price(normalized["pricerange"])
    return normalized


def preprocess_restaurants(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Clean a sequence of raw restaurant records."""

    cleaned: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        normalized = normalize_record(record, source_id=f"restaurant-{index}")
        key = normalized["name_norm"] or normalized["source_id"]
        if not key or key in seen:
            continue
        seen.add(key)
        cleaned.append(normalized)
    return cleaned


def find_restaurant_db_path(raw_multiwoz_path: Path) -> Path | None:
    """Find `restaurant_db.json` in common MultiWOZ repository layouts."""

    candidates = [
        raw_multiwoz_path / "restaurant_db.json",
        raw_multiwoz_path / "db" / "restaurant_db.json",
        raw_multiwoz_path / "data" / "MultiWOZ_2.1" / "db" / "restaurant_db.json",
        raw_multiwoz_path / "data" / "MultiWOZ_2.2" / "db" / "restaurant_db.json",
        raw_multiwoz_path / "data" / "multi-woz" / "db" / "restaurant_db.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if raw_multiwoz_path.exists():
        matches = list(raw_multiwoz_path.rglob("restaurant_db.json"))
        if matches:
            return matches[0]
    return None


def load_multiwoz_restaurant_db(raw_multiwoz_path: Path) -> list[dict[str, Any]]:
    """Load raw restaurant records from a local MultiWOZ checkout."""

    db_path = find_restaurant_db_path(raw_multiwoz_path)
    if db_path is None:
        raise FileNotFoundError(f"Could not find restaurant_db.json under {raw_multiwoz_path}")
    with db_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of restaurant records in {db_path}")
    return data


def save_jsonl(records: Iterable[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")
