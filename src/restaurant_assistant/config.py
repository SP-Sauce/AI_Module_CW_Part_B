"""Configuration helpers for project paths and runtime settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Runtime settings with conservative defaults."""

    raw_multiwoz_path: Path = PROJECT_ROOT / "data" / "raw"
    processed_restaurant_path: Path = PROJECT_ROOT / "data" / "processed" / "restaurants.jsonl"
    sample_data_path: Path = PROJECT_ROOT / "data" / "samples" / "sample_restaurants.json"
    booking_db_path: Path = PROJECT_ROOT / "data" / "runtime" / "bookings.sqlite3"
    top_k: int = 3
    enable_llm: bool = False
    enable_response_llm: bool = False
    model_name: str = "google/flan-t5-base"
    response_model_name: str = "google/flan-t5-base"
    enable_response_pretrained_fallback: bool = True
    slot_model_name: str = "google/flan-t5-small"
    slot_num_beams: int = 1
    timezone: str = "Europe/London"


def get_settings() -> Settings:
    """Build settings from environment variables where present."""

    raw_path = Path(os.getenv("MULTIWOZ_PATH", str(PROJECT_ROOT / "data" / "raw")))
    top_k = int(os.getenv("TOP_K", "3"))
    response_model = os.getenv(
        "HF_RESPONSE_MODEL_NAME",
        os.getenv("HF_MODEL_NAME", "google/flan-t5-base"),
    )
    return Settings(
        raw_multiwoz_path=raw_path,
        booking_db_path=Path(os.getenv("BOOKING_DB_PATH", str(PROJECT_ROOT / "data" / "runtime" / "bookings.sqlite3"))),
        top_k=top_k,
        enable_llm=_env_bool("ENABLE_LLM", False),
        enable_response_llm=_env_bool("ENABLE_RESPONSE_LLM", False),
        model_name=response_model,
        response_model_name=response_model,
        enable_response_pretrained_fallback=_env_bool("ENABLE_RESPONSE_PRETRAINED_FALLBACK", True),
        slot_model_name=os.getenv("HF_SLOT_MODEL_NAME", "google/flan-t5-small"),
        slot_num_beams=int(os.getenv("HF_SLOT_NUM_BEAMS", "1")),
        timezone=os.getenv("ASSISTANT_TIMEZONE", "Europe/London"),
    )
