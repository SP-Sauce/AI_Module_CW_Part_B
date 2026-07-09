"""Structured response plans for the restaurant assistant."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from restaurant_assistant.ranking import RankedRestaurant
from restaurant_assistant.retrieval import RetrievedRestaurant


PUBLIC_RESTAURANT_FIELDS = (
    "name",
    "food",
    "area",
    "pricerange",
    "address",
    "postcode",
    "phone",
    "type",
)


@dataclass
class ResponsePlan:
    """Grounded, non-user-facing plan consumed by the NLG layer."""

    dialogue_act: str
    user_intent: str
    constraints: dict[str, Any] = field(default_factory=dict)
    retrieved_restaurants: list[dict[str, Any]] = field(default_factory=list)
    selected_restaurant: dict[str, Any] | None = None
    missing_constraints: list[str] = field(default_factory=list)
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    next_action: str = "respond"
    warnings: list[str] = field(default_factory=list)
    internal_notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def public_restaurant(record: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return only fields safe to use in customer-facing responses."""

    if not record:
        return None
    return {key: record.get(key) for key in PUBLIC_RESTAURANT_FIELDS if record.get(key) not in (None, "")}


def public_restaurants(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a list of customer-safe restaurant records."""

    safe_records = []
    for record in records:
        safe_record = public_restaurant(record)
        if safe_record:
            safe_records.append(safe_record)
    return safe_records


def ranked_to_public_restaurants(ranked: Iterable[RankedRestaurant]) -> list[dict[str, Any]]:
    """Return public restaurant records from ranked retrieval results."""

    return public_restaurants(item.record for item in ranked)


def retrieved_to_public_restaurants(retrieved: Iterable[RetrievedRestaurant]) -> list[dict[str, Any]]:
    """Return public restaurant records from raw retrieval results."""

    return public_restaurants(item.record for item in retrieved)
