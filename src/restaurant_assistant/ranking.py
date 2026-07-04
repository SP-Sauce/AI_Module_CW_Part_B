"""Transparent preference-fit ranking for retrieved restaurants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from restaurant_assistant.dialogue_state import DialogueState
from restaurant_assistant.preprocessing import normalize_area, normalize_food, normalize_price, normalize_text
from restaurant_assistant.retrieval import RetrievedRestaurant


@dataclass(frozen=True)
class RankedRestaurant:
    record: dict[str, Any]
    score: float
    matched_constraints: list[str]
    missing_unmatched_constraints: list[str]
    explanation: str
    similarity: float = 0.0


def rank_candidates(
    candidates: Iterable[RetrievedRestaurant | dict[str, Any]],
    state: DialogueState | dict[str, Any],
    top_k: int = 3,
) -> list[RankedRestaurant]:
    """Rank candidates using exact slot matches plus retrieval similarity."""

    state_dict = state.to_dict() if isinstance(state, DialogueState) else dict(state)
    ranked = [_score_candidate(candidate, state_dict) for candidate in candidates]
    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked[:top_k]


def _score_candidate(candidate: RetrievedRestaurant | dict[str, Any], state: dict[str, Any]) -> RankedRestaurant:
    if isinstance(candidate, RetrievedRestaurant):
        record = candidate.record
        similarity = candidate.similarity
    elif "record" in candidate:
        record = candidate["record"]
        similarity = float(candidate.get("similarity", 0.0))
    else:
        record = candidate
        similarity = float(candidate.get("similarity", 0.0))

    checks = {
        "food": (normalize_food(state.get("food")), normalize_text(record.get("food_norm") or record.get("food"))),
        "area": (normalize_area(state.get("area")), normalize_text(record.get("area_norm") or record.get("area"))),
        "pricerange": (
            normalize_price(state.get("pricerange")),
            normalize_text(record.get("pricerange_norm") or record.get("pricerange")),
        ),
    }
    matched: list[str] = []
    unmatched: list[str] = []
    score = float(similarity)
    for slot, (wanted, actual) in checks.items():
        if not wanted:
            continue
        if actual == wanted:
            matched.append(slot)
            score += 3.0
        else:
            unmatched.append(slot)
            score -= 1.0
    if not record.get("name"):
        score -= 0.2
    explanation_parts = []
    if matched:
        explanation_parts.append("matched " + ", ".join(matched))
    if unmatched:
        explanation_parts.append("unmatched " + ", ".join(unmatched))
    explanation_parts.append(f"tf-idf similarity {similarity:.3f}")
    return RankedRestaurant(
        record=record,
        score=round(score, 4),
        matched_constraints=matched,
        missing_unmatched_constraints=unmatched,
        explanation="; ".join(explanation_parts),
        similarity=similarity,
    )

