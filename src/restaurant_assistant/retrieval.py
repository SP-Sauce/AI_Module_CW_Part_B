"""TF-IDF restaurant retrieval with exact constraint filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from restaurant_assistant.dialogue_state import DialogueState
from restaurant_assistant.preprocessing import normalize_area, normalize_food, normalize_price, normalize_text


@dataclass(frozen=True)
class RetrievedRestaurant:
    record: dict[str, Any]
    similarity: float


class RestaurantRetriever:
    """Retrieve restaurant records from known constraints and TF-IDF text."""

    def __init__(self) -> None:
        self.restaurants: list[dict[str, Any]] = []
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.matrix = None

    def fit(self, restaurants: Iterable[dict[str, Any]]) -> "RestaurantRetriever":
        self.restaurants = list(restaurants)
        documents = [self._build_search_text(record) for record in self.restaurants]
        if documents:
            self.matrix = self.vectorizer.fit_transform(documents)
        else:
            self.matrix = None
        return self

    def search(self, query: str, state: DialogueState | dict[str, Any] | None = None, top_k: int = 3) -> list[RetrievedRestaurant]:
        if not self.restaurants or self.matrix is None:
            return []
        state_dict = self._state_to_dict(state)
        candidate_indices = self._constraint_indices(state_dict)
        if not candidate_indices:
            candidate_indices = list(range(len(self.restaurants)))

        query_text = self._build_query_text(query, state_dict)
        query_vector = self.vectorizer.transform([query_text or "restaurant"])
        scores = cosine_similarity(query_vector, self.matrix)[0]
        ordered = sorted(candidate_indices, key=lambda index: scores[index], reverse=True)
        return [RetrievedRestaurant(self.restaurants[index], float(scores[index])) for index in ordered[:top_k]]

    def retrieve_by_constraints(
        self, state: DialogueState | dict[str, Any] | None = None, top_k: int = 3
    ) -> list[RetrievedRestaurant]:
        state_dict = self._state_to_dict(state)
        candidate_indices = self._constraint_indices(state_dict)
        if not candidate_indices:
            candidate_indices = list(range(len(self.restaurants)))
        return [RetrievedRestaurant(self.restaurants[index], 0.0) for index in candidate_indices[:top_k]]

    def _constraint_indices(self, state: dict[str, Any]) -> list[int]:
        constraints = {
            "food_norm": normalize_food(state.get("food")),
            "area_norm": normalize_area(state.get("area")),
            "pricerange_norm": normalize_price(state.get("pricerange")),
        }
        active = {field: value for field, value in constraints.items() if value}
        if not active:
            return list(range(len(self.restaurants)))
        raw_fields = {
            "food_norm": "food",
            "area_norm": "area",
            "pricerange_norm": "pricerange",
        }
        matches = []
        for index, record in enumerate(self.restaurants):
            if all(
                normalize_text(record.get(field) or record.get(raw_fields[field])) == value
                for field, value in active.items()
            ):
                matches.append(index)
        return matches

    def _build_search_text(self, record: dict[str, Any]) -> str:
        fields = [
            record.get("name", ""),
            record.get("food", ""),
            record.get("area", ""),
            record.get("pricerange", ""),
            record.get("address", ""),
            record.get("postcode", ""),
        ]
        return " ".join(normalize_text(field) for field in fields if field)

    def _build_query_text(self, query: str, state: dict[str, Any]) -> str:
        parts = [
            query,
            state.get("food") or "",
            state.get("area") or "",
            state.get("pricerange") or "",
        ]
        return " ".join(normalize_text(part) for part in parts if part)

    def _state_to_dict(self, state: DialogueState | dict[str, Any] | None) -> dict[str, Any]:
        if state is None:
            return {}
        if isinstance(state, DialogueState):
            return state.to_dict()
        return dict(state)
