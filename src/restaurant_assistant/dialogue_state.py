"""Session-level dialogue state for the restaurant assistant."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from restaurant_assistant.preprocessing import normalize_area, normalize_food, normalize_price, normalize_time


SEARCH_SLOTS = ("food", "area", "pricerange")
BOOKING_SLOTS = ("day", "time", "people")


@dataclass
class DialogueState:
    """Track restaurant preferences and session booking details."""

    food: str | None = None
    area: str | None = None
    pricerange: str | None = None
    day: str | None = None
    booking_date: str | None = None
    time: str | None = None
    people: int | None = None
    selected_restaurant: dict[str, Any] | None = None
    booking_restaurant: dict[str, Any] | None = None
    booking_status: str = "none"
    booking_reference: str | None = None
    pending_day_modifier: str | None = None
    conversation_history: list[dict[str, str]] = field(default_factory=list)

    def update_slots(self, new_slots: dict[str, Any]) -> None:
        """Update known state slots, ignoring empty or unsupported keys."""

        for key, value in new_slots.items():
            if value in (None, ""):
                continue
            if key == "food":
                self.food = normalize_food(value)
            elif key == "area":
                self.area = normalize_area(value)
            elif key == "pricerange":
                self.pricerange = normalize_price(value)
            elif key == "day":
                self.day = str(value).strip().lower()
            elif key == "booking_date":
                self.booking_date = str(value).strip()
            elif key == "time":
                self.time = normalize_time(str(value))
            elif key == "people":
                try:
                    people = int(value)
                except (TypeError, ValueError):
                    continue
                if people > 0:
                    self.people = people
            elif key == "selected_restaurant" and isinstance(value, dict):
                self.selected_restaurant = value

    def missing_search_slots(self, required: Iterable[str] = SEARCH_SLOTS) -> list[str]:
        return [slot for slot in required if not getattr(self, slot)]

    def missing_booking_slots(self, *, include_restaurant: bool = True) -> list[str]:
        missing = [slot for slot in BOOKING_SLOTS if not getattr(self, slot)]
        if include_restaurant and not self.selected_restaurant:
            missing.insert(0, "restaurant")
        return missing

    def has_search_constraint(self) -> bool:
        return any(getattr(self, slot) for slot in SEARCH_SLOTS)

    def add_turn(self, user_message: str, assistant_message: str, *, timestamp: str | None = None) -> None:
        turn = {"user": user_message, "assistant": assistant_message}
        if timestamp:
            turn["timestamp"] = timestamp
        self.conversation_history.append(turn)

    def reset(self) -> None:
        self.food = None
        self.area = None
        self.pricerange = None
        self.day = None
        self.booking_date = None
        self.time = None
        self.people = None
        self.selected_restaurant = None
        self.booking_restaurant = None
        self.booking_status = "none"
        self.booking_reference = None
        self.pending_day_modifier = None
        self.conversation_history.clear()

    def to_dict(self) -> dict[str, Any]:
        return {
            "food": self.food,
            "area": self.area,
            "pricerange": self.pricerange,
            "day": self.day,
            "booking_date": self.booking_date,
            "time": self.time,
            "people": self.people,
            "selected_restaurant": self.selected_restaurant,
            "booking_restaurant": self.booking_restaurant,
            "booking_status": self.booking_status,
            "booking_reference": self.booking_reference,
            "pending_day_modifier": self.pending_day_modifier,
            "conversation_history": list(self.conversation_history),
        }
