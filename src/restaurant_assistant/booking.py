"""Simulated restaurant booking operations."""

from __future__ import annotations

import secrets
import string
from dataclasses import dataclass, field

from restaurant_assistant.dialogue_state import DialogueState


@dataclass(frozen=True)
class BookingResult:
    success: bool
    message: str
    missing_slots: list[str] = field(default_factory=list)


class BookingManager:
    """Manage simulated booking state without claiming real availability."""

    def create_booking(self, state: DialogueState) -> BookingResult:
        missing = state.missing_booking_slots(include_restaurant=True)
        if missing:
            return BookingResult(False, "Missing information for simulated booking.", missing)
        state.booking_reference = self._make_reference()
        state.booking_status = "confirmed"
        state.booking_restaurant = state.selected_restaurant
        name = self._restaurant_name(state)
        message = (
            f"I have created a simulated booking for {name} on {state.day} at {state.time} "
            f"for {state.people} people. Your simulated reference is {state.booking_reference}. "
            "This is not a live restaurant booking."
        )
        return BookingResult(True, message)

    def reschedule_booking(self, state: DialogueState) -> BookingResult:
        if state.booking_status != "confirmed" or not state.booking_reference:
            return BookingResult(False, "There is no active simulated booking to reschedule.", ["booking"])
        missing = state.missing_booking_slots(include_restaurant=True)
        if missing:
            return BookingResult(False, "Missing information for simulated rescheduling.", missing)
        name = self._restaurant_name(state)
        message = (
            f"I have updated the simulated booking {state.booking_reference} for {name} "
            f"to {state.day} at {state.time} for {state.people} people."
        )
        return BookingResult(True, message)

    def cancel_booking(self, state: DialogueState) -> BookingResult:
        if state.booking_status != "confirmed" or not state.booking_reference:
            return BookingResult(False, "There is no active simulated booking to cancel.", ["booking"])
        reference = state.booking_reference
        state.booking_status = "cancelled"
        message = f"I have cancelled the simulated booking {reference}. No real restaurant booking was affected."
        return BookingResult(True, message)

    def _make_reference(self) -> str:
        alphabet = string.ascii_uppercase + string.digits
        return "SIM-" + "".join(secrets.choice(alphabet) for _ in range(6))

    def _restaurant_name(self, state: DialogueState) -> str:
        restaurant = state.booking_restaurant or state.selected_restaurant
        if restaurant and restaurant.get("name"):
            return str(restaurant["name"])
        return "the selected restaurant"
