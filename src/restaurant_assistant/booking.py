"""Restaurant booking record operations for the proof-of-concept assistant."""

from __future__ import annotations

import secrets
import string
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from restaurant_assistant.date_utils import format_booking_date
from restaurant_assistant.dialogue_state import DialogueState

if TYPE_CHECKING:
    from restaurant_assistant.storage import BookingStore


@dataclass(frozen=True)
class BookingResult:
    success: bool
    message: str
    missing_slots: list[str] = field(default_factory=list)


class BookingManager:
    """Manage session booking state without claiming external availability."""

    def __init__(
        self,
        *,
        store: "BookingStore | None" = None,
        session_id: str | None = None,
        user_id: int | None = None,
    ) -> None:
        self.store = store
        self.session_id = session_id
        self.user_id = user_id

    def create_booking(self, state: DialogueState) -> BookingResult:
        missing = state.missing_booking_slots(include_restaurant=True)
        if missing:
            return BookingResult(False, "Missing information for booking.", missing)
        state.booking_reference = self._make_reference()
        state.booking_status = "confirmed"
        state.booking_restaurant = state.selected_restaurant
        name = self._restaurant_name(state)
        date_text = format_booking_date(state.booking_date, state.day)
        message = (
            f"I have created a booking record for {name} on {date_text} at {state.time} "
            f"for {state.people} people. Your reference is {state.booking_reference}."
        )
        self._persist(state)
        return BookingResult(True, message)

    def reschedule_booking(self, state: DialogueState) -> BookingResult:
        if state.booking_status != "confirmed" or not state.booking_reference:
            return BookingResult(False, "There is no active booking to reschedule.", ["booking"])
        missing = state.missing_booking_slots(include_restaurant=True)
        if missing:
            return BookingResult(False, "Missing information for rescheduling.", missing)
        name = self._restaurant_name(state)
        date_text = format_booking_date(state.booking_date, state.day)
        message = (
            f"I have updated booking {state.booking_reference} for {name} "
            f"to {date_text} at {state.time} for {state.people} people."
        )
        self._persist(state)
        return BookingResult(True, message)

    def cancel_booking(self, state: DialogueState) -> BookingResult:
        if state.booking_status != "confirmed" or not state.booking_reference:
            return BookingResult(False, "There is no active booking to cancel.", ["booking"])
        reference = state.booking_reference
        state.booking_status = "cancelled"
        message = f"I have cancelled booking {reference}."
        self._persist(state)
        return BookingResult(True, message)

    def list_bookings(self) -> list[dict[str, Any]]:
        if self.store is None:
            return []
        if self.user_id is not None:
            return self.store.list_user_bookings(self.user_id)
        if not self.session_id:
            return []
        return self.store.list_bookings(self.session_id)

    def get_booking(self, reference: str) -> dict[str, Any] | None:
        if self.store is None:
            return None
        if self.user_id is not None:
            return self.store.get_user_booking(self.user_id, reference)
        if not self.session_id:
            return None
        return self.store.get_booking(self.session_id, reference)

    def cancel_bookings_except_restaurant(self, keep_restaurant: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if self.store is None or self.user_id is None:
            return [], []
        kept = self.store.kept_user_bookings_for_restaurant(self.user_id, keep_restaurant)
        if not kept:
            return [], []
        cancelled = self.store.cancel_user_bookings_except_restaurant(self.user_id, keep_restaurant)
        return cancelled, kept

    def _make_reference(self) -> str:
        alphabet = string.ascii_uppercase + string.digits
        return "BK-" + "".join(secrets.choice(alphabet) for _ in range(6))

    def _restaurant_name(self, state: DialogueState) -> str:
        restaurant = state.booking_restaurant or state.selected_restaurant
        if restaurant and restaurant.get("name"):
            return str(restaurant["name"])
        return "the selected restaurant"

    def _persist(self, state: DialogueState) -> None:
        if self.store is not None and self.session_id:
            self.store.upsert_booking(self.session_id, state)
