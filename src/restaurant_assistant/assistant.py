"""High-level orchestration for the restaurant assistant pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from collections.abc import Callable

from restaurant_assistant.booking import BookingManager
from restaurant_assistant.config import Settings, get_settings
from restaurant_assistant.data_loader import load_restaurants
from restaurant_assistant.date_utils import now_in_timezone, resolve_relative_day
from restaurant_assistant.dialogue_state import BOOKING_SLOTS, DialogueState
from restaurant_assistant.llm_generator import GroundedResponseGenerator
from restaurant_assistant.ranking import RankedRestaurant, rank_candidates
from restaurant_assistant.retrieval import RestaurantRetriever, RetrievedRestaurant
from restaurant_assistant.slot_extraction import extract_slots


@dataclass(frozen=True)
class AssistantResponse:
    response: str
    debug: dict[str, Any] = field(default_factory=dict)


class RestaurantAssistant:
    """Coordinate extraction, state, retrieval, ranking, generation and booking."""

    def __init__(
        self,
        restaurants: list[dict[str, Any]] | None = None,
        *,
        settings: Settings | None = None,
        use_sample: bool = False,
        enable_llm: bool | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.clock = clock or (lambda: now_in_timezone(self.settings.timezone))
        self.state = DialogueState()
        self.restaurants = restaurants if restaurants is not None else load_restaurants(self.settings, use_sample=use_sample)
        self.retriever = RestaurantRetriever().fit(self.restaurants)
        llm_enabled = self.settings.enable_llm if enable_llm is None else enable_llm
        self.generator = GroundedResponseGenerator(enable_llm=llm_enabled, model_name=self.settings.model_name)
        self.booking = BookingManager()

    def process(self, user_message: str, *, debug: bool = False) -> AssistantResponse:
        turn_time = self.clock()
        extraction = extract_slots(
            user_message,
            use_llm=False,
            model_name=self.settings.model_name,
        )
        relative_resolution = self._resolve_temporal_slots(extraction.slots, turn_time)
        retrieved: list[RetrievedRestaurant] = []
        ranked: list[RankedRestaurant] = []
        generation_mode = "direct"
        named_restaurant = self._find_named_restaurant(user_message)
        named_restaurant_changed = bool(named_restaurant) and not self._same_restaurant(
            named_restaurant, self.state.selected_restaurant or {}
        )
        search_context_changed = self._search_slots_would_change(extraction.slots)
        effective_intent = self._effective_intent(extraction.intent, extraction.slots, named_restaurant)
        starts_new_booking = effective_intent == "book" and (
            search_context_changed
            or bool(self.state.booking_reference)
            or any(slot in extraction.slots for slot in ("food", "area", "pricerange"))
        )

        if extraction.unsupported_slots:
            response = self._unsupported_slot_response(extraction.unsupported_slots)
        elif extraction.intent == "unsupported":
            response = (
                "I can only help with MultiWOZ restaurant search and simulated restaurant bookings. "
                "Try asking for a food type, area or price range."
            )
        else:
            if effective_intent in {"search", "list"} and search_context_changed:
                self.state.selected_restaurant = None
            if starts_new_booking:
                self._clear_pending_booking_details()
                if named_restaurant_changed or search_context_changed:
                    self.state.selected_restaurant = None
            self.state.update_slots(extraction.slots)

        if extraction.unsupported_slots or extraction.intent == "unsupported":
            pass
        elif effective_intent == "greeting":
            generated = self.generator.generate(user_message, self.state, intent="greeting")
            response = generated.text
            generation_mode = generated.mode
        elif effective_intent == "cancel":
            reference_response = self._reference_mismatch_response(extraction.slots)
            if reference_response:
                response = reference_response
            else:
                booking_result = self.booking.cancel_booking(self.state)
                response = booking_result.message
        elif effective_intent in {"reschedule", "correct"}:
            reference_response = self._reference_mismatch_response(extraction.slots)
            if reference_response:
                response = reference_response
            elif self.state.booking_status != "confirmed" or not self.state.booking_reference:
                response = "There is no active simulated booking to reschedule in this session."
            elif not self._has_new_booking_details(extraction.slots):
                response = (
                    "What would you like to change for the simulated booking? "
                    "Please give a new day, time or number of people."
                )
            else:
                booking_result = self.booking.reschedule_booking(self.state)
                if booking_result.missing_slots and booking_result.missing_slots != ["booking"]:
                    generated = self.generator.generate(
                        user_message,
                        self.state,
                        intent="reschedule",
                        missing_slots=booking_result.missing_slots,
                    )
                    response = generated.text
                    generation_mode = generated.mode
                else:
                    response = booking_result.message
        elif effective_intent == "alternative":
            if not self.state.has_search_constraint():
                response = "Tell me a food type, area or price range first, then I can suggest alternatives."
            else:
                retrieved, ranked = self._rank_current_options(user_message, exclude_selected=True, top_k=self.settings.top_k)
                response = self._format_ranked_list(
                    ranked,
                    empty_message="I do not have another matching restaurant in the loaded MultiWOZ records.",
                    prefix="Other matching options:",
                )
        elif effective_intent == "list":
            if not self.state.has_search_constraint():
                response = "Please give at least one filter, such as area, food type or price range, before I list restaurants."
            else:
                retrieved, ranked = self._rank_current_options(user_message, exclude_selected=False, top_k=5)
                if ranked:
                    self.state.selected_restaurant = ranked[0].record
                response = self._format_ranked_list(
                    ranked,
                    empty_message="I could not find restaurants matching those filters in the loaded MultiWOZ records.",
                    prefix="Matching restaurants:",
                )
        elif effective_intent == "booking_info":
            reference_response = self._reference_mismatch_response(extraction.slots)
            if reference_response:
                response = reference_response
            else:
                response = self._booking_info_response()
        elif effective_intent == "book":
            if named_restaurant:
                self.state.selected_restaurant = named_restaurant
            retrieved, ranked = self._select_restaurant_if_needed(user_message)
            booking_result = self.booking.create_booking(self.state)
            if booking_result.missing_slots:
                generated = self.generator.generate(
                    user_message,
                    self.state,
                    ranked,
                    intent="book",
                    missing_slots=booking_result.missing_slots,
                )
                response = generated.text
                generation_mode = generated.mode
            else:
                response = booking_result.message
        elif effective_intent == "search":
            if not self.state.has_search_constraint():
                generated = self.generator.generate(
                    user_message,
                    self.state,
                    intent="search",
                    missing_slots=self.state.missing_search_slots(),
                )
                response = generated.text
                generation_mode = generated.mode
            else:
                retrieved = self.retriever.search(user_message, self.state, top_k=self.settings.top_k)
                ranked = rank_candidates(retrieved, self.state, top_k=self.settings.top_k)
                if ranked and not ranked[0].missing_unmatched_constraints:
                    self.state.selected_restaurant = ranked[0].record
                generated = self.generator.generate(user_message, self.state, ranked, intent="search")
                response = generated.text
                generation_mode = generated.mode
        else:
            response = (
                "I can only help with MultiWOZ restaurant search and simulated restaurant bookings. "
                "Try asking for a restaurant by food type, area or price range."
            )

        turn_timestamp = turn_time.isoformat(timespec="seconds")
        self.state.add_turn(user_message, response, timestamp=turn_timestamp)
        return AssistantResponse(
            response=response,
            debug=self._debug(extraction, effective_intent, retrieved, ranked, generation_mode, turn_timestamp, relative_resolution)
            if debug
            else {},
        )

    def reset(self) -> None:
        self.state.reset()

    def _select_restaurant_if_needed(self, user_message: str) -> tuple[list[RetrievedRestaurant], list[RankedRestaurant]]:
        if self.state.selected_restaurant:
            return [], []
        if not self.state.has_search_constraint():
            return [], []
        retrieved = self.retriever.search(user_message, self.state, top_k=self.settings.top_k)
        ranked = rank_candidates(retrieved, self.state, top_k=self.settings.top_k)
        if ranked and not ranked[0].missing_unmatched_constraints:
            self.state.selected_restaurant = ranked[0].record
        return retrieved, ranked

    def _find_named_restaurant(self, user_message: str) -> dict[str, Any] | None:
        normalized_message = user_message.lower()
        matches = []
        for record in self.restaurants:
            name = str(record.get("name", "")).lower().strip()
            if not name:
                continue
            significant_tokens = [token for token in name.split() if token not in {"the", "restaurant", "and", "bar"}]
            if name in normalized_message or (
                significant_tokens and all(token in normalized_message for token in significant_tokens)
            ):
                matches.append(record)
        matches.sort(key=lambda record: len(str(record.get("name", ""))), reverse=True)
        if matches:
            return matches[0]
        return None

    def _rank_current_options(
        self,
        user_message: str,
        *,
        exclude_selected: bool,
        top_k: int,
    ) -> tuple[list[RetrievedRestaurant], list[RankedRestaurant]]:
        retrieval_k = max(top_k + 5, self.settings.top_k + 5)
        retrieved = self.retriever.search(user_message, self.state, top_k=retrieval_k)
        ranked = rank_candidates(retrieved, self.state, top_k=retrieval_k)
        if exclude_selected and self.state.selected_restaurant:
            ranked = [item for item in ranked if not self._same_restaurant(item.record, self.state.selected_restaurant)]
        return retrieved, ranked[:top_k]

    def _same_restaurant(self, first: dict[str, Any], second: dict[str, Any]) -> bool:
        first_id = first.get("source_id")
        second_id = second.get("source_id")
        if first_id and second_id:
            return first_id == second_id
        return str(first.get("name", "")).lower() == str(second.get("name", "")).lower()

    def _format_ranked_list(self, ranked: list[RankedRestaurant], *, empty_message: str, prefix: str) -> str:
        if not ranked:
            return empty_message
        parts = []
        for index, item in enumerate(ranked, start=1):
            record = item.record
            name = record.get("name", "unknown restaurant")
            food = record.get("food", "unknown food")
            area = record.get("area", "unknown area")
            price = record.get("pricerange", "unknown price")
            parts.append(f"{index}. {name} ({price} {food}, {area})")
        return prefix + " " + " ".join(parts)

    def _has_new_booking_details(self, slots: dict[str, Any]) -> bool:
        return any(slot in slots for slot in BOOKING_SLOTS)

    def _effective_intent(
        self,
        raw_intent: str,
        slots: dict[str, Any],
        named_restaurant: dict[str, Any] | None,
    ) -> str:
        if raw_intent != "unknown" or not self._has_new_booking_details(slots):
            return raw_intent
        if self.state.booking_status == "confirmed" and self.state.booking_reference:
            return "reschedule"
        if named_restaurant or self.state.selected_restaurant or self.state.has_search_constraint():
            return "book"
        return raw_intent

    def _resolve_temporal_slots(self, slots: dict[str, Any], turn_time: datetime) -> dict[str, Any] | None:
        relative_day = slots.pop("relative_day", None)
        if not relative_day:
            return None
        resolved_day = resolve_relative_day(
            str(relative_day),
            reference_time=turn_time,
            active_booking_day=self.state.day if self.state.booking_status == "confirmed" else None,
        )
        slots["day"] = resolved_day
        return {"relative_day": relative_day, "resolved_day": resolved_day, "turn_timestamp": turn_time.isoformat(timespec="seconds")}

    def _search_slots_would_change(self, slots: dict[str, Any]) -> bool:
        for slot in ("food", "area", "pricerange"):
            if slot in slots and getattr(self.state, slot) != slots[slot]:
                return True
        return False

    def _clear_pending_booking_details(self) -> None:
        self.state.day = None
        self.state.time = None
        self.state.people = None

    def _reference_mismatch_response(self, slots: dict[str, Any]) -> str | None:
        requested = slots.get("booking_reference")
        if not requested or not self.state.booking_reference:
            return None
        if requested != self.state.booking_reference:
            return (
                f"I only have active simulated booking {self.state.booking_reference} in this session. "
                f"I cannot look up {requested} as a real booking reference."
            )
        return None

    def _booking_info_response(self) -> str:
        if not self.state.booking_reference:
            return "I do not have a simulated booking reference in this session yet."
        restaurant = self.state.booking_restaurant or self.state.selected_restaurant
        name = restaurant.get("name", "the selected restaurant") if restaurant else "the selected restaurant"
        status = self.state.booking_status
        return (
            f"Simulated booking {self.state.booking_reference} is {status} for {name}: "
            f"{self.state.day} at {self.state.time} for {self.state.people} people. "
            "This is session-only and is not a live restaurant booking."
        )

    def _unsupported_slot_response(self, unsupported_slots: dict[str, str]) -> str:
        if unsupported_slots.get("area"):
            return (
                "The loaded MultiWOZ restaurant database only supports these areas: centre, north, south, east and west. "
                f"I do not have a '{unsupported_slots['area']}' restaurant area to search."
            )
        if unsupported_slots.get("day"):
            return "Please give a specific booking day, such as Monday, Tuesday or Sunday."
        return "I could not validate one of those preferences against the loaded MultiWOZ restaurant data."

    def _debug(
        self,
        extraction: Any,
        effective_intent: str,
        retrieved: list[RetrievedRestaurant],
        ranked: list[RankedRestaurant],
        generation_mode: str,
        turn_timestamp: str,
        relative_resolution: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "turn_timestamp": turn_timestamp,
            "intent": extraction.intent,
            "effective_intent": effective_intent,
            "slots": extraction.slots,
            "unsupported_slots": extraction.unsupported_slots,
            "relative_day_resolution": relative_resolution,
            "slot_extraction_used_llm": extraction.used_llm,
            "dialogue_state": self.state.to_dict(),
            "retrieved_restaurants": [
                {"name": item.record.get("name"), "similarity": round(item.similarity, 4)} for item in retrieved
            ],
            "ranking": [
                {
                    "name": item.record.get("name"),
                    "score": item.score,
                    "matched_constraints": item.matched_constraints,
                    "missing_unmatched_constraints": item.missing_unmatched_constraints,
                    "explanation": item.explanation,
                }
                for item in ranked
            ],
            "generation_mode": generation_mode,
        }
