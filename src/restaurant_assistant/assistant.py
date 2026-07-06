"""High-level orchestration for the restaurant assistant pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from collections.abc import Callable

from restaurant_assistant.booking import BookingManager
from restaurant_assistant.config import Settings, get_settings
from restaurant_assistant.data_loader import load_restaurants
from restaurant_assistant.date_utils import (
    format_booking_date,
    now_in_timezone,
    resolve_relative_day,
    resolve_relative_day_date,
    resolve_weekday_date,
)
from restaurant_assistant.dialogue_state import BOOKING_SLOTS, DialogueState
from restaurant_assistant.llm_generator import GroundedResponseGenerator
from restaurant_assistant.ranking import RankedRestaurant, rank_candidates
from restaurant_assistant.retrieval import RestaurantRetriever, RetrievedRestaurant
from restaurant_assistant.slot_extraction import extract_slots
from restaurant_assistant.storage import BookingStore


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
        booking_store: BookingStore | None = None,
        session_id: str | None = None,
        user_id: int | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.clock = clock or (lambda: now_in_timezone(self.settings.timezone))
        self.state = DialogueState()
        self.restaurants = restaurants if restaurants is not None else load_restaurants(self.settings, use_sample=use_sample)
        self.retriever = RestaurantRetriever().fit(self.restaurants)
        llm_enabled = self.settings.enable_llm if enable_llm is None else enable_llm
        self.generator = GroundedResponseGenerator(enable_llm=llm_enabled, model_name=self.settings.model_name)
        self.booking = BookingManager(store=booking_store, session_id=session_id, user_id=user_id)

    def process(self, user_message: str, *, debug: bool = False) -> AssistantResponse:
        turn_time = self.clock()
        extraction = extract_slots(
            user_message,
            use_llm=False,
            model_name=self.settings.model_name,
        )
        if self._mentions_next_week_without_day(user_message, extraction.slots):
            self.state.pending_day_modifier = "next_week"
        relative_resolution = self._resolve_temporal_slots(extraction.slots, turn_time, extraction.intent)
        retrieved: list[RetrievedRestaurant] = []
        ranked: list[RankedRestaurant] = []
        generation_mode = "direct"
        named_restaurant = self._find_named_restaurant(user_message)
        named_restaurant_changed = bool(named_restaurant) and not self._same_restaurant(
            named_restaurant, self.state.selected_restaurant or {}
        )
        excluded_food = self._apply_context_resets(user_message, extraction.intent, extraction.slots)
        search_context_changed = self._search_slots_would_change(extraction.slots)
        effective_intent = self._effective_intent(extraction.intent, extraction.slots, named_restaurant)
        duplicate_booking_request = self._is_duplicate_booking_request(user_message, effective_intent)
        starts_new_booking = effective_intent == "book" and (
            search_context_changed
            or bool(self.state.booking_reference)
            or any(slot in extraction.slots for slot in ("food", "area", "pricerange"))
        ) and not duplicate_booking_request

        if extraction.unsupported_slots.get("food"):
            self.state.food = None
            self.state.selected_restaurant = None

        if extraction.unsupported_slots:
            response = self._unsupported_slot_response(extraction.unsupported_slots)
        elif extraction.intent == "unsupported":
            response = (
                "I can only help with MultiWOZ restaurant search and restaurant booking records. "
                "Try asking for a food type, area or price range."
            )
        else:
            if effective_intent in {"search", "list", "dish_preference"} and search_context_changed:
                self.state.selected_restaurant = None
            if effective_intent == "dish_preference":
                self.state.food = None
                self.state.selected_restaurant = None
            if extraction.slots.get("cuisine_group"):
                self.state.food = None
                if effective_intent != "alternative":
                    self.state.selected_restaurant = None
            if starts_new_booking:
                self._clear_pending_booking_details()
                if named_restaurant_changed or search_context_changed:
                    self.state.selected_restaurant = None
            slots_to_update = extraction.slots
            if effective_intent == "distance_info":
                slots_to_update = {
                    key: value
                    for key, value in extraction.slots.items()
                    if key in {"day", "booking_date", "time", "people", "booking_reference"}
                }
            self.state.update_slots(slots_to_update)

        if extraction.unsupported_slots or extraction.intent == "unsupported":
            pass
        elif effective_intent == "greeting":
            generated = self.generator.generate(user_message, self.state, intent="greeting")
            response = generated.text
            generation_mode = generated.mode
        elif effective_intent == "thanks":
            response = "You're welcome. I can help with the current session booking if you need to view, change or cancel it."
        elif effective_intent == "date_clarification":
            response = self._next_week_day_prompt()
        elif effective_intent == "cancel":
            is_bulk_cancel, keep_restaurant = self._bulk_cancel_exception(user_message, named_restaurant)
            if is_bulk_cancel:
                response = self._bulk_cancel_except_response(keep_restaurant)
            else:
                reference_response = self._resolve_requested_booking_reference(extraction.slots)
                if reference_response:
                    response = reference_response
                else:
                    booking_result = self.booking.cancel_booking(self.state)
                    response = booking_result.message
        elif effective_intent in {"reschedule", "correct"}:
            reference_response = self._resolve_requested_booking_reference(extraction.slots)
            if reference_response:
                response = reference_response
            elif self.state.booking_status != "confirmed" or not self.state.booking_reference:
                response = "There is no active booking to reschedule in this session."
            elif not self._has_new_booking_details(extraction.slots):
                response = (
                    "What would you like to change for the booking? "
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
            if extraction.slots.get("cuisine_group"):
                retrieved, ranked = self._rank_food_candidates(
                    user_message,
                    extraction.slots.get("food_candidates", []),
                    top_k=self.settings.top_k,
                    exclude_selected=True,
                )
                response = self._cuisine_group_response(
                    extraction.slots.get("cuisine_group"),
                    extraction.slots.get("food_candidates", []),
                    ranked,
                    prefix="Other matching options:",
                    empty_message="I do not have another matching restaurant in those cuisine categories.",
                )
            elif not self.state.has_search_constraint():
                response = "Tell me a food type, area or price range first, then I can suggest alternatives."
            else:
                retrieved, ranked = self._rank_current_options(
                    user_message,
                    exclude_selected=True,
                    top_k=self.settings.top_k,
                    exclude_food=excluded_food,
                )
                response = self._format_ranked_list(
                    ranked,
                    empty_message="I do not have another matching restaurant in the loaded MultiWOZ records.",
                    prefix="Other matching options:",
                )
        elif effective_intent == "list":
            if extraction.slots.get("cuisine_group"):
                retrieved, ranked = self._rank_food_candidates(
                    user_message,
                    extraction.slots.get("food_candidates", []),
                    top_k=10,
                    exclude_selected=False,
                )
            else:
                retrieved, ranked = self._rank_current_options(
                    user_message,
                    exclude_selected=False,
                    top_k=10 if self._is_explicit_full_list_request(user_message) or not self.state.has_search_constraint() else 5,
                    exclude_food=excluded_food,
                )
            if ranked:
                self.state.selected_restaurant = ranked[0].record
            if extraction.slots.get("cuisine_group"):
                response = self._cuisine_group_response(
                    extraction.slots.get("cuisine_group"),
                    extraction.slots.get("food_candidates", []),
                    ranked,
                    prefix="Matching restaurants:",
                    empty_message="I could not find restaurants in those cuisine categories in the loaded MultiWOZ records.",
                )
            else:
                response = self._format_ranked_list(
                    ranked,
                    empty_message="I could not find restaurants matching those filters in the loaded MultiWOZ records.",
                    prefix="Matching restaurants:",
                )
        elif effective_intent == "booking_info":
            reference_response = self._resolve_requested_booking_reference(extraction.slots)
            if reference_response:
                response = reference_response
            else:
                response = self._booking_info_response()
        elif effective_intent == "booking_list":
            response = self._booking_list_response()
        elif effective_intent == "distance_info":
            restaurant = named_restaurant or self.state.selected_restaurant
            if named_restaurant:
                self.state.selected_restaurant = named_restaurant
            response = self._distance_info_response(restaurant, extraction.slots.get("area"))
        elif effective_intent == "restaurant_info":
            restaurant = named_restaurant or self.state.booking_restaurant or self.state.selected_restaurant
            if restaurant:
                self.state.selected_restaurant = restaurant
                response = self._restaurant_detail_response(restaurant)
            else:
                response = "Which restaurant would you like the address or contact details for?"
        elif effective_intent == "filter_info":
            response = self._filter_info_response()
        elif effective_intent == "cuisine_help":
            response = self._cuisine_help_response()
        elif effective_intent == "dish_preference":
            retrieved, ranked = self._rank_food_candidates(
                user_message,
                extraction.slots.get("food_candidates", []),
                top_k=self.settings.top_k,
            )
            if ranked:
                self.state.selected_restaurant = ranked[0].record
            response = self._dish_preference_response(
                extraction.slots.get("dish"),
                extraction.slots.get("food_candidates", []),
                ranked,
            )
        elif effective_intent == "book":
            if duplicate_booking_request:
                if self.state.booking_status != "confirmed" or not self.state.booking_reference:
                    response = "There is no active booking in this session to copy."
                else:
                    self.state.selected_restaurant = self.state.booking_restaurant or self.state.selected_restaurant
                    booking_result = self.booking.create_booking(self.state)
                    response = booking_result.message
            else:
                if named_restaurant:
                    self.state.selected_restaurant = named_restaurant
                retrieved, ranked = self._select_restaurant_if_needed(user_message)
                booking_result = self.booking.create_booking(self.state)
                if booking_result.missing_slots:
                    if "day" in booking_result.missing_slots and self.state.pending_day_modifier == "next_week":
                        response = self._next_week_day_prompt()
                    else:
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
                "I can only help with MultiWOZ restaurant search and restaurant booking records. "
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
        normalized_message = re.sub(r"\bpecking\b", "peking", normalized_message)
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

    def _is_explicit_full_list_request(self, user_message: str) -> bool:
        text = user_message.lower()
        return bool(
            re.search(
                r"\bshow\s+me\s+all\b|\blist\s+all\b|\ball\b.*\brestaurants?\b|\brestaurants?\b.*\ball\b|\bnot\s+in\s+(?:the\s+)?list\b",
                text,
            )
        )

    def _rank_current_options(
        self,
        user_message: str,
        *,
        exclude_selected: bool,
        top_k: int,
        exclude_food: str | None = None,
    ) -> tuple[list[RetrievedRestaurant], list[RankedRestaurant]]:
        retrieval_k = max(top_k + 5, self.settings.top_k + 5)
        retrieved = self.retriever.search(user_message, self.state, top_k=retrieval_k)
        ranked = rank_candidates(retrieved, self.state, top_k=retrieval_k)
        ranked = [item for item in ranked if not item.missing_unmatched_constraints]
        if exclude_selected and self.state.selected_restaurant:
            ranked = [item for item in ranked if not self._same_restaurant(item.record, self.state.selected_restaurant)]
        if exclude_food:
            ranked = [item for item in ranked if str(item.record.get("food", "")).lower() != exclude_food]
        if re.search(r"\b(?:no|not)\s+pizza\b", user_message.lower()):
            ranked = [item for item in ranked if "pizza" not in str(item.record.get("name", "")).lower()]
        return retrieved, ranked[:top_k]

    def _rank_food_candidates(
        self,
        user_message: str,
        foods: list[str],
        *,
        top_k: int,
        exclude_selected: bool = False,
    ) -> tuple[list[RetrievedRestaurant], list[RankedRestaurant]]:
        retrieved: list[RetrievedRestaurant] = []
        best_by_restaurant: dict[str, RankedRestaurant] = {}
        retrieval_k = max(top_k, self.settings.top_k)
        for food in foods:
            candidate_state = self.state.to_dict()
            candidate_state["food"] = food
            candidate_retrieved = self.retriever.search(
                f"{user_message} {food}",
                candidate_state,
                top_k=retrieval_k,
            )
            retrieved.extend(candidate_retrieved)
            ranked = rank_candidates(candidate_retrieved, candidate_state, top_k=retrieval_k)
            for item in ranked:
                if item.missing_unmatched_constraints:
                    continue
                key = str(item.record.get("source_id") or item.record.get("name", "")).lower()
                current = best_by_restaurant.get(key)
                if current is None or item.score > current.score:
                    best_by_restaurant[key] = item
        combined = sorted(best_by_restaurant.values(), key=lambda item: item.score, reverse=True)
        if exclude_selected and self.state.selected_restaurant:
            combined = [item for item in combined if not self._same_restaurant(item.record, self.state.selected_restaurant)]
        return retrieved, combined[:top_k]

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

    def _dish_preference_response(
        self,
        dish: str | None,
        foods: list[str],
        ranked: list[RankedRestaurant],
    ) -> str:
        dish_text = dish or "that dish"
        cuisine_text = ", ".join(food.title() for food in foods)
        lead = (
            f"I do not have dish-level menu data for {dish_text}, but it can fit these cuisine categories: "
            f"{cuisine_text}."
        )
        list_text = self._format_ranked_list(
            ranked,
            empty_message="I could not find matching restaurants for those cuisine categories in the loaded MultiWOZ records.",
            prefix="Matching restaurants:",
        )
        return f"{lead}\n{list_text}"

    def _cuisine_group_response(
        self,
        group: str | None,
        foods: list[str],
        ranked: list[RankedRestaurant],
        *,
        prefix: str,
        empty_message: str,
    ) -> str:
        group_text = group or "That cuisine group"
        cuisine_text = ", ".join(food.title() for food in foods)
        lead = (
            f"{group_text} is not a direct cuisine label in the loaded MultiWOZ records, "
            f"so I searched these supported categories: {cuisine_text}."
        )
        list_text = self._format_ranked_list(ranked, empty_message=empty_message, prefix=prefix)
        return f"{lead}\n{list_text}"

    def _has_new_booking_details(self, slots: dict[str, Any]) -> bool:
        return any(slot in slots for slot in BOOKING_SLOTS)

    def _mentions_next_week_without_day(self, user_message: str, slots: dict[str, Any]) -> bool:
        if slots.get("day"):
            return False
        return bool(re.search(r"\b(?:next|following)\s+week\b", user_message.lower()))

    def _apply_context_resets(self, user_message: str, raw_intent: str, slots: dict[str, Any]) -> str | None:
        text = user_message.lower()
        text = re.sub(r"\bresuratns\b|\bresurants\b|\bresturants\b|\bresutrants\b|\brestraunts\b", "restaurants", text)
        text = re.sub(r"\bcusines\b|\bcusine\b", "cuisines", text)
        text = re.sub(r"\bitalion\b", "italian", text)
        text = re.sub(r"\bmoderatly\b|\bmoderatley\b", "moderately", text)
        if raw_intent not in {"search", "list", "alternative"}:
            return None
        negated_food = re.search(r"\bnot\s+(?:just|only)?\s*\w+", text) or re.search(r"\bno\s+\w+", text)
        not_just_request = re.search(r"\bnot\s+(?:just|only)\b", text)
        broad_all_request = re.search(r"\blist\s+all\b|\ball\b.*\brestaurants?\b", text)
        broad_food_request = re.search(
            r"\ball\s+(?:the\s+)?restaurants?\b|\bany\s+restaurants?\b|\bother\s+cuisines?\b|\bdifferent\s+cuisines?\b",
            text,
        )
        excluded_food = str(slots.get("food") or "").lower() if negated_food else None
        if not excluded_food and broad_food_request and re.search(r"\bother\s+cuisines?\b|\bdifferent\s+cuisines?\b", text):
            excluded_food = self.state.food
        if negated_food or (broad_food_request and "food" not in slots):
            slots.pop("food", None)
            self.state.food = None
            self.state.selected_restaurant = None
        if broad_all_request and any(slot in slots for slot in ("food", "area", "pricerange")) and not not_just_request:
            for slot in ("food", "area", "pricerange"):
                if slot not in slots:
                    setattr(self.state, slot, None)
            self.state.selected_restaurant = None
        return excluded_food

    def _effective_intent(
        self,
        raw_intent: str,
        slots: dict[str, Any],
        named_restaurant: dict[str, Any] | None,
    ) -> str:
        if raw_intent == "table_view":
            last_turn = self.state.conversation_history[-1] if self.state.conversation_history else {}
            last_response = last_turn.get("assistant", "")
            if (
                "Current session booking records:" in last_response
                or "Current account booking records:" in last_response
                or self.state.booking_reference
            ):
                return "booking_list"
            if "Matching restaurants:" in last_response or "Other matching options:" in last_response:
                return "list"
            return "unknown"
        if raw_intent != "unknown" or not self._has_new_booking_details(slots):
            return raw_intent
        if self.state.booking_status == "confirmed" and self.state.booking_reference:
            return "reschedule"
        if named_restaurant or self.state.selected_restaurant or self.state.has_search_constraint():
            return "book"
        return raw_intent

    def _resolve_temporal_slots(
        self,
        slots: dict[str, Any],
        turn_time: datetime,
        raw_intent: str,
    ) -> dict[str, Any] | None:
        relative_day = slots.pop("relative_day", None)
        day_modifier = slots.pop("day_modifier", None)
        use_active_booking_date = self.state.booking_status == "confirmed" and raw_intent in {
            "reschedule",
            "correct",
            "unknown",
        }
        if relative_day:
            active_booking_date = self.state.booking_date if use_active_booking_date else None
            resolved_date = resolve_relative_day_date(
                str(relative_day),
                reference_time=turn_time,
                active_booking_date=active_booking_date,
            )
            resolved_day = resolve_relative_day(
                str(relative_day),
                reference_time=turn_time,
                active_booking_day=self.state.day if use_active_booking_date else None,
            )
            slots["day"] = resolved_day
            slots["booking_date"] = resolved_date.isoformat()
            self.state.pending_day_modifier = None
            return {
                "relative_day": relative_day,
                "resolved_day": resolved_day,
                "booking_date": resolved_date.isoformat(),
                "formatted_date": format_booking_date(resolved_date.isoformat(), resolved_day),
                "turn_timestamp": turn_time.isoformat(timespec="seconds"),
            }
        if slots.get("day"):
            effective_modifier = str(day_modifier) if day_modifier else None
            if self.state.pending_day_modifier == "next_week" and effective_modifier in {None, "next"}:
                effective_modifier = "next_week"
            resolved_date = resolve_weekday_date(
                str(slots["day"]),
                reference_time=turn_time,
                modifier=effective_modifier,
                active_booking_date=self.state.booking_date if use_active_booking_date else None,
            )
            slots["booking_date"] = resolved_date.isoformat()
            self.state.pending_day_modifier = None
            return {
                "day": slots["day"],
                "day_modifier": effective_modifier,
                "booking_date": resolved_date.isoformat(),
                "formatted_date": format_booking_date(resolved_date.isoformat(), str(slots["day"])),
                "turn_timestamp": turn_time.isoformat(timespec="seconds"),
            }
        return None

    def _search_slots_would_change(self, slots: dict[str, Any]) -> bool:
        for slot in ("food", "area", "pricerange"):
            if slot in slots and getattr(self.state, slot) != slots[slot]:
                return True
        return False

    def _clear_pending_booking_details(self) -> None:
        self.state.day = None
        self.state.booking_date = None
        self.state.time = None
        self.state.people = None

    def _resolve_requested_booking_reference(self, slots: dict[str, Any]) -> str | None:
        requested = slots.get("booking_reference")
        if not requested:
            return None
        if requested == self.state.booking_reference:
            return None
        booking = self.booking.get_booking(str(requested))
        if not booking:
            scope = "your account" if self.booking.user_id is not None else "this current session"
            privacy = (
                "Only booking records linked to your logged-in account can be opened."
                if self.booking.user_id is not None
                else "For privacy, this chat can only open booking records created in the active session."
            )
            return (
                f"I cannot find booking reference {requested} in {scope}. "
                f"{privacy}"
            )
        self._load_booking_into_state(booking)
        return None

    def _load_booking_into_state(self, booking: dict[str, Any]) -> None:
        restaurant = {
            "name": booking.get("restaurant_name"),
            "food": booking.get("food"),
            "area": booking.get("area"),
            "pricerange": booking.get("pricerange"),
            "address": booking.get("address"),
            "postcode": booking.get("postcode"),
            "phone": booking.get("phone"),
        }
        self.state.food = booking.get("food")
        self.state.area = booking.get("area")
        self.state.pricerange = booking.get("pricerange")
        self.state.day = booking.get("day")
        self.state.booking_date = booking.get("booking_date")
        self.state.time = booking.get("time")
        self.state.people = booking.get("people")
        self.state.selected_restaurant = restaurant
        self.state.booking_restaurant = restaurant
        self.state.booking_status = booking.get("status") or "none"
        self.state.booking_reference = booking.get("reference")

    def _booking_info_response(self) -> str:
        if not self.state.booking_reference:
            if self.state.selected_restaurant:
                return self._restaurant_detail_response(self.state.selected_restaurant)
            if self.booking.user_id is not None:
                return "I do not have a booking reference selected yet."
            return "I do not have a booking reference in this session yet."
        restaurant = self.state.booking_restaurant or self.state.selected_restaurant
        name = restaurant.get("name", "the selected restaurant") if restaurant else "the selected restaurant"
        status = self.state.booking_status
        return (
            f"Booking {self.state.booking_reference} is {status} for {name}: "
            f"{format_booking_date(self.state.booking_date, self.state.day)} at {self.state.time} "
            f"for {self.state.people} people."
        )

    def _booking_list_response(self) -> str:
        bookings = self.booking.list_bookings()
        if not bookings and self.state.booking_reference:
            restaurant = self.state.booking_restaurant or self.state.selected_restaurant or {}
            bookings = [
                {
                    "reference": self.state.booking_reference,
                    "restaurant_name": restaurant.get("name", "the selected restaurant"),
                    "day": self.state.day,
                    "booking_date": self.state.booking_date,
                    "time": self.state.time,
                    "people": self.state.people,
                    "status": self.state.booking_status,
                }
            ]
        if not bookings:
            if self.booking.user_id is not None:
                return "I do not have any booking records for your account yet."
            return "I do not have any booking records for this current session."
        parts = []
        for index, booking in enumerate(bookings, start=1):
            date_text = format_booking_date(booking.get("booking_date"), booking.get("day"))
            parts.append(
                f"{index}. {booking.get('reference')}: {booking.get('restaurant_name')} on {date_text} "
                f"at {booking.get('time')} for {booking.get('people')} people ({booking.get('status')})"
            )
        label = "Current account booking records" if self.booking.user_id is not None else "Current session booking records"
        return f"{label}: " + " ".join(parts)

    def _bulk_cancel_exception(
        self,
        user_message: str,
        named_restaurant: dict[str, Any] | None,
    ) -> tuple[bool, str | None]:
        text = user_message.lower()
        text = re.sub(r"\bresuratns\b|\bresurants\b|\bresturants\b|\bresutrants\b|\brestraunts\b", "restaurants", text)
        text = re.sub(r"\bresuratn\b|\bresurant\b|\bresturant\b|\brestraunt\b|\bresutrant\b", "restaurant", text)
        is_bulk = bool(
            re.search(r"\b(cancel|delete|remove)\b", text)
            and re.search(r"\b(all|every)\b", text)
            and re.search(r"\b(bookings?|reservations?)\b", text)
        )
        if not is_bulk:
            return False, None
        if named_restaurant and named_restaurant.get("name"):
            return True, str(named_restaurant["name"])

        keep_match = re.search(
            r"\b(?:apart\s+from|except|besides|other\s+than|but\s+not)\s+(.+?)(?:[?.!]|$)",
            text,
        )
        if not keep_match:
            return True, None

        keep_text = keep_match.group(1)
        keep_text = re.sub(r"\b(?:the|my|booking|bookings|reservation|reservations|restaurant|restaurants|please)\b", " ", keep_text)
        keep_text = " ".join(keep_text.split())
        return True, keep_text or None

    def _bulk_cancel_except_response(self, keep_restaurant: str | None) -> str:
        if self.booking.user_id is None:
            return "I can only bulk-cancel account booking records in the logged-in web app."
        if not keep_restaurant:
            return (
                "Which restaurant should I keep? "
                "For safety, I need an exception such as 'cancel all my bookings apart from Nandos'."
            )

        cancelled, kept = self.booking.cancel_bookings_except_restaurant(keep_restaurant)
        if not kept:
            return (
                f"I could not find a confirmed booking for {keep_restaurant}, so I did not cancel anything. "
                "Please give the restaurant name or booking reference you want to keep."
            )
        kept_refs = ", ".join(str(booking.get("reference")) for booking in kept if booking.get("reference"))
        if not cancelled:
            return f"I did not find any other confirmed bookings to cancel. Kept {self._kept_restaurant_label(kept, keep_restaurant)} unchanged."

        cancelled_refs = ", ".join(str(booking.get("reference")) for booking in cancelled if booking.get("reference"))
        kept_text = kept_refs or keep_restaurant
        return (
            f"I cancelled {len(cancelled)} booking record{'s' if len(cancelled) != 1 else ''}: {cancelled_refs}. "
            f"I kept {kept_text} for {self._kept_restaurant_label(kept, keep_restaurant)} unchanged."
        )

    def _kept_restaurant_label(self, kept: list[dict[str, Any]], fallback: str) -> str:
        names = []
        for booking in kept:
            name = str(booking.get("restaurant_name") or "").strip()
            if name and name not in names:
                names.append(name)
        return ", ".join(names) if names else fallback

    def _is_duplicate_booking_request(self, user_message: str, effective_intent: str) -> bool:
        if effective_intent != "book":
            return False
        text = user_message.lower()
        text = re.sub(r"\bresuratns\b|\bresurants\b|\bresturants\b|\bresutrants\b|\brestraunts\b", "restaurants", text)
        text = re.sub(r"\bresuratn\b|\bresurant\b|\bresturant\b|\brestraunt\b|\bresutrant\b", "restaurant", text)
        has_another_booking = re.search(
            r"\b(another|same|copy|duplicate|repeat|one more|new)\b.*\b(booking|reservation)\b"
            r"|\b(create|make|book)\b.*\b(another|same|one more|new)\b",
            text,
        )
        has_same_details = re.search(r"\bsame\s+(restaurant|people|time|booking|reservation|details)\b", text)
        return bool(has_another_booking and has_same_details)

    def _restaurant_detail_response(self, restaurant: dict[str, Any]) -> str:
        name = restaurant.get("name", "the selected restaurant")
        descriptors = []
        if restaurant.get("pricerange"):
            descriptors.append(str(restaurant["pricerange"]))
        if restaurant.get("food"):
            descriptors.append(str(restaurant["food"]))
        detail = str(name)
        if descriptors:
            detail += " (" + " ".join(descriptors) + ")"
        if restaurant.get("area"):
            detail += f" in the {restaurant['area']} area"
        fields = []
        if restaurant.get("address"):
            fields.append(f"Address: {restaurant['address']}")
        if restaurant.get("postcode"):
            fields.append(f"Postcode: {restaurant['postcode']}")
        if restaurant.get("phone"):
            fields.append(f"Phone: {restaurant['phone']}")
        if fields:
            detail += ". " + ". ".join(fields)
        return detail + "."

    def _distance_info_response(self, restaurant: dict[str, Any] | None, target_area: str | None) -> str:
        limitation = (
            "I do not have exact distance or travel-time data in the loaded MultiWOZ restaurant records; "
            "the data only gives broad Cambridge areas."
        )
        if not restaurant:
            return f"{limitation} Tell me a restaurant name and I can say which area it is recorded in."

        name = restaurant.get("name", "that restaurant")
        source_area = restaurant.get("area")
        if source_area and target_area and source_area != target_area:
            return (
                f"{limitation} {name} is recorded in the {source_area} area, while {target_area} is a different area. "
                f"If you want restaurants in the {target_area}, I can list those instead."
            )
        if source_area and target_area:
            return f"{limitation} {name} is recorded in the {source_area} area."
        if source_area:
            return f"{limitation} {name} is recorded in the {source_area} area."
        return f"{limitation} I do not have an area recorded for {name}."

    def _next_week_day_prompt(self) -> str:
        restaurant = self.state.booking_restaurant or self.state.selected_restaurant or {}
        name = restaurant.get("name")
        target = f" for {name}" if name else ""
        return (
            f"Which day next week would you like{target}? "
            "Please choose Monday, Tuesday, Wednesday, Thursday, Friday, Saturday or Sunday."
        )

    def _filter_info_response(self) -> str:
        return (
            "You can filter restaurants by area: centre, north, south, east and west. "
            "You can also filter by price range: cheap, moderate or expensive, and by cuisine such as Indian, Turkish, Italian or British."
        )

    def _cuisine_help_response(self) -> str:
        cuisines = sorted({str(record.get("food", "")).strip() for record in self.restaurants if record.get("food")})
        if not cuisines:
            return "I do not have cuisine categories loaded yet."
        display = ", ".join(cuisine.title() for cuisine in cuisines[:24])
        return (
            "You can search by cuisine categories in the loaded MultiWOZ restaurant data. "
            f"Available cuisines include: {display}."
        )

    def _unsupported_slot_response(self, unsupported_slots: dict[str, str]) -> str:
        if unsupported_slots.get("food"):
            food = unsupported_slots["food"]
            if food in {"egyptian", "yemeni"}:
                return (
                    f"I do not have {food.title()} as a cuisine category in the loaded MultiWOZ restaurant data. "
                    "The closest supported Middle Eastern-style category here is Lebanese; you can also search Turkish or Mediterranean."
                )
            return (
                f"I do not have dish-level menu data for '{food}'. "
                "The MultiWOZ restaurant records are organised by cuisine, such as Italian, Indian, Chinese, Turkish, Lebanese or Mediterranean."
            )
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
