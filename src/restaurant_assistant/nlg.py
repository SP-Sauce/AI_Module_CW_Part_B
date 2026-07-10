"""Deterministic natural-language generation from response plans."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable

from restaurant_assistant.date_utils import format_booking_date
from restaurant_assistant.response_plan import ResponsePlan


FORBIDDEN_RESPONSE_TERMS = (
    "source_id",
    "name_norm",
    "food_norm",
    "area_norm",
    "pricerange_norm",
    "similarity",
    "score",
    "debug",
    "slot_model_name",
    "generation_mode",
)


def contains_json_or_debug_leakage(text: str) -> bool:
    """Return True when text looks like raw JSON/debug/database output."""

    if not text:
        return False
    lowered = text.casefold()
    if "{" in text or "}" in text:
        return True
    if any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in FORBIDDEN_RESPONSE_TERMS):
        return True
    json_key_pattern = r"""["'](?:name|food|area|pricerange|address|postcode|phone|type|intent|slots)["']\s*:"""
    return re.search(json_key_pattern, text, flags=re.IGNORECASE) is not None


def safe_user_text(text: str, fallback: str | None = None) -> str:
    """Return text only if it is safe for the customer-facing channel."""

    fallback_text = fallback or (
        "Sorry, I had trouble formatting that response. Please try the restaurant request again."
    )
    if contains_json_or_debug_leakage(text):
        return fallback_text
    return text


class NaturalLanguageGenerator:
    """Convert structured ResponsePlan objects into customer-facing text."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = defaultdict(int)
        self._phrases = {
            "greeting": [
                "Hi - I can help find MultiWOZ restaurant records and create booking records.",
                "Hello - I can help with restaurant searches and booking records from the loaded MultiWOZ data.",
            ],
            "thanks": [
                "You're welcome. I can help with the current session booking if you need to view, change or cancel it.",
                "You're welcome. I can also help view, change or cancel the current booking record.",
            ],
            "goodbye": [
                "Goodbye. I can help again whenever you need another restaurant search or booking record.",
                "Goodbye. Come back any time you want to search restaurants or manage a booking record.",
            ],
            "fallback_error": [
                "Sorry, I had trouble formatting that response. Please try the restaurant request again.",
                "Sorry, I could not turn that into a clean reply. Please ask for the restaurant search again.",
            ],
        }

    def generate(self, plan: ResponsePlan) -> str:
        """Render a plan into a safe assistant message."""

        renderer = getattr(self, f"_render_{plan.dialogue_act}", None)
        if renderer is None:
            text = self._render_direct_message(plan)
        else:
            text = renderer(plan)
        return safe_user_text(text, fallback=self._render_fallback_error(plan))

    def _phrase(self, key: str) -> str:
        options = self._phrases[key]
        index = self._counters[key] % len(options)
        self._counters[key] += 1
        return options[index]

    def _render_greeting(self, plan: ResponsePlan) -> str:
        return self._phrase("greeting")

    def _render_thanks(self, plan: ResponsePlan) -> str:
        return self._phrase("thanks")

    def _render_goodbye(self, plan: ResponsePlan) -> str:
        return self._phrase("goodbye")

    def _render_no_results(self, plan: ResponsePlan) -> str:
        message = plan.metadata.get("empty_message")
        if message:
            return str(message)
        constraint_text = self._constraint_text(plan.constraints)
        if constraint_text:
            return f"I could not find restaurants matching {constraint_text} in the loaded MultiWOZ records."
        return "I could not find restaurants matching those filters in the loaded MultiWOZ records."

    def _render_exact_match_list(self, plan: ResponsePlan) -> str:
        prefix = str(plan.metadata.get("prefix") or "Matching restaurants:")
        restaurants = plan.alternatives or plan.retrieved_restaurants
        if not restaurants:
            return self._render_no_results(plan)
        return prefix + " " + " ".join(
            f"{index}. {self._restaurant_summary(record)}"
            for index, record in enumerate(restaurants, start=1)
        )

    def _render_partial_match(self, plan: ResponsePlan) -> str:
        restaurants = plan.alternatives or plan.retrieved_restaurants
        if not restaurants:
            return self._render_no_results(plan)
        lead = str(
            plan.metadata.get("lead")
            or "I could not find an exact match, but these are the closest options I can suggest:"
        )
        return lead + " " + " ".join(
            f"{index}. {self._restaurant_summary(record)}"
            for index, record in enumerate(restaurants, start=1)
        )

    def _render_single_recommendation(self, plan: ResponsePlan) -> str:
        restaurant = plan.selected_restaurant or self._first(plan.retrieved_restaurants)
        if not restaurant:
            return self._render_no_results(plan)
        constraint_text = self._constraint_text(plan.constraints)
        if constraint_text:
            contact = self._restaurant_contact_sentence(restaurant)
            suffix = f" {contact}" if contact else ""
            return (
                f"I found {self._restaurant_summary(restaurant)}, which matches your request for "
                f"{constraint_text}.{suffix}"
            )
        return f"I found {self._restaurant_detail(restaurant)}."

    def _render_restaurant_details(self, plan: ResponsePlan) -> str:
        restaurant = plan.selected_restaurant
        if not restaurant:
            return "Which restaurant would you like the address or contact details for?"
        return self._restaurant_detail(restaurant) + "."

    def _render_booking_missing_details(self, plan: ResponsePlan) -> str:
        missing = plan.missing_constraints
        if "restaurant" in missing:
            return "Sure - please choose a restaurant first, or tell me the food type, area and price range to search."
        readable = {
            "food": "food type",
            "area": "area",
            "pricerange": "price range",
            "day": "day",
            "time": "time",
            "people": "number of people",
        }
        missing_text = ", ".join(readable.get(slot, slot) for slot in missing)
        restaurant = plan.selected_restaurant
        restaurant_name = restaurant.get("name") if restaurant else "the restaurant"
        if any(slot in {"day", "time", "people"} for slot in missing):
            return (
                f"Great, I can create a booking record for {restaurant_name}. "
                f"To complete the booking for {restaurant_name}, I still need the {missing_text}."
            )
        if missing == ["food"]:
            return "Sure - what kind of food would you like me to search for?"
        return f"Please tell me your preferred {missing_text}."

    def _render_booking_confirmation(self, plan: ResponsePlan) -> str:
        restaurant = plan.selected_restaurant or {}
        name = restaurant.get("name", "the selected restaurant")
        reference = plan.metadata.get("reference")
        date_text = self._plan_date_text(plan)
        time = plan.metadata.get("time")
        people = plan.metadata.get("people")
        if reference:
            return (
                f"Great, I have created a booking record for {name} on {date_text} at {time} "
                f"for {people} people. Your reference is {reference}."
            )
        return f"Great, I have created a booking record for {name}."

    def _render_booking_rescheduled(self, plan: ResponsePlan) -> str:
        restaurant = plan.selected_restaurant or {}
        name = restaurant.get("name", "the selected restaurant")
        reference = plan.metadata.get("reference")
        date_text = self._plan_date_text(plan)
        time = plan.metadata.get("time")
        people = plan.metadata.get("people")
        return (
            f"I have updated booking {reference} for {name} "
            f"to {date_text} at {time} for {people} people."
        )

    def _render_booking_cancellation(self, plan: ResponsePlan) -> str:
        message = plan.metadata.get("message")
        if message:
            return str(message)
        reference = plan.metadata.get("reference")
        if reference:
            return f"Done - I have cancelled booking {reference}."
        return "There is no active booking to cancel."

    def _render_booking_list(self, plan: ResponsePlan) -> str:
        bookings = list(plan.metadata.get("bookings") or [])
        label = str(plan.metadata.get("label") or "Current session booking records")
        if not bookings:
            return str(plan.metadata.get("empty_message") or "I do not have any booking records for this current session.")
        if plan.metadata.get("table"):
            rows = [
                "| Reference | Restaurant | Date | Time | People | Status |",
                "| --- | --- | --- | --- | ---: | --- |",
            ]
            for booking in bookings:
                rows.append(
                    "| "
                    + " | ".join(
                        [
                            str(booking.get("reference") or ""),
                            str(booking.get("restaurant_name") or "the selected restaurant"),
                            format_booking_date(booking.get("booking_date"), booking.get("day")),
                            str(booking.get("time") or ""),
                            str(booking.get("people") or ""),
                            str(booking.get("status") or ""),
                        ]
                    )
                    + " |"
                )
            return f"{label}:\n" + "\n".join(rows)
        parts = []
        for index, booking in enumerate(bookings, start=1):
            date_text = format_booking_date(booking.get("booking_date"), booking.get("day"))
            parts.append(
                f"{index}. {booking.get('reference')}: {booking.get('restaurant_name')} on {date_text} "
                f"at {booking.get('time')} for {booking.get('people')} people ({booking.get('status')})"
            )
        return f"{label}: " + " ".join(parts)

    def _render_cuisine_help(self, plan: ResponsePlan) -> str:
        cuisines = list(plan.metadata.get("cuisines") or [])
        if not cuisines:
            return "I do not have cuisine categories loaded yet."
        display = ", ".join(str(cuisine).title() for cuisine in cuisines[:24])
        return (
            "You can search by cuisine categories in the loaded MultiWOZ restaurant data. "
            f"Available cuisines include: {display}."
        )

    def _render_area_filter_help(self, plan: ResponsePlan) -> str:
        return (
            "You can filter restaurants by area: centre, north, south, east and west. "
            "You can also filter by price range: cheap, moderate or expensive, and by cuisine such as Indian, Turkish, Italian or British."
        )

    def _render_table_view(self, plan: ResponsePlan) -> str:
        restaurants = plan.alternatives or plan.retrieved_restaurants
        if not restaurants:
            return self._render_no_results(plan)
        rows = [
            "| Name | Price | Cuisine | Area |",
            "| --- | --- | --- | --- |",
        ]
        for record in restaurants:
            rows.append(
                "| "
                + " | ".join(
                    [
                        str(record.get("name") or "unknown restaurant"),
                        str(record.get("pricerange") or "unknown price"),
                        str(record.get("food") or "unknown food"),
                        str(record.get("area") or "unknown area"),
                    ]
                )
                + " |"
            )
        return "Matching restaurants:\n" + "\n".join(rows)

    def _render_unsupported_request(self, plan: ResponsePlan) -> str:
        return str(
            plan.metadata.get("message")
            or "I can only help with MultiWOZ restaurant search and restaurant booking records. Try asking for a food type, area or price range."
        )

    def _render_fallback_error(self, plan: ResponsePlan) -> str:
        return self._phrase("fallback_error")

    def _render_cuisine_group(self, plan: ResponsePlan) -> str:
        group = plan.metadata.get("group") or "That cuisine group"
        foods = list(plan.metadata.get("foods") or [])
        cuisine_text = ", ".join(str(food).title() for food in foods)
        lead = (
            f"{group} is not a direct cuisine label in the loaded MultiWOZ records, "
            f"so I searched these supported categories: {cuisine_text}."
        )
        list_plan = ResponsePlan(
            dialogue_act=plan.metadata.get("list_dialogue_act", "exact_match_list"),
            user_intent=plan.user_intent,
            constraints=plan.constraints,
            retrieved_restaurants=plan.retrieved_restaurants,
            selected_restaurant=plan.selected_restaurant,
            alternatives=plan.alternatives,
            metadata={
                "prefix": plan.metadata.get("prefix", "Matching restaurants:"),
                "empty_message": plan.metadata.get("empty_message"),
            },
        )
        return f"{lead}\n{self.generate(list_plan)}"

    def _render_dish_preference(self, plan: ResponsePlan) -> str:
        dish = plan.metadata.get("dish") or "that dish"
        foods = list(plan.metadata.get("foods") or [])
        cuisine_text = ", ".join(str(food).title() for food in foods)
        lead = (
            f"I do not have dish-level menu data for {dish}, but it can fit these cuisine categories: "
            f"{cuisine_text}."
        )
        list_plan = ResponsePlan(
            dialogue_act=plan.metadata.get("list_dialogue_act", "exact_match_list"),
            user_intent=plan.user_intent,
            constraints=plan.constraints,
            retrieved_restaurants=plan.retrieved_restaurants,
            selected_restaurant=plan.selected_restaurant,
            alternatives=plan.alternatives,
            metadata={
                "prefix": plan.metadata.get("prefix", "Matching restaurants:"),
                "empty_message": plan.metadata.get("empty_message"),
            },
        )
        return f"{lead}\n{self.generate(list_plan)}"

    def _render_direct_message(self, plan: ResponsePlan) -> str:
        message = plan.metadata.get("message")
        if message:
            return str(message)
        return self._render_fallback_error(plan)

    def _restaurant_summary(self, record: dict[str, Any]) -> str:
        name = record.get("name", "unknown restaurant")
        food = record.get("food", "unknown food")
        area = record.get("area", "unknown area")
        price = record.get("pricerange", "unknown price")
        return f"{name} ({price} {food}, {area})"

    def _restaurant_detail(self, record: dict[str, Any]) -> str:
        name = record.get("name") or "an unnamed restaurant"
        descriptors = []
        if record.get("pricerange"):
            descriptors.append(str(record["pricerange"]))
        if record.get("food"):
            descriptors.append(str(record["food"]))
        detail = f"{name}"
        if descriptors:
            detail += " (" + " ".join(descriptors) + ")"
        if record.get("area"):
            detail += f" in the {record['area']} area"
        fields = []
        if record.get("address"):
            fields.append(f"Address: {record['address']}")
        if record.get("postcode"):
            fields.append(f"Postcode: {record['postcode']}")
        if record.get("phone"):
            fields.append(f"Phone: {record['phone']}")
        if fields:
            detail += ". " + ". ".join(fields)
        return detail

    def _restaurant_contact_sentence(self, record: dict[str, Any]) -> str:
        fields = []
        if record.get("address"):
            fields.append(f"Address: {record['address']}")
        if record.get("postcode"):
            fields.append(f"Postcode: {record['postcode']}")
        if record.get("phone"):
            fields.append(f"Phone: {record['phone']}")
        return ". ".join(fields) + "." if fields else ""

    def _constraint_text(self, constraints: dict[str, Any]) -> str:
        parts = []
        if constraints.get("pricerange"):
            parts.append(str(constraints["pricerange"]))
        if constraints.get("food"):
            parts.append(str(constraints["food"]))
        phrase = " ".join(parts)
        if phrase:
            phrase += " restaurant"
        elif constraints.get("area"):
            phrase = "a restaurant"
        if constraints.get("area"):
            phrase += f" in the {constraints['area']} area"
        return phrase

    def _plan_date_text(self, plan: ResponsePlan) -> str:
        return format_booking_date(plan.metadata.get("booking_date"), plan.metadata.get("day"))

    def _first(self, records: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
        for record in records:
            return record
        return None
