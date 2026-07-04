"""Grounded response generation with a safe template fallback."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

from restaurant_assistant.dialogue_state import DialogueState
from restaurant_assistant.ranking import RankedRestaurant


@dataclass(frozen=True)
class GenerationResult:
    text: str
    used_llm: bool
    mode: str


class GroundedResponseGenerator:
    """Generate concise support responses grounded in retrieved records."""

    def __init__(self, *, enable_llm: bool = False, model_name: str = "google/flan-t5-small") -> None:
        self.enable_llm = enable_llm
        self.model_name = model_name
        self._pipeline = None

    def generate(
        self,
        user_message: str,
        state: DialogueState,
        ranked_results: Iterable[RankedRestaurant] | None = None,
        *,
        intent: str = "search",
        missing_slots: list[str] | None = None,
    ) -> GenerationResult:
        results = list(ranked_results or [])
        if self.enable_llm:
            llm_result = self._try_llm(user_message, state, results, intent=intent, missing_slots=missing_slots)
            if llm_result is not None:
                return llm_result
        return GenerationResult(
            text=self._template_response(state, results, intent=intent, missing_slots=missing_slots),
            used_llm=False,
            mode="template",
        )

    def _try_llm(
        self,
        user_message: str,
        state: DialogueState,
        ranked_results: list[RankedRestaurant],
        *,
        intent: str,
        missing_slots: list[str] | None,
    ) -> GenerationResult | None:
        pipe = self._load_pipeline()
        if not pipe:
            return None
        evidence = [self._public_record(result.record) for result in ranked_results[:3]]
        prompt = (
            "You are a restaurant support assistant. Use only the JSON restaurant evidence. "
            "Do not invent phone numbers, addresses, postcodes, food types, payments, live availability, "
            "or verified dietary claims. If booking, say it is simulated. Keep the reply concise.\n"
            f"Intent: {intent}\n"
            f"Missing slots: {missing_slots or []}\n"
            f"State: {json.dumps(state.to_dict(), default=str)}\n"
            f"Evidence: {json.dumps(evidence)}\n"
            f"User: {user_message}\n"
            "Assistant:"
        )
        try:
            output = pipe(prompt, max_new_tokens=120, do_sample=False)[0]["generated_text"].strip()
        except Exception:
            return None
        if not output:
            return None
        return GenerationResult(text=output, used_llm=True, mode=f"transformers:{self.model_name}")

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline

            self._pipeline = pipeline("text2text-generation", model=self.model_name)
        except Exception:
            self._pipeline = False
        return self._pipeline

    def _template_response(
        self,
        state: DialogueState,
        ranked_results: list[RankedRestaurant],
        *,
        intent: str,
        missing_slots: list[str] | None,
    ) -> str:
        if intent == "greeting":
            return "Hello. I can help find MultiWOZ restaurant records and create simulated bookings."
        if missing_slots:
            return self._clarification(state, missing_slots)
        if not ranked_results:
            return (
                "I could not find a matching restaurant record from the loaded data. "
                "Try a food type, area or price range such as cheap Italian in the centre."
            )
        top = ranked_results[0]
        record = top.record
        prefix = "I found"
        if top.missing_unmatched_constraints:
            prefix = "I could not find an exact match, but the closest retrieved option is"
        response = f"{prefix} {self._format_restaurant(record)}."
        if top.missing_unmatched_constraints:
            response += " It may not match every preference, so you may want to adjust the search."
        return response

    def _clarification(self, state: DialogueState, missing_slots: list[str]) -> str:
        if "restaurant" in missing_slots:
            return "Please choose a restaurant first, or tell me the food type, area and price range to search."
        readable = {
            "food": "food type",
            "area": "area",
            "pricerange": "price range",
            "day": "day",
            "time": "time",
            "people": "number of people",
        }
        missing_text = ", ".join(readable.get(slot, slot) for slot in missing_slots)
        restaurant_name = state.selected_restaurant.get("name") if state.selected_restaurant else "the restaurant"
        if any(slot in {"day", "time", "people"} for slot in missing_slots):
            detail = ""
            if state.selected_restaurant:
                detail = self._format_restaurant(state.selected_restaurant) + ". "
            return f"{detail}To create a simulated booking for {restaurant_name}, please provide: {missing_text}."
        return f"Please tell me your preferred {missing_text}."

    def _format_restaurant(self, record: dict[str, Any]) -> str:
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

    def _public_record(self, record: dict[str, Any]) -> dict[str, Any]:
        allowed = ["name", "food", "area", "pricerange", "address", "postcode", "phone", "type"]
        return {key: record.get(key) for key in allowed if record.get(key)}
