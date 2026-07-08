"""Grounded response generation with a safe template fallback."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from restaurant_assistant.dialogue_state import DialogueState
from restaurant_assistant.llm_runtime import llm_backend_error
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
        self._load_error: str | None = None

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
        template_text = self._template_response(state, results, intent=intent, missing_slots=missing_slots)
        if missing_slots:
            return GenerationResult(text=template_text, used_llm=False, mode="template")
        if self.enable_llm:
            llm_result = self._try_llm(user_message, state, results, intent=intent, missing_slots=missing_slots)
            if llm_result is not None:
                return llm_result
        return GenerationResult(text=template_text, used_llm=False, mode="template")

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
            "or verified dietary claims. If booking, describe only the session booking record. Keep the reply concise.\n"
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
        output = self._clean_llm_output(output)
        if not output or self._looks_like_prompt_leak(output):
            return None
        return GenerationResult(text=output, used_llm=True, mode=f"transformers:{self.model_name}")

    def _clean_llm_output(self, output: str) -> str:
        text = output.strip()
        if "Assistant:" in text:
            text = text.rsplit("Assistant:", 1)[-1].strip()
        return text

    def _looks_like_prompt_leak(self, output: str) -> bool:
        text = output.strip()
        lowered = text.lower()
        if text.startswith(("{", "[")):
            return True
        leak_patterns = [
            r'"\s*user\s*"\s*:',
            r'"\s*assistant\s*"\s*:',
            r'"\s*timestamp\s*"\s*:',
            r'"\s*conversation_history\s*"\s*:',
            r"\bstate\s*:",
            r"\bevidence\s*:",
            r"\bmissing slots\s*:",
            r"\bintent\s*:",
        ]
        return any(re.search(pattern, lowered) for pattern in leak_patterns)

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        backend_error = llm_backend_error()
        if backend_error:
            self._load_error = backend_error
            self._pipeline = False
            return self._pipeline
        try:
            from transformers import pipeline

            self._pipeline = pipeline("text2text-generation", model=self.model_name)
        except Exception as exc:
            self._load_error = str(exc)
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
            return "Hello. I can help find MultiWOZ restaurant records and create booking records."
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
            return f"{detail}To complete the booking for {restaurant_name}, please provide: {missing_text}."
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
