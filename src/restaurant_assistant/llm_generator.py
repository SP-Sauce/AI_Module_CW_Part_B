"""Optional guarded response generation with deterministic fallback."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from restaurant_assistant.dialogue_state import DialogueState
from restaurant_assistant.llm_runtime import llm_backend_error
from restaurant_assistant.nlg import contains_json_or_debug_leakage
from restaurant_assistant.ranking import RankedRestaurant


DEFAULT_RESPONSE_MODEL = "google/flan-t5-base"

RAW_PROMPT_LABELS = (
    "Evidence",
    "State",
    "Intent",
    "Missing slots",
    "Task",
    "User",
    "Assistant",
    "Response",
)

UNSUPPORTED_CLAIM_PATTERNS = (
    r"\blive\s+availability\b",
    r"\bavailable\s+(?:now|today|tonight|tomorrow)\b",
    r"\btable\s+(?:is|will be|has been)\s+available\b",
    r"\bavailability\s+(?:is|has been)\s+confirmed\b",
    r"\bpayment\b",
    r"\bpayments\b",
    r"\bcredit\s+card\b",
    r"\bdebit\s+card\b",
    r"\bcard\s+(?:details|payment|handling|accepted|accepted)\b",
    r"\bpay\s+(?:now|online|by card|with card)\b",
    r"\breviews?\b",
    r"\bratings?\b",
    r"\b\d(?:\.\d)?\s*(?:star|stars)\b",
    r"\bhalal\b",
    r"\ballerg(?:y|ies|ens|ic)\b",
    r"\bgluten[- ]free\b.*\b(?:safe|verified|certified)\b",
    r"\bdietary\b.*\b(?:safe|verified|certified|guaranteed)\b",
)

UK_POSTCODE_PATTERN = re.compile(
    r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b",
    flags=re.IGNORECASE,
)
PHONE_PATTERN = re.compile(r"(?:\+?\d[\d\s().-]{6,}\d)")
ADDRESS_PATTERN = re.compile(
    r"\b(?:[A-Z]?\d{1,5}\s+)?[A-Za-z0-9' -]{2,80}?\s+"
    r"(?:street|st|road|rd|avenue|ave|lane|ln|drive|dr|way|parade|square|place|pl|park|terrace|court|common)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ResponseValidationResult:
    ok: bool
    reason: str | None = None


@dataclass(frozen=True)
class GenerationResult:
    text: str
    used_llm: bool
    mode: str
    attempted: bool = False
    rejected_reason: str | None = None
    final_response_mode: str = "baseline_template"
    response_generation_mode: str | None = None
    latency_seconds: float = 0.0


def validate_generated_response(
    text: str,
    *,
    evidence_records: Iterable[dict[str, Any]] | None = None,
    known_restaurant_records: Iterable[dict[str, Any]] | None = None,
) -> ResponseValidationResult:
    """Validate customer-facing generated text against evidence records."""

    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return ResponseValidationResult(False, "empty_response")
    if len(cleaned) > 700 or len(cleaned.split()) > 120:
        return ResponseValidationResult(False, "response_too_long")
    if contains_json_or_debug_leakage(cleaned):
        return ResponseValidationResult(False, "json_or_debug_leakage")
    lowered = cleaned.casefold()
    for label in RAW_PROMPT_LABELS:
        if re.search(rf"\b{re.escape(label.casefold())}\s*:", lowered):
            return ResponseValidationResult(False, "prompt_label_leakage")
    for pattern in UNSUPPORTED_CLAIM_PATTERNS:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            return ResponseValidationResult(False, "unsupported_claim")

    evidence = [record for record in evidence_records or [] if record]
    known = [record for record in known_restaurant_records or [] if record]
    if not known:
        known = evidence

    evidence_names = _normalised_values(record.get("name") for record in evidence)
    known_names = _normalised_values(record.get("name") for record in known)
    mentioned_names = {name for name in known_names if name and name in lowered}
    if mentioned_names and not mentioned_names <= evidence_names:
        return ResponseValidationResult(False, "invented_restaurant_name")

    evidence_phones = {_digits(record.get("phone")) for record in evidence if _digits(record.get("phone"))}
    for match in PHONE_PATTERN.findall(cleaned):
        digits = _digits(match)
        if len(digits) >= 6 and digits not in evidence_phones:
            return ResponseValidationResult(False, "invented_phone")

    evidence_postcodes = {_postcode(record.get("postcode")) for record in evidence if _postcode(record.get("postcode"))}
    for match in UK_POSTCODE_PATTERN.findall(cleaned):
        if _postcode(match) not in evidence_postcodes:
            return ResponseValidationResult(False, "invented_postcode")

    evidence_addresses = _normalised_values(record.get("address") for record in evidence)
    known_addresses = _normalised_values(record.get("address") for record in known)
    mentioned_known_addresses = {address for address in known_addresses if address and address in lowered}
    if mentioned_known_addresses and not mentioned_known_addresses <= evidence_addresses:
        return ResponseValidationResult(False, "invented_address")
    for match in ADDRESS_PATTERN.findall(cleaned):
        address = _normalise_text(match)
        if address and not any(address in allowed or allowed in address for allowed in evidence_addresses):
            return ResponseValidationResult(False, "invented_address")

    return ResponseValidationResult(True)


def _normalise_text(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _normalised_values(values: Iterable[Any]) -> set[str]:
    return {normalised for value in values if (normalised := _normalise_text(value))}


def _digits(value: Any) -> str:
    return re.sub(r"\D", "", str(value or ""))


def _postcode(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "").casefold())


class GroundedResponseGenerator:
    """Generate concise support responses grounded in retrieved records."""

    def __init__(
        self,
        *,
        enable_llm: bool = False,
        model_name: str = DEFAULT_RESPONSE_MODEL,
        pretrained_fallback: bool = True,
        pretrained_fallback_model: str = DEFAULT_RESPONSE_MODEL,
        known_restaurants: Iterable[dict[str, Any]] | None = None,
    ) -> None:
        self.enable_llm = enable_llm
        self.model_name = model_name
        self.pretrained_fallback = pretrained_fallback
        self.pretrained_fallback_model = pretrained_fallback_model
        self.known_restaurants = list(known_restaurants or [])
        self._pipeline: Any = None
        self._pipelines: dict[str, Any] = {}
        self._load_error: str | None = None
        self.last_result: GenerationResult | None = None

    def generate(
        self,
        user_message: str,
        state: DialogueState,
        ranked_results: Iterable[RankedRestaurant] | None = None,
        *,
        intent: str = "search",
        missing_slots: list[str] | None = None,
        baseline_text: str | None = None,
    ) -> GenerationResult:
        results = list(ranked_results or [])
        template_text = baseline_text or self._template_response(
            state,
            results,
            intent=intent,
            missing_slots=missing_slots,
        )
        start = time.perf_counter()
        evidence = self._evidence_records(state, results)
        baseline_validation = validate_generated_response(
            template_text,
            evidence_records=evidence,
            known_restaurant_records=self.known_restaurants,
        )
        if not baseline_validation.ok:
            template_text = self._safe_minimal_fallback(intent)

        if not self.enable_llm:
            result = GenerationResult(
                text=template_text,
                used_llm=False,
                mode="template",
                attempted=False,
                final_response_mode="baseline_template",
                response_generation_mode="disabled",
                latency_seconds=round(time.perf_counter() - start, 6),
            )
            self.last_result = result
            return result

        attempts = self._attempt_chain()
        rejected_reason = None
        last_mode = None
        for model_name, mode in attempts:
            last_mode = mode
            generated_text, error = self._try_llm(
                model_name,
                mode,
                user_message,
                state,
                evidence,
                intent=intent,
                missing_slots=missing_slots,
                baseline_text=template_text,
            )
            if error:
                rejected_reason = error
                continue
            validation = validate_generated_response(
                generated_text,
                evidence_records=evidence,
                known_restaurant_records=self.known_restaurants,
            )
            if validation.ok:
                result = GenerationResult(
                    text=generated_text,
                    used_llm=True,
                    mode=f"transformers:{model_name}",
                    attempted=True,
                    final_response_mode=mode,
                    response_generation_mode=mode,
                    latency_seconds=round(time.perf_counter() - start, 6),
                )
                self.last_result = result
                return result
            rejected_reason = validation.reason

        result = GenerationResult(
            text=template_text,
            used_llm=False,
            mode="template",
            attempted=True,
            rejected_reason=rejected_reason,
            final_response_mode="final_guarded_response",
            response_generation_mode=last_mode or "not_attempted",
            latency_seconds=round(time.perf_counter() - start, 6),
        )
        self.last_result = result
        return result

    def _attempt_chain(self) -> list[tuple[str, str]]:
        if self._is_adapter_path(self.model_name):
            attempts = [(self.model_name, "trained_lora_response")]
            if self.pretrained_fallback:
                attempts.append((self.pretrained_fallback_model, "pretrained_flan_t5_base_response"))
            return attempts
        mode = (
            "pretrained_flan_t5_base_response"
            if self.model_name == DEFAULT_RESPONSE_MODEL
            else "pretrained_response"
        )
        return [(self.model_name, mode)]

    def _try_llm(
        self,
        model_name: str,
        mode: str,
        user_message: str,
        state: DialogueState,
        evidence: list[dict[str, Any]],
        *,
        intent: str,
        missing_slots: list[str] | None,
        baseline_text: str,
    ) -> tuple[str, str | None]:
        pipe = self._load_pipeline(model_name, mode)
        if not pipe:
            return "", self._load_error or "response_model_unavailable"
        prompt = self._build_prompt(
            user_message,
            state,
            evidence,
            intent=intent,
            missing_slots=missing_slots,
            baseline_text=baseline_text,
        )
        try:
            output = pipe(prompt, max_new_tokens=120, do_sample=False)[0]["generated_text"].strip()
        except Exception as exc:
            return "", f"generation_error:{exc}"
        output = self._clean_llm_output(output)
        if not output:
            return "", "empty_response"
        return output, None

    def _build_prompt(
        self,
        user_message: str,
        state: DialogueState,
        evidence: list[dict[str, Any]],
        *,
        intent: str,
        missing_slots: list[str] | None,
        baseline_text: str,
    ) -> str:
        safe_state = {
            key: value
            for key, value in state.to_dict().items()
            if key in {"food", "area", "pricerange", "day", "time", "people", "booking_status"}
            and value not in (None, "")
        }
        return (
            "Task: Generate a grounded restaurant assistant response.\n"
            "Rules: Use only the evidence. Do not mention JSON, state, slots, scores, source ids, "
            "payments, reviews, live availability, halal status, allergies, or unsupported facts.\n"
            f"Intent: {intent}\n"
            f"User: {user_message}\n"
            f"State: {json.dumps(safe_state, sort_keys=True)}\n"
            f"Evidence: {json.dumps(evidence, sort_keys=True)}\n"
            f"Missing slots: {missing_slots or []}\n"
            f"Baseline response: {baseline_text}\n"
            "Response:"
        )

    def _clean_llm_output(self, output: str) -> str:
        text = output.strip()
        for marker in ("Response:", "Assistant:"):
            if marker in text:
                text = text.rsplit(marker, 1)[-1].strip()
        return text

    def _is_adapter_path(self, model_name: str) -> bool:
        path = Path(model_name)
        return path.exists() and (path / "adapter_config.json").exists()

    def _load_pipeline(self, model_name: str, mode: str) -> Any:
        cache_key = f"{mode}:{model_name}"
        if cache_key in self._pipelines:
            return self._pipelines[cache_key]
        if self._pipeline is not None and model_name == self.model_name:
            self._pipelines[cache_key] = self._pipeline
            return self._pipeline
        backend_error = llm_backend_error()
        if backend_error:
            self._load_error = backend_error
            self._pipelines[cache_key] = False
            return False
        try:
            if mode == "trained_lora_response":
                pipe = self._load_lora_pipeline(model_name)
            else:
                from transformers import pipeline

                pipe = pipeline("text2text-generation", model=model_name)
            self._pipelines[cache_key] = pipe
            if model_name == self.model_name:
                self._pipeline = pipe
            self._load_error = None
        except Exception as exc:
            self._load_error = str(exc)
            self._pipelines[cache_key] = False
        return self._pipelines[cache_key]

    def _load_lora_pipeline(self, adapter_path: str) -> Any:
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

        peft_config = PeftConfig.from_pretrained(adapter_path)
        base_model_name = peft_config.base_model_name_or_path or self.pretrained_fallback_model
        adapter = Path(adapter_path)
        tokenizer_source = adapter_path if (adapter / "tokenizer_config.json").exists() else base_model_name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(base_model, adapter_path)
        return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    def _template_response(
        self,
        state: DialogueState,
        ranked_results: list[RankedRestaurant],
        *,
        intent: str,
        missing_slots: list[str] | None,
    ) -> str:
        if intent == "greeting":
            return "Hi - I can help find MultiWOZ restaurant records and create booking records."
        if missing_slots:
            return self._clarification(state, missing_slots)
        if not ranked_results:
            return (
                "I could not find a matching restaurant record in the loaded data. "
                "Try a food type, area or price range, such as cheap Italian in the centre."
            )
        top = ranked_results[0]
        record = top.record
        request_text = self._constraint_phrase(state, record)
        if top.missing_unmatched_constraints:
            response = (
                f"I could not find an exact match, but the closest option I have is "
                f"{self._restaurant_summary(record)}."
            )
            response += " It may not match every preference, so you may want to adjust the search."
        elif request_text:
            response = (
                f"I found {self._restaurant_summary(record)}, which matches your request for "
                f"{request_text}."
            )
        else:
            response = f"I found {self._restaurant_summary(record)}."
        contact = self._contact_details(record)
        if contact:
            response += " " + contact
        return response

    def _clarification(self, state: DialogueState, missing_slots: list[str]) -> str:
        if "restaurant" in missing_slots:
            return (
                "Sure - please choose a restaurant first, or tell me the food type, area and price range "
                "to search."
            )
        readable = {
            "food": "food type",
            "area": "area",
            "pricerange": "price range",
            "day": "day",
            "time": "time",
            "people": "number of people",
        }
        missing_text = self._join_readable(readable.get(slot, slot) for slot in missing_slots)
        restaurant_name = state.selected_restaurant.get("name") if state.selected_restaurant else "the restaurant"
        if any(slot in {"day", "time", "people"} for slot in missing_slots):
            return (
                f"Great, I can create a booking record for {restaurant_name}. "
                f"To complete the booking for {restaurant_name}, I still need the {missing_text}."
            )
        if missing_slots == ["food"]:
            return "Sure - what kind of food would you like me to search for?"
        return f"Please tell me your preferred {missing_text}."

    def _restaurant_summary(self, record: dict[str, Any]) -> str:
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
        return detail

    def _contact_details(self, record: dict[str, Any]) -> str:
        fields = []
        if record.get("address"):
            fields.append(f"Address: {record['address']}")
        if record.get("postcode"):
            fields.append(f"Postcode: {record['postcode']}")
        if record.get("phone"):
            fields.append(f"Phone: {record['phone']}")
        return ". ".join(fields) + "." if fields else ""

    def _constraint_phrase(self, state: DialogueState, record: dict[str, Any]) -> str:
        food = state.food or record.get("food")
        area = state.area or record.get("area")
        price = state.pricerange or record.get("pricerange")
        parts = []
        if price:
            parts.append(str(price))
        if food:
            parts.append(str(food))
        phrase = " ".join(parts)
        if phrase:
            phrase += " restaurant"
        elif area:
            phrase = "a restaurant"
        if area:
            phrase += f" in the {area} area"
        return phrase

    def _evidence_records(
        self,
        state: DialogueState,
        ranked_results: list[RankedRestaurant],
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for result in ranked_results:
            self._append_public_record(records, seen, result.record)
        self._append_public_record(records, seen, state.selected_restaurant)
        self._append_public_record(records, seen, state.booking_restaurant)
        return records

    def _append_public_record(
        self,
        records: list[dict[str, Any]],
        seen: set[str],
        record: dict[str, Any] | None,
    ) -> None:
        public = self._public_record(record or {})
        if not public:
            return
        key = str(public.get("name") or public.get("address") or public)
        if key in seen:
            return
        seen.add(key)
        records.append(public)

    def _public_record(self, record: dict[str, Any]) -> dict[str, Any]:
        allowed = ["name", "food", "area", "pricerange", "address", "postcode", "phone", "type"]
        return {key: record.get(key) for key in allowed if record.get(key)}

    def _safe_minimal_fallback(self, intent: str) -> str:
        if intent == "book":
            return "I can help create a booking record once the restaurant, day, time and number of people are clear."
        return "I can help with restaurant search and booking records from the loaded MultiWOZ data."

    def _join_readable(self, values: Iterable[str]) -> str:
        items = [value for value in values if value]
        if not items:
            return "details"
        if len(items) == 1:
            return items[0]
        return ", ".join(items[:-1]) + " and " + items[-1]
