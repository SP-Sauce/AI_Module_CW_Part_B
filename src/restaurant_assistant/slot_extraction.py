"""Intent and slot extraction for restaurant dialogue turns."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from restaurant_assistant.date_utils import DAY_MODIFIERS, RELATIVE_DAYS
from restaurant_assistant.llm_runtime import llm_backend_error
from restaurant_assistant.preprocessing import normalize_area, normalize_food, normalize_price, normalize_time


ALLOWED_SLOT_KEYS = {
    "food",
    "food_candidates",
    "cuisine_group",
    "dish",
    "area",
    "pricerange",
    "day",
    "relative_day",
    "day_modifier",
    "time",
    "people",
    "restaurant_name",
    "booking_reference",
}

STRICT_MODEL_INTENTS = {
    "search",
    "list",
    "restaurant_info",
    "book",
    "update_booking",
    "cancel_booking",
    "booking_list",
    "greeting",
    "thanks",
    "goodbye",
    "unsupported",
}

STRICT_MODEL_SLOT_KEYS = {
    "food",
    "food_candidates",
    "cuisine_group",
    "area",
    "pricerange",
    "day",
    "time",
    "people",
    "restaurant_name",
    "booking_reference",
}

MODEL_INTENT_ALIASES = {
    "update_booking": "reschedule",
    "cancel_booking": "cancel",
}


def adapter_slot_prompt(text: str) -> str:
    """Return the shared instruction format for adapter training and inference."""
    return (
        "Task: Extract the restaurant assistant intent and slots.\n"
        "Return only one valid minified JSON object.\n"
        "Do not explain.\n"
        "Do not use markdown.\n"
        "Allowed intents: search, list, restaurant_info, book, update_booking, cancel_booking, "
        "booking_list, greeting, thanks, goodbye, unsupported.\n"
        "Allowed slots: food, food_candidates, cuisine_group, area, pricerange, day, time, people, "
        "restaurant_name, booking_reference.\n"
        f"User: {text}\n"
        "JSON:"
    )


def _raw_output_preview(text: str, limit: int = 300) -> str:
    """Return a compact, bounded preview suitable for errors and reports."""
    preview = " ".join(str(text).split())
    if len(preview) > limit:
        return preview[: limit - 3] + "..."
    return preview


def _text_after_last_json_marker(text: str) -> str:
    candidate_text = str(text)
    marker_position = candidate_text.rfind("JSON:")
    if marker_position != -1:
        candidate_text = candidate_text[marker_position + len("JSON:") :]
    return candidate_text


def parse_llm_json_output(text: str) -> dict[str, Any]:
    """Find the first complete slot JSON object in model-generated text."""
    candidate_text = _text_after_last_json_marker(text)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", candidate_text):
        try:
            parsed, _ = decoder.raw_decode(candidate_text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "intent" in parsed and "slots" in parsed:
            return parsed

    preview = _raw_output_preview(text)
    raise ValueError(f"LLM output did not contain a valid intent/slots JSON object. Raw output: {preview!r}")


def strict_parse_llm_json_output(text: str) -> dict[str, Any]:
    """Parse exactly one raw intent/slots JSON object with no prompt text."""

    candidate_text = str(text).strip()
    decoder = json.JSONDecoder()
    try:
        parsed, end = decoder.raw_decode(candidate_text)
    except json.JSONDecodeError as exc:
        preview = _raw_output_preview(text)
        raise ValueError(f"LLM output was not strict JSON. Raw output: {preview!r}") from exc
    if candidate_text[end:].strip():
        preview = _raw_output_preview(text)
        raise ValueError(f"LLM output had trailing text after JSON. Raw output: {preview!r}")
    if not isinstance(parsed, dict) or set(parsed) != {"intent", "slots"}:
        raise ValueError('Strict JSON output must contain only "intent" and "slots" keys')
    if not isinstance(parsed["intent"], str) or not isinstance(parsed["slots"], dict):
        raise ValueError('Strict JSON output must use a string "intent" and object "slots"')
    return parsed


def canonicalize_model_intent(intent: Any) -> str:
    """Map strict-model labels onto internal assistant intent labels."""

    if not isinstance(intent, str):
        return "unknown"
    normalized = intent.strip()
    return MODEL_INTENT_ALIASES.get(normalized, normalized)


def _first_json_value_after_key(text: str, key: str) -> Any:
    decoder = json.JSONDecoder()
    pattern = re.compile(rf'"{re.escape(key)}"\s*:\s*')
    for match in pattern.finditer(text):
        try:
            value, _ = decoder.raw_decode(text, match.end())
        except json.JSONDecodeError:
            continue
        return value
    raise ValueError(f"LLM output did not contain a valid value for {key!r}")


def repair_llm_json_output(text: str) -> tuple[dict[str, Any], str]:
    """Repair a narrow malformed intent/slots fragment into canonical JSON."""
    try:
        parsed = parse_llm_json_output(text)
    except ValueError:
        parsed = None
    if parsed is not None:
        repaired_output = json.dumps(parsed, ensure_ascii=True, separators=(",", ":"))
        return parsed, repaired_output

    candidate_text = _text_after_last_json_marker(text)
    intent = _first_json_value_after_key(candidate_text, "intent")
    if not isinstance(intent, str):
        raise ValueError("LLM output intent was not a string")

    slots_marker = re.search(r'"slots"\s*:\s*', candidate_text)
    if slots_marker is None:
        raise ValueError("LLM output did not contain a slots fragment")

    slot_fragment = candidate_text[slots_marker.end() :]
    slots: dict[str, Any] = {}
    key_pattern = re.compile(r'"(?P<key>[^"]+)"\s*:\s*')
    decoder = json.JSONDecoder()
    for match in key_pattern.finditer(slot_fragment):
        key = match.group("key")
        if key not in ALLOWED_SLOT_KEYS or key in slots:
            continue
        try:
            value, _ = decoder.raw_decode(slot_fragment, match.end())
        except json.JSONDecodeError:
            continue
        slots[key] = value

    repaired = {"intent": intent, "slots": slots}
    repaired_output = json.dumps(repaired, ensure_ascii=True, separators=(",", ":"))
    return repaired, repaired_output


def _repaired_slots_shape(text: str) -> tuple[bool, bool]:
    """Return whether output has dangling slots and a complete slots object."""
    candidate_text = _text_after_last_json_marker(text).strip()
    slots_marker = re.search(r'"slots"\s*:\s*', candidate_text)
    if slots_marker is None:
        return False, False
    slot_fragment = candidate_text[slots_marker.end() :]
    dangling_slots = not slot_fragment.strip()
    try:
        slots_value, _ = json.JSONDecoder().raw_decode(slot_fragment)
    except json.JSONDecodeError:
        return dangling_slots, False
    return dangling_slots, isinstance(slots_value, dict)


SUPPORTED_AREAS = {"centre", "north", "south", "east", "west"}
SUPPORTED_PRICES = {"cheap", "moderate", "expensive"}
VAGUE_LLM_AREAS = {"cambridge", "town", "around town", "city"}
SUPPORTED_FOODS = {
    "african",
    "asian oriental",
    "british",
    "cantonese",
    "chinese",
    "european",
    "french",
    "fusion",
    "gastropub",
    "greek",
    "indian",
    "international",
    "italian",
    "jamaican",
    "japanese",
    "korean",
    "lebanese",
    "mediterranean",
    "mexican",
    "modern european",
    "north american",
    "persian",
    "polish",
    "portuguese",
    "romanian",
    "seafood",
    "spanish",
    "thai",
    "turkish",
    "vegetarian",
    "vietnamese",
}

FOOD_ALIASES = {
    "pizza": "italian",
    "spaghetti": "italian",
    "pasta": "italian",
    "portugese": "portuguese",
    "west african": "african",
    "east african": "african",
    "north african": "african",
    "south african": "african",
    "south asian": "indian",
    "east asian": "asian oriental",
    "south east asian": "asian oriental",
    "southeast asian": "asian oriental",
    "asian": "asian oriental",
    "veggie": "vegetarian",
    "mediteranian": "mediterranean",
    "mediterranian": "mediterranean",
    "arab": "lebanese",
    "arabic": "lebanese",
    "middle eastern": "lebanese",
    "middle-eastern": "lebanese",
}

CUISINE_GROUP_SUGGESTIONS = {
    "Middle Eastern": {
        "patterns": [r"\bmiddle[- ]eastern\b", r"\barab(?:ic|s)?\b"],
        "foods": ["lebanese", "turkish", "mediterranean"],
    },
    "South Asian": {
        "patterns": [r"\bsouth[- ]asian\b"],
        "foods": ["indian"],
    },
    "Southeast Asian": {
        "patterns": [r"\bsouth[- ]?east[- ]asian\b|\bsoutheast[- ]asian\b"],
        "foods": ["thai", "vietnamese", "asian oriental"],
    },
    "East Asian": {
        "patterns": [r"\beast[- ]asian\b"],
        "foods": ["chinese", "cantonese", "japanese", "korean", "asian oriental"],
    },
    "North African": {
        "patterns": [r"\bnorth[- ]african\b"],
        "foods": ["african"],
    },
    "West African": {
        "patterns": [r"\bwest[- ]african\b"],
        "foods": ["african"],
    },
}

DISH_CUISINE_SUGGESTIONS = {
    "chicken and rice": {
        "patterns": [r"\bchicken\b.*\brice\b|\brice\b.*\bchicken\b"],
        "foods": ["indian", "chinese", "thai", "lebanese", "turkish"],
    },
    "cake or dessert": {
        "patterns": [
            r"\bcakes?\b",
            r"\bdesserts?\b",
            r"\bdeserts?\b",
            r"\bsweets?\b",
            r"\bpudding\b",
            r"\bpastr(?:y|ies)\b",
        ],
        "foods": ["british", "european", "french", "international"],
    },
    "burger": {
        "patterns": [r"\bburgers?\b"],
        "foods": ["north american", "gastropub"],
    },
    "curry": {
        "patterns": [r"\bcurr(?:y|ies)\b"],
        "foods": ["indian", "thai"],
    },
    "noodles": {
        "patterns": [r"\bnoodles?\b"],
        "foods": ["chinese", "asian oriental", "thai", "vietnamese"],
    },
    "sushi": {
        "patterns": [r"\bsushi\b"],
        "foods": ["japanese"],
    },
    "lamb and rice": {
        "patterns": [r"\blamb\b.*\brice\b|\brice\b.*\blamb\b"],
        "foods": ["lebanese", "turkish", "mediterranean"],
    },
    "mandi": {
        "patterns": [r"\bmandi\b|\bmandhi\b|\bmandy\b"],
        "foods": ["lebanese", "turkish", "mediterranean"],
    },
    "grilled meat or kebab": {
        "patterns": [r"\bgrilled\s+meat\b|\bkebabs?\b"],
        "foods": ["turkish", "lebanese", "mediterranean"],
    },
}

UNSUPPORTED_FOOD_PATTERNS = {
    "egyptian": r"\begyptian\b|\begyption\b",
    "moroccan": r"\bmorocc?an\b|\bmorccan\b|\bmorocan\b",
    "yemeni": r"\byemeni\b|\byemen\b",
}

WEEKDAY_VALUES = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
DAY_VALUES = set(WEEKDAY_VALUES) | {"today", "tomorrow"}

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

ALLOWED_INTENTS = {
    "search",
    "book",
    "reschedule",
    "update_booking",
    "cancel",
    "cancel_booking",
    "greeting",
    "thanks",
    "goodbye",
    "alternative",
    "list",
    "correct",
    "booking_info",
    "booking_list",
    "table_view",
    "restaurant_info",
    "filter_info",
    "cuisine_help",
    "dish_preference",
    "distance_info",
    "date_clarification",
    "unsupported",
    "unknown",
}

RULE_GUARDED_INTENTS = {
    "book",
    "booking_info",
    "booking_list",
    "cancel",
    "cancel_booking",
    "correct",
    "cuisine_help",
    "date_clarification",
    "dish_preference",
    "distance_info",
    "filter_info",
    "greeting",
    "goodbye",
    "restaurant_info",
    "reschedule",
    "update_booking",
    "table_view",
    "thanks",
    "unsupported",
}


@dataclass
class SlotExtractionResult:
    intent: str
    slots: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    used_llm: bool = False
    llm_attempted: bool = False
    llm_parse_success: bool = False
    llm_repair_success: bool = False
    llm_repair_weak: bool = False
    llm_intent_trusted: bool = False
    llm_slots_trusted: bool = False
    llm_meaningful_slot_contribution: bool = False
    llm_raw_output: str | None = None
    llm_repaired_output: str | None = None
    errors: list[str] = field(default_factory=list)
    unsupported_slots: dict[str, str] = field(default_factory=dict)


class RuleBasedSlotExtractor:
    """Deterministic extractor tuned to common MultiWOZ restaurant values."""

    def extract(self, message: str) -> SlotExtractionResult:
        text = self._normalize_message(message)
        slots: dict[str, Any] = {}
        unsupported_slots: dict[str, str] = {}

        cuisine_group = self._extract_cuisine_group(text)
        if cuisine_group:
            group_label, food_candidates = cuisine_group
            slots["cuisine_group"] = group_label
            slots["food_candidates"] = food_candidates
        else:
            food = self._extract_food(text)
            if food:
                slots["food"] = food
            else:
                dish_preference = self._extract_dish_preference(text)
                if dish_preference:
                    dish, food_candidates = dish_preference
                    slots["dish"] = dish
                    slots["food_candidates"] = food_candidates
                    unsupported_food = None
                else:
                    unsupported_food = self._extract_unsupported_food(text)
                if unsupported_food:
                    unsupported_slots["food"] = unsupported_food

        area = self._extract_area(text)
        if area:
            slots["area"] = area
        else:
            unsupported_area = self._extract_unsupported_area(text)
            if unsupported_area:
                unsupported_slots["area"] = unsupported_area

        price = self._extract_price(text)
        if price:
            slots["pricerange"] = price

        relative_day = self._extract_relative_day(text)
        if relative_day:
            slots["relative_day"] = relative_day

        day = self._extract_day(text)
        if day:
            slots["day"] = day
            day_modifier = self._extract_day_modifier(text, day)
            if day_modifier:
                slots["day_modifier"] = day_modifier
        else:
            unsupported_day = self._extract_unsupported_day(text)
            if unsupported_day:
                unsupported_slots["day"] = unsupported_day

        time = self._extract_time(text)
        if time:
            slots["time"] = time

        people = self._extract_people(text)
        if people:
            slots["people"] = people

        booking_reference = self._extract_booking_reference(text)
        if booking_reference:
            slots["booking_reference"] = booking_reference

        intent = self._detect_intent(text, slots)
        return SlotExtractionResult(intent=intent, slots=slots, unsupported_slots=unsupported_slots)

    def _normalize_message(self, message: str) -> str:
        text = message.lower().replace(">", " ")
        replacements = [
            (r"\bothe\s+r\b", "other"),
            (r"\bresuratns\b", "restaurants"),
            (r"\bresurants\b", "restaurants"),
            (r"\bresturants\b", "restaurants"),
            (r"\brestraunts\b", "restaurants"),
            (r"\brresutrants\b", "restaurants"),
            (r"\bresutrants\b", "restaurants"),
            (r"\bresturants\b", "restaurants"),
            (r"\bresuratn\b", "restaurant"),
            (r"\bresurant\b", "restaurant"),
            (r"\bresturant\b", "restaurant"),
            (r"\brestraunt\b", "restaurant"),
            (r"\brresutrant\b", "restaurant"),
            (r"\bresutrant\b", "restaurant"),
            (r"\br+esu?t+r?a?u?r?a?n?t\b", "restaurant"),
            (r"\bcusines\b", "cuisines"),
            (r"\bcusine\b", "cuisine"),
            (r"\blists\b", "list"),
            (r"\bmoderatley\b", "moderately"),
            (r"\bmoderatly\b", "moderately"),
            (r"\bplese\b", "please"),
            (r"\bhellow\b|\bhelo\b|\bhelllo\b", "hello"),
            (r"\btbale\b", "table"),
            (r"\bitalion\b", "italian"),
            (r"\begyption\b", "egyptian"),
            (r"\blamd\b", "lamb"),
            (r"\bmandhi\b", "mandi"),
            (r"\bcancle\b", "cancel"),
            (r"\bcancelation\b", "cancellation"),
            (r"\bresechdeule\b", "reschedule"),
            (r"\bresechedule\b", "reschedule"),
            (r"\breschdeule\b", "reschedule"),
            (r"\breschduele\b", "reschedule"),
            (r"\breschdule\b", "reschedule"),
            (r"\breschudle\b", "reschedule"),
            (r"\bthursdat\b|\bthurday\b|\bthrusday\b", "thursday"),
        ]
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)
        return " ".join(text.split())

    def _extract_food(self, text: str) -> str | None:
        for phrase, normalized in sorted(FOOD_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
            if re.search(rf"\b{re.escape(phrase)}\b", text):
                return normalized
        for food in sorted(SUPPORTED_FOODS, key=len, reverse=True):
            if re.search(rf"\b{re.escape(food)}\b", text):
                return food
        return None

    def _extract_unsupported_food(self, text: str) -> str | None:
        for label, pattern in UNSUPPORTED_FOOD_PATTERNS.items():
            if re.search(pattern, text):
                return label
        return None

    def _extract_dish_preference(self, text: str) -> tuple[str, list[str]] | None:
        for label, spec in DISH_CUISINE_SUGGESTIONS.items():
            patterns = spec["patterns"]
            if any(re.search(pattern, text) for pattern in patterns):
                foods = [food for food in spec["foods"] if food in SUPPORTED_FOODS]
                if foods:
                    return label, foods
        return None

    def _extract_cuisine_group(self, text: str) -> tuple[str, list[str]] | None:
        for label, spec in CUISINE_GROUP_SUGGESTIONS.items():
            patterns = spec["patterns"]
            if any(re.search(pattern, text) for pattern in patterns):
                foods = [food for food in spec["foods"] if food in SUPPORTED_FOODS]
                if foods:
                    return label, foods
        return None

    def _extract_area(self, text: str) -> str | None:
        text = re.sub(r"\bmiddle[- ]eastern\b", "middleeastern", text)
        text = re.sub(r"\b(?:south|east|south[- ]?east|southeast)\s+asian\b", "regionalasian", text)
        text = re.sub(r"\b(?:west|east|north|south)\s+african\b", "african", text)
        text = re.sub(r"\bnorth\s+american\b", "northamerican", text)
        area_patterns = {
            "centre": [
                r"\bcentre\b",
                r"\bcenter\b",
                r"\bcity centre\b",
                r"\bcity center\b",
                r"\bcentral\b",
                r"\bcity\b",
            ],
            "north": [r"\bnorth\b", r"\bnorthern\b"],
            "south": [r"\bsouth\b", r"\bsouthern\b"],
            "east": [r"\beast\b", r"\beastern\b"],
            "west": [r"\bwest\b", r"\bwestern\b"],
        }
        for area, patterns in area_patterns.items():
            if any(re.search(pattern, text) for pattern in patterns):
                return area
        return None

    def _extract_unsupported_area(self, text: str) -> str | None:
        unsupported_patterns = {
            "countryside": r"\b(countryside|rural|village|villages|country side)\b",
            "outside centre": r"\b(outside|outskirts|suburbs?)\b",
            "near me": r"\bnear me\b|\bnearby me\b|\bclose to me\b",
        }
        for label, pattern in unsupported_patterns.items():
            if re.search(pattern, text):
                return label
        return None

    def _extract_price(self, text: str) -> str | None:
        numeric_range = re.search(
            r"(?:£|\bpounds?\b|\bpriced?\b|\bprice\b|\baround\b|\babout\b|\bunder\b|\bmax\b|\bmaximum\b)?\s*"
            r"(\d{1,3})\s*(?:-|to|–|—)\s*[£! ]*(\d{1,3})\s*(?:pounds?)?",
            text,
        )
        if numeric_range:
            first = int(numeric_range.group(1))
            second = int(numeric_range.group(2))
            highest = max(first, second)
            lowest = min(first, second)
            if highest > 100 and lowest <= 25:
                return "moderate"
            if highest <= 10:
                return "cheap"
            if highest <= 25:
                return "moderate"
            return "expensive"
        single_price = re.search(r"\b(?:under|max|maximum|around|about)\s*£?\s*(\d{1,3})\s*(?:pounds?)?\b", text)
        if single_price:
            amount = int(single_price.group(1))
            if amount <= 10:
                return "cheap"
            if amount <= 25:
                return "moderate"
            return "expensive"
        price_patterns = {
            "cheap": [r"\bcheap\b", r"\bbudget\b", r"\binexpensive\b", r"\blow cost\b"],
            "moderate": [r"\bmoderate\b", r"\bmoderately\b", r"\bmid[- ]?range\b", r"\breasonable\b"],
            "expensive": [r"\bexpensive\b", r"\bupscale\b", r"\bupmarket\b", r"\bhigh end\b"],
        }
        for price, patterns in price_patterns.items():
            if any(re.search(pattern, text) for pattern in patterns):
                return price
        return None

    def _extract_day(self, text: str) -> str | None:
        matches = []
        for day in WEEKDAY_VALUES:
            match = re.search(rf"\b{day}\b", text)
            if match:
                matches.append((match.start(), day))
        if matches:
            return min(matches)[1]
        return None

    def _extract_relative_day(self, text: str) -> str | None:
        if re.search(r"\bday after tomorrow\b", text):
            return "day_after_tomorrow"
        if re.search(r"\b(the day after|day after|following day)\b", text):
            return "day_after"
        if re.search(r"\btomorrow\b", text):
            return "tomorrow"
        if re.search(r"\btoday\b", text):
            return "today"
        return None

    def _extract_day_modifier(self, text: str, day: str) -> str | None:
        escaped_day = re.escape(day)
        if re.search(rf"\bnext\s+week\s+{escaped_day}\b|\b{escaped_day}\s+next\s+week\b", text):
            return "next_week"
        if re.search(rf"\bnext\s+{escaped_day}\b", text):
            return "next"
        if re.search(rf"\bthis\s+{escaped_day}\b", text):
            return "this"
        return None

    def _extract_unsupported_day(self, text: str) -> str | None:
        return None

    def _extract_time(self, text: str) -> str | None:
        if re.search(r"\b\d{1,3}\s*(?:-|to|–|—)\s*[£! ]*\d{1,3}\b", text) and not re.search(r"\b(am|pm)\b", text):
            return None
        if re.search(r"\bmidnight\b", text):
            return "00:00"
        if re.search(r"\bnoon\b", text):
            return "12:00"
        clock = re.search(r"\b([01]?\d|2[0-3])[:.]([0-5]\d)\s*(am|pm)?\b", text)
        if clock:
            suffix = clock.group(3) or ""
            return normalize_time(f"{clock.group(1)}:{clock.group(2)}{suffix}")
        with_suffix = re.search(r"\b(?:at|around|for|by)?\s*(1[0-2]|0?[1-9])\s*(am|pm)\b", text)
        if with_suffix:
            return normalize_time(f"{with_suffix.group(1)}{with_suffix.group(2)}")
        ambiguous = re.search(r"\b(?:at|around|by)\s+(1[0-2]|0?[1-9])\b", text)
        if ambiguous:
            hour = int(ambiguous.group(1))
            if 1 <= hour <= 9:
                hour += 12
            return normalize_time(f"{hour}:00")
        return None

    def _extract_people(self, text: str) -> int | None:
        number = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
        patterns = [
            rf"\b(?:party of|table for)\s+{number}\b",
            rf"\bfor\s+{number}\s+(?:people|persons|guests|diners)\b",
            rf"\b{number}\s+(?:people|persons|guests|diners)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                token = match.group(1)
                return NUMBER_WORDS.get(token, int(token) if token.isdigit() else None)
        return None

    def _extract_booking_reference(self, text: str) -> str | None:
        match = re.search(r"\b((?:bk|sim)-[a-z0-9]{6})\b", text, flags=re.IGNORECASE)
        return match.group(1).upper() if match else None

    def _detect_intent(self, text: str, slots: dict[str, Any]) -> str:
        if re.search(
            r"\b(gun|weapon|firearm|knife|drugs?|passport|credit card|taxi|hotel|train|flight|fireworks?|dentist)\b",
            text,
        ):
            return "unsupported"
        if re.search(r"\b(goodbye|bye|see you|see ya|farewell)\b", text):
            return "goodbye"
        if re.search(r"^(hi|hello|hey|good morning|good afternoon|good evening)\b", text) and not re.search(
            r"\b(book|reserve|find|search|restaurants?|cuisines?|food|what to eat|suggest)\b",
            text,
        ):
            return "greeting"
        if re.search(r"\b(thanks|thank you|cheers|ta)\b", text):
            return "thanks"
        if slots.get("cuisine_group"):
            if re.search(r"\b(other|another|alternatives?|else|anymore|any more|more options?|more restaurants?)\b", text):
                return "alternative"
            return "list"
        if re.search(r"\b(book|reserve)\b", text):
            return "book"
        if re.search(r"\b(make|create|set up)\s+(?:a\s+|another\s+|one\s+more\s+|new\s+)?(booking|reservation)\b", text):
            return "book"
        if slots.get("dish") and slots.get("food_candidates"):
            return "dish_preference"
        if re.search(r"\bwhat\b.*\brestaurants?\b.*\b(?:are there|available)\b", text):
            return "list"
        if re.search(r"\bhow\s+far\b|\bdistance\b|\btravel\s+time\b|\bnear\s+to\b", text) or (
            "area" in slots and re.search(r"\b(?:is|are)\b.*\bnear\b", text)
        ):
            return "distance_info"
        if re.search(
            r"\bwhat\s+areas?\b|\bwhich\s+areas?\b|\bareas?\s+(?:are|can|to)\b|\bfilter\s+through\b|"
            r"\bwhat\s+price\s+ranges?\b|\bprice\s+ranges?.*\bfilter\b|"
            r"\bwhich\s+(?:parts|areas)\b.*\b(?:searchable|supported|filter)\b",
            text,
        ):
            return "filter_info"
        if re.search(
            r"\b(what\s+(?:food\s+)?cuisines?|cuisines?.*look for|food types?.*(?:available|can|look)|what to eat|unsure what to eat|suggest.*cuisines?)\b",
            text,
        ) and not slots.get("food"):
            return "cuisine_help"
        if re.search(r"\b(cancel|delete|remove)\b", text) and re.search(
            r"\b(bookings?|reservations?|table|it|this|(?:bk|sim)-[a-z0-9]{6})\b", text
        ):
            return "cancel"
        if re.search(r"\b(reschedule|move|change|amend|update)\b", text) and re.search(
            r"\b(booking|reservation|table|it|this|time|day|(?:bk|sim)-[a-z0-9]{6})\b", text
        ):
            return "reschedule"
        if re.search(r"\b(reschedule|move|change|amend|update)\b", text) and any(
            slot in slots for slot in ("day", "relative_day", "time", "people")
        ):
            return "reschedule"
        if re.search(
            r"\b(address|postcode|post code|phone|telephone|contact|location|located|where is|where's)\b",
            text,
        ):
            return "restaurant_info"
        if re.search(r"\b(tell me about|details?|status|show|what are|what is|what's|about)\b", text) and (
            slots.get("booking_reference") or re.search(r"\b(booking|reservation|reference|it|this)\b", text)
        ):
            return "booking_info"
        if re.search(
            r"\b(list|show|view|what|which|any|all)\b.*\b(bookings?|reservations?)\b|\b(bookings?|reservations?)\b.*\b(there|current|made|have)\b",
            text,
        ):
            return "booking_list"
        if re.search(r"\b(?:as|in)\s+a\s+table\b|\btable\s+format\b|\bshow\s+as\s+table\b", text):
            return "table_view"
        if re.search(r"\b(tell me about|details?|status|show|what is|what's|about)\b", text) and (
            slots.get("booking_reference") or re.search(r"\b(booking|reservation|reference|it|this)\b", text)
        ):
            return "booking_info"
        if re.search(
            r"\b(i said|actually|instead|meant|no pm|no am|not pm|not am|i booked it for|it was for|should be|why is)\b",
            text,
        ) and any(slot in slots for slot in ("day", "relative_day", "time", "people")):
            return "correct"
        if re.fullmatch(r"(?:next|following)\s+week", text):
            return "date_clarification"
        if re.search(
            r"\ball\s+of\s+them\b|\blist\s+all\b|\ball\b.*\brestaurants?\b|"
            r"\bshow\s+every\s+restaurants?\b|\brestaurants?\b.*\blist\s+all\b",
            text,
        ):
            return "list"
        if "pricerange" in slots and re.search(r"\b(that are|which are|ones that|around|about)\b", text):
            return "list"
        if re.search(r"\b(other|different|another)\s+cuisines?\b|\bnot\s+(?:just|only)\b", text) and re.search(
            r"\b(list|restaurants?|cuisines?|places?)\b", text
        ):
            return "list"
        if re.search(r"\b(other|another|alternatives?|else|nearby|anymore|any more|more options?|more restaurants?)\b", text):
            return "alternative"
        if re.search(
            r"\b(list|options?|show me all|all restaurants|restaurants that are|restaurants in|restaurants for)\b",
            text,
        ):
            return "list"
        if re.search(r"\b(see|show|find)\b.*\b(restaurants|places)\b", text):
            return "list"
        if re.search(r"\b(find|search|need|looking|recommend|show|restaurant|eat|food|cuisine|place)\b", text):
            return "search"
        if any(slot in slots for slot in ("food", "area", "pricerange")):
            return "search"
        return "unknown"


def validate_slots(slots: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize slots before they can update dialogue state."""

    valid: dict[str, Any] = {}
    if "food" in slots:
        food = normalize_food(slots["food"])
        food = FOOD_ALIASES.get(food, food)
        if food in SUPPORTED_FOODS:
            valid["food"] = food
    if "dish" in slots:
        dish = re.sub(r"[^a-z0-9 /-]", "", str(slots["dish"]).strip().lower())
        if dish:
            valid["dish"] = dish[:80]
    if "cuisine_group" in slots:
        group = str(slots["cuisine_group"]).strip()
        known_groups = {label.lower(): label for label in CUISINE_GROUP_SUGGESTIONS}
        if group.lower() in known_groups:
            valid["cuisine_group"] = known_groups[group.lower()]
    if "food_candidates" in slots and isinstance(slots["food_candidates"], list):
        candidates = []
        for candidate in slots["food_candidates"]:
            food = normalize_food(candidate)
            food = FOOD_ALIASES.get(food, food)
            if food in SUPPORTED_FOODS and food not in candidates:
                candidates.append(food)
        if candidates:
            valid["food_candidates"] = candidates
    if "area" in slots:
        area = normalize_area(slots["area"])
        if area in SUPPORTED_AREAS:
            valid["area"] = area
    if "pricerange" in slots:
        price = normalize_price(slots["pricerange"])
        if price in SUPPORTED_PRICES:
            valid["pricerange"] = price
    if "day" in slots:
        day = str(slots["day"]).strip().lower()
        if day in DAY_VALUES:
            valid["day"] = day
    if "relative_day" in slots:
        relative_day = str(slots["relative_day"]).strip().lower()
        if relative_day in RELATIVE_DAYS:
            valid["relative_day"] = relative_day
    if "day_modifier" in slots:
        day_modifier = str(slots["day_modifier"]).strip().lower()
        if day_modifier in DAY_MODIFIERS:
            valid["day_modifier"] = day_modifier
    if "time" in slots:
        time = normalize_time(str(slots["time"]))
        if re.fullmatch(r"\d{2}:\d{2}", time):
            valid["time"] = time
    if "people" in slots:
        try:
            people = int(slots["people"])
        except (TypeError, ValueError):
            people = 0
        if people > 0:
            valid["people"] = people
    if "restaurant_name" in slots:
        restaurant_name = re.sub(r"\s+", " ", str(slots["restaurant_name"]).strip().lower())
        restaurant_name = re.sub(r"[^a-z0-9 '&.-]", "", restaurant_name)
        if restaurant_name:
            valid["restaurant_name"] = restaurant_name[:120]
    if "booking_reference" in slots:
        reference = str(slots["booking_reference"]).strip().upper()
        if re.fullmatch(r"(?:BK|SIM)-[A-Z0-9]{6}", reference):
            valid["booking_reference"] = reference
    return valid


def validate_llm_slots(slots: Any) -> dict[str, Any]:
    """Validate generated slots with stricter rules for vague ontology values."""
    if not isinstance(slots, dict):
        return {}
    candidates = dict(slots)
    if "area" in candidates:
        raw_area = " ".join(str(candidates["area"]).strip().casefold().split())
        if raw_area in VAGUE_LLM_AREAS or normalize_area(raw_area) not in SUPPORTED_AREAS:
            candidates.pop("area")
    if "pricerange" in candidates:
        if normalize_price(candidates["pricerange"]) not in SUPPORTED_PRICES:
            candidates.pop("pricerange")
    if "booking_reference" in candidates:
        reference = str(candidates["booking_reference"]).strip().upper()
        if not re.fullmatch(r"(?:BK|SIM)-[A-Z0-9]{6}", reference):
            candidates.pop("booking_reference")
    return validate_slots(candidates)


def _explicitly_named_food(message: str, food: Any, cuisine_group: Any) -> bool:
    """Return whether a single cuisine name appears outside the broad group phrase."""
    normalized_message = re.sub(r"[^a-z0-9]+", " ", message.casefold()).strip()
    normalized_food = re.sub(r"[^a-z0-9]+", " ", str(food).casefold()).strip()
    normalized_group = re.sub(r"[^a-z0-9]+", " ", str(cuisine_group).casefold()).strip()
    if not normalized_food:
        return False
    if normalized_group:
        normalized_message = re.sub(
            rf"\b{re.escape(normalized_group)}\b",
            " ",
            normalized_message,
        )
    return re.search(rf"\b{re.escape(normalized_food)}\b", normalized_message) is not None


def merge_llm_slots(
    message: str,
    rule_slots: dict[str, Any],
    llm_slots: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Add only genuinely missing, non-duplicative generated slot values."""
    merged = dict(rule_slots)
    meaningful_contribution = False
    for key, value in llm_slots.items():
        if key in merged:
            continue
        if key == "food" and {"cuisine_group", "food_candidates"}.issubset(rule_slots):
            if not _explicitly_named_food(message, value, rule_slots["cuisine_group"]):
                continue
        if key in {"cuisine_group", "food_candidates"} and "food" in rule_slots:
            continue
        if key == "day" and "relative_day" in rule_slots:
            if str(value).strip().casefold() in {"today", "tomorrow", "tonight"}:
                continue
        if key == "relative_day" and "day" in rule_slots:
            if str(value).strip().casefold() in {"today", "tomorrow", "tonight"}:
                continue
        merged[key] = value
        meaningful_contribution = True
    return validate_slots(merged), meaningful_contribution


class OptionalLLMSlotExtractor:
    """Strict JSON LLM extractor with rule-based fallback."""

    def __init__(self, model_name: str = "google/flan-t5-small", *, num_beams: int = 1) -> None:
        self.model_name = model_name
        self.num_beams = max(1, int(num_beams))
        # Kept as an injectable test seam for existing callers. Normal inference
        # uses the tokenizer/model pair below rather than a Transformers pipeline.
        self._pipeline = None
        self._model = None
        self._tokenizer = None
        self._load_attempted = False
        self._load_error: str | None = None
        self._uses_adapter = False
        self.rule_extractor = RuleBasedSlotExtractor()

    def _load_model(self) -> bool:
        if self._pipeline is not None:
            return True
        if self._model is not None and self._tokenizer is not None:
            return True
        if self._load_attempted:
            return False
        self._load_attempted = True
        backend_error = llm_backend_error()
        if backend_error:
            self._load_error = backend_error
            return False
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            model_path = Path(self.model_name)
            if model_path.exists() and (model_path / "adapter_config.json").exists():
                from peft import AutoPeftModelForSeq2SeqLM

                self._uses_adapter = True
                self._tokenizer = AutoTokenizer.from_pretrained(model_path)
                self._model = AutoPeftModelForSeq2SeqLM.from_pretrained(model_path)
            else:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self._model.eval()
        except Exception as exc:
            self._load_error = str(exc)
            self._model = None
            self._tokenizer = None
            return False
        return True

    def _generate(self, prompt: str) -> str:
        generation_kwargs = {
            "max_new_tokens": 128,
            "do_sample": False,
            "num_beams": self.num_beams,
            "early_stopping": True,
            "repetition_penalty": 1.2,
        }
        if self._pipeline is not None:
            output = self._pipeline(prompt, **generation_kwargs)
            return str(output[0]["generated_text"])

        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True)
        try:
            input_device = self._model.get_input_embeddings().weight.device
            inputs = {key: value.to(input_device) for key, value in inputs.items()}
        except (AttributeError, RuntimeError):
            # CPU models already receive CPU tensors; some sharded/quantized
            # models manage placement internally and do not expose one device.
            pass

        import torch

        with torch.inference_mode():
            generated = self._model.generate(**inputs, **generation_kwargs)
        return str(self._tokenizer.decode(generated[0], skip_special_tokens=True))

    def extract(self, message: str) -> SlotExtractionResult:
        rule_result = self.rule_extractor.extract(message)
        if not self._load_model():
            if self._load_error and self._load_error not in rule_result.errors:
                rule_result.errors.append(self._load_error)
            return rule_result
        prompt = adapter_slot_prompt(message)
        raw_output: str | None = None
        repaired_output: str | None = None
        parse_success = False
        repair_success = False
        parse_error: ValueError | None = None
        try:
            raw_output = self._generate(prompt)
            try:
                parsed = parse_llm_json_output(raw_output)
                parse_success = True
            except ValueError as exc:
                parse_error = exc
                parsed, repaired_output = repair_llm_json_output(raw_output)
                repair_success = True

            raw_intent = parsed.get("intent")
            candidate_intent = canonicalize_model_intent(raw_intent)
            intent_is_valid = candidate_intent in ALLOWED_INTENTS
            llm_slots = validate_llm_slots(parsed.get("slots", {}))

            rule_intent_confident = rule_result.intent != "unknown"
            intent_conflict = (
                intent_is_valid
                and rule_intent_confident
                and candidate_intent != rule_result.intent
            )
            dangling_slots, complete_slots_object = _repaired_slots_shape(raw_output)
            repair_weak = repair_success and (
                (dangling_slots and not llm_slots)
                or not intent_is_valid
                or (not llm_slots and bool(rule_result.slots))
                or intent_conflict
            )

            llm_intent_trusted = False
            intent = rule_result.intent
            if rule_result.intent not in RULE_GUARDED_INTENTS:
                if parse_success and intent_is_valid:
                    intent = candidate_intent
                    llm_intent_trusted = True
                elif (
                    repair_success
                    and not repair_weak
                    and intent_is_valid
                    and (
                        complete_slots_object
                        or (candidate_intent != "unknown" and not intent_conflict)
                    )
                ):
                    intent = candidate_intent
                    llm_intent_trusted = True

            merged_slots, meaningful_slot_contribution = merge_llm_slots(
                message,
                rule_result.slots,
                llm_slots,
            )
            llm_slots_trusted = bool(llm_slots)

            return SlotExtractionResult(
                intent=intent,
                slots=merged_slots,
                confidence=0.9,
                used_llm=True,
                llm_attempted=True,
                llm_parse_success=parse_success,
                llm_repair_success=repair_success,
                llm_repair_weak=repair_weak,
                llm_intent_trusted=llm_intent_trusted,
                llm_slots_trusted=llm_slots_trusted,
                llm_meaningful_slot_contribution=meaningful_slot_contribution,
                llm_raw_output=raw_output,
                llm_repaired_output=repaired_output,
                errors=[str(parse_error)] if parse_error is not None else [],
                unsupported_slots=dict(rule_result.unsupported_slots),
            )
        except Exception as exc:
            fallback = rule_result
            fallback.llm_attempted = True
            fallback.llm_parse_success = parse_success
            fallback.llm_repair_success = repair_success
            fallback.llm_raw_output = _raw_output_preview(raw_output) if raw_output is not None else None
            fallback.llm_repaired_output = repaired_output
            if parse_error is not None:
                fallback.errors.append(str(parse_error))
            if not fallback.errors or str(exc) != fallback.errors[-1]:
                fallback.errors.append(str(exc))
            return fallback

    def _parse_json(self, text: str) -> dict[str, Any]:
        """Backward-compatible wrapper around the robust parser."""
        return parse_llm_json_output(text)


def extract_slots(message: str, *, use_llm: bool = False, model_name: str = "google/flan-t5-small") -> SlotExtractionResult:
    extractor = OptionalLLMSlotExtractor(model_name) if use_llm else RuleBasedSlotExtractor()
    result = extractor.extract(message)
    result.slots = validate_slots(result.slots)
    return result
