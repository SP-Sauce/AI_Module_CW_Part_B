"""Intent and slot extraction for restaurant dialogue turns."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from restaurant_assistant.date_utils import DAY_MODIFIERS, RELATIVE_DAYS
from restaurant_assistant.preprocessing import normalize_area, normalize_food, normalize_price, normalize_time


SUPPORTED_AREAS = {"centre", "north", "south", "east", "west"}
SUPPORTED_PRICES = {"cheap", "moderate", "expensive"}
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
        "patterns": [r"\bmiddle[- ]eastern\b"],
        "foods": ["lebanese", "turkish", "mediterranean"],
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
}

UNSUPPORTED_FOOD_PATTERNS = {
    "egyptian": r"\begyptian\b|\begyption\b",
    "yemeni": r"\byemeni\b|\byemen\b",
}

DAY_VALUES = {
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "today",
    "tomorrow",
}

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


@dataclass
class SlotExtractionResult:
    intent: str
    slots: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    used_llm: bool = False
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
        for day in DAY_VALUES:
            if day in {"today", "tomorrow"}:
                continue
            if re.search(rf"\b{day}\b", text):
                return day
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
        if re.search(r"\b(gun|weapon|firearm|knife|drugs?|passport|credit card)\b", text):
            return "unsupported"
        if slots.get("dish") and slots.get("food_candidates"):
            return "dish_preference"
        if slots.get("cuisine_group"):
            if re.search(r"\b(other|another|alternatives?|else|anymore|any more|more options?|more restaurants?)\b", text):
                return "alternative"
            return "list"
        if re.search(r"\bwhat\b.*\brestaurants?\b.*\b(?:are there|available)\b", text):
            return "list"
        if re.search(r"\bhow\s+far\b|\bdistance\b|\btravel\s+time\b|\bnear\s+to\b", text):
            return "distance_info"
        if re.search(r"\bwhat\s+areas?\b|\bwhich\s+areas?\b|\bareas?\s+(?:are|can|to)\b|\bfilter\s+through\b", text):
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
        if re.search(r"\b(book|reserve)\b", text):
            return "book"
        if re.search(r"\b(make|create|set up)\s+(?:a\s+|another\s+|one\s+more\s+|new\s+)?(booking|reservation)\b", text):
            return "book"
        if text.strip() in {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}:
            return "greeting"
        if re.search(r"\b(thanks|thank you|cheers|ta)\b", text):
            return "thanks"
        if re.fullmatch(r"(?:next|following)\s+week", text):
            return "date_clarification"
        if re.search(r"\ball\s+of\s+them\b|\blist\s+all\b|\ball\b.*\brestaurants?\b|\brestaurants?\b.*\blist\s+all\b", text):
            return "list"
        if "pricerange" in slots and re.search(r"\b(that are|which are|ones that|around|about|priced?)\b", text):
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
    if "booking_reference" in slots:
        reference = str(slots["booking_reference"]).strip().upper()
        if re.fullmatch(r"(?:BK|SIM)-[A-Z0-9]{6}", reference):
            valid["booking_reference"] = reference
    return valid


class OptionalLLMSlotExtractor:
    """Strict JSON LLM extractor with rule-based fallback."""

    def __init__(self, model_name: str = "google/flan-t5-small") -> None:
        self.model_name = model_name
        self._pipeline = None
        self.rule_extractor = RuleBasedSlotExtractor()

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline

            self._pipeline = pipeline("text2text-generation", model=self.model_name)
        except Exception:
            self._pipeline = False
        return self._pipeline

    def extract(self, message: str) -> SlotExtractionResult:
        pipe = self._load_pipeline()
        if not pipe:
            return self.rule_extractor.extract(message)
        prompt = (
            "Extract restaurant assistant intent and slots as compact JSON. "
            "Allowed intents: search, book, reschedule, cancel, greeting, thanks, alternative, list, correct, booking_info, booking_list, table_view, restaurant_info, filter_info, cuisine_help, dish_preference, distance_info, date_clarification, unsupported, unknown. "
            "Allowed slots: food, food_candidates, cuisine_group, dish, area, pricerange, day, relative_day, day_modifier, time, people. "
            f"User: {message}"
        )
        try:
            output = pipe(prompt, max_new_tokens=96, do_sample=False)[0]["generated_text"]
            parsed = self._parse_json(output)
            intent = parsed.get("intent", "unknown")
            slots = validate_slots(parsed.get("slots", {}))
            if intent not in {
                "search",
                "book",
                "reschedule",
                "cancel",
                "greeting",
                "thanks",
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
            }:
                intent = "unknown"
            return SlotExtractionResult(intent=intent, slots=slots, used_llm=True)
        except Exception as exc:
            fallback = self.rule_extractor.extract(message)
            fallback.errors.append(str(exc))
            return fallback

    def _parse_json(self, text: str) -> dict[str, Any]:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("LLM output did not contain JSON")
        parsed = json.loads(text[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("LLM output JSON was not an object")
        return parsed


def extract_slots(message: str, *, use_llm: bool = False, model_name: str = "google/flan-t5-small") -> SlotExtractionResult:
    extractor = OptionalLLMSlotExtractor(model_name) if use_llm else RuleBasedSlotExtractor()
    result = extractor.extract(message)
    result.slots = validate_slots(result.slots)
    return result
