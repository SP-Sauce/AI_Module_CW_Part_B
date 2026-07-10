"""Build deterministic response-generation JSONL datasets."""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from restaurant_assistant.config import get_settings
from restaurant_assistant.data_loader import load_restaurants
from restaurant_assistant.llm_generator import validate_generated_response
from restaurant_assistant.response_prompt import (
    RESPONSE_INSTRUCTION,
    build_response_input,
    format_evidence,
    parse_response_input,
    public_evidence_record,
)


DEFAULT_TRAIN_OUTPUT = ROOT / "data" / "training" / "response_generation_examples.jsonl"
DEFAULT_EVAL_OUTPUT = ROOT / "data" / "evaluation" / "response_generation_eval.jsonl"
DEFAULT_CHALLENGE_OUTPUT = ROOT / "data" / "evaluation" / "response_generation_challenge.jsonl"
DEFAULT_REPORT_JSON = ROOT / "reports" / "response_generation_dataset_report.json"
DEFAULT_REPORT_MD = ROOT / "reports" / "response_generation_dataset_report.md"

DEFAULT_TRAIN_COUNT = 800
DEFAULT_EVAL_COUNT = 160
DEFAULT_CHALLENGE_COUNT = 100
DEFAULT_SEED = 6062026

INSTRUCTION = RESPONSE_INSTRUCTION
REQUIRED_FIELDS = ("instruction", "input", "output")
BOOKING_REF_RE = re.compile(r"\bBK-[A-Z0-9]{6}\b")


CATEGORIES: tuple[str, ...] = (
    "greeting",
    "thanks",
    "goodbye",
    "exact_recommendation",
    "partial_match",
    "no_exact_match",
    "no_result",
    "missing_food",
    "missing_area",
    "missing_price",
    "missing_multiple_search",
    "list_results",
    "alternative_suggestions",
    "address_info",
    "phone_postcode_info",
    "cuisine_help",
    "booking_missing_day",
    "booking_missing_time",
    "booking_missing_people",
    "booking_missing_multiple",
    "booking_confirmation",
    "booking_reschedule",
    "booking_cancellation",
    "booking_info",
    "booking_list",
    "unsupported_hotel",
    "unsupported_train",
    "unsupported_taxi",
    "unsupported_payment",
    "unsupported_review",
    "cautious_halal",
    "cautious_allergy",
    "cautious_live_availability",
)

CATEGORY_INTENTS: dict[str, str] = {
    "greeting": "greeting",
    "thanks": "thanks",
    "goodbye": "goodbye",
    "exact_recommendation": "search",
    "partial_match": "search",
    "no_exact_match": "search",
    "no_result": "search",
    "missing_food": "search",
    "missing_area": "search",
    "missing_price": "search",
    "missing_multiple_search": "search",
    "list_results": "list",
    "alternative_suggestions": "alternative",
    "address_info": "restaurant_info",
    "phone_postcode_info": "restaurant_info",
    "cuisine_help": "cuisine_help",
    "booking_missing_day": "book",
    "booking_missing_time": "book",
    "booking_missing_people": "book",
    "booking_missing_multiple": "book",
    "booking_confirmation": "book",
    "booking_reschedule": "reschedule",
    "booking_cancellation": "cancel",
    "booking_info": "booking_info",
    "booking_list": "booking_list",
    "unsupported_hotel": "unsupported",
    "unsupported_train": "unsupported",
    "unsupported_taxi": "unsupported",
    "unsupported_payment": "unsupported",
    "unsupported_review": "unsupported",
    "cautious_halal": "restaurant_info",
    "cautious_allergy": "restaurant_info",
    "cautious_live_availability": "book",
}

PRICE_WORDS = {
    "cheap": ["cheap", "low-cost", "inexpensive", "budget-friendly", "not too pricey"],
    "moderate": ["moderate", "mid-range", "reasonably priced", "not too expensive", "fairly priced"],
    "expensive": ["expensive", "pricey", "high-end", "upmarket", "fancier"],
}
AREA_WORDS = {
    "centre": ["centre", "city centre", "central Cambridge", "downtown", "middle of town"],
    "north": ["north", "north side", "northern part", "up north", "north Cambridge"],
    "south": ["south", "south side", "southern part", "down south", "south Cambridge"],
    "east": ["east", "east side", "eastern part", "out east", "east Cambridge"],
    "west": ["west", "west side", "western part", "out west", "west Cambridge"],
}

TRAIN_FILLERS = [
    "please",
    "for me",
    "if you can",
    "in this chat",
    "for the restaurant demo",
    "when you have a moment",
    "with the loaded records",
    "thanks",
]
EVAL_FILLERS = [
    "would you",
    "for this search",
    "when possible",
    "using the restaurant records",
    "for my plan",
    "in Cambridge",
    "for this request",
    "clearly please",
]
CHALLENGE_FILLERS = [
    "ta",
    "mate",
    "cheers",
    "if poss",
    "not fussed",
    "quick one",
    "pls",
    "for tonight maybe",
]
FILLER_CONTEXTS = [
    "",
    "for dinner",
    "for lunch",
    "before I decide",
    "for this plan",
    "with the Cambridge records",
    "for a quick check",
    "for my booking",
    "while I compare options",
    "for later today",
    "for the weekend",
    "using only the loaded data",
]


@dataclass
class Example:
    split: str
    category: str
    intent: str
    row: dict[str, str]
    restaurant_ids: list[str] = field(default_factory=list)
    evidence_present: bool = False
    booking_state_present: bool = False
    safety_failure: str | None = None
    booking_reference_failure: str | None = None


@dataclass
class BuildResult:
    train_rows: list[dict[str, str]]
    eval_rows: list[dict[str, str]]
    challenge_rows: list[dict[str, str]]
    examples: dict[str, list[Example]]
    report: dict[str, Any]
    warnings: list[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build deterministic response-generation JSONL datasets.")
    parser.add_argument("--sample-data", action="store_true", help="Use bundled sample restaurants.")
    parser.add_argument("--train-count", type=int, default=DEFAULT_TRAIN_COUNT)
    parser.add_argument("--eval-count", type=int, default=DEFAULT_EVAL_COUNT)
    parser.add_argument("--challenge-count", type=int, default=DEFAULT_CHALLENGE_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--eval-output", type=Path, default=DEFAULT_EVAL_OUTPUT)
    parser.add_argument("--challenge-output", type=Path, default=DEFAULT_CHALLENGE_OUTPUT)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    return parser


def normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).casefold())).strip()


def restaurant_id(record: dict[str, Any]) -> str:
    source_id = str(record.get("source_id") or "").strip()
    if source_id:
        return source_id
    parts = [record.get("name"), record.get("area"), record.get("food"), record.get("pricerange")]
    return normalise_text("|".join(str(part or "") for part in parts))


def _field(record: dict[str, Any], key: str, fallback: str = "") -> str:
    return str(record.get(key) or fallback).strip()


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    return public_evidence_record(record)


def _evidence(records: Iterable[dict[str, Any]]) -> str:
    return format_evidence(records)


def evidence_records_from_input(input_text: str) -> list[dict[str, str]]:
    return parse_response_input(input_text).evidence_records


def _state(**slots: Any) -> str:
    return ", ".join(f"{key}={value}" for key, value in slots.items() if value not in (None, "", []))


def _input(
    *,
    intent: str,
    user: str,
    state: str = "",
    evidence: str = "",
    missing_slots: Iterable[str] | None = None,
) -> str:
    evidence_records = []
    if evidence:
        evidence_records = parse_response_input(f"Evidence: {evidence}").evidence_records
    return build_response_input(
        intent=intent,
        user=user,
        state=state,
        evidence_records=evidence_records,
        missing_slots=missing_slots,
    )


def _summary(record: dict[str, Any]) -> str:
    name = _field(record, "name", "the restaurant")
    price = _field(record, "pricerange")
    food = _field(record, "food")
    area = _field(record, "area")
    descriptor = " ".join(part for part in [price, food] if part)
    detail = f"{name} ({descriptor})" if descriptor else name
    if area:
        detail += f" in the {area} area"
    return detail


def article_for(phrase: str) -> str:
    first_word = re.sub(r"[^A-Za-z].*$", "", str(phrase or "").strip())
    if not first_word:
        return "a"
    return "an" if first_word[0].lower() in "aeiou" else "a"


def with_article(phrase: str) -> str:
    phrase = " ".join(str(phrase or "").split())
    if not phrase:
        return "a restaurant"
    if re.match(r"(?i)^(?:a|an|the)\s+", phrase):
        return phrase
    return f"{article_for(phrase)} {phrase}"


def people_text(count: Any) -> str:
    try:
        value = int(count)
    except (TypeError, ValueError):
        return "people"
    return "1 person" if value == 1 else f"{value} people"


def _people_text(index: int) -> str:
    return people_text(_people(index))


def restaurant_request_phrase(
    *,
    food: str = "",
    area: str = "",
    price: str = "",
) -> str:
    parts = [part for part in [price, food] if part]
    phrase = " ".join(parts) + " restaurant" if parts else "restaurant"
    if area:
        phrase += f" in the {area} area"
    return with_article(phrase)


def _constraint_phrase(record: dict[str, Any], *, override_area: str | None = None, override_price: str | None = None) -> str:
    price = override_price or _field(record, "pricerange")
    food = _field(record, "food")
    area = override_area or _field(record, "area")
    return restaurant_request_phrase(food=food, area=area, price=price)


def _word(pool: dict[str, list[str]], value: str, index: int) -> str:
    options = pool.get(value, [value])
    return options[index % len(options)]


def _filler(split: str, index: int) -> str:
    pools = {"train": TRAIN_FILLERS, "eval": EVAL_FILLERS, "challenge": CHALLENGE_FILLERS}
    base = pools[split][index % len(pools[split])]
    context = FILLER_CONTEXTS[(index // len(pools[split])) % len(FILLER_CONTEXTS)]
    return " ".join(part for part in [base, context] if part)


def _booking_reference(rng: random.Random) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    return "BK-" + "".join(rng.choice(alphabet) for _ in range(6))


def _other_area(area: str, index: int) -> str:
    areas = ["centre", "north", "south", "east", "west"]
    choices = [candidate for candidate in areas if candidate != area]
    return choices[index % len(choices)]


def _other_price(price: str, index: int) -> str:
    prices = ["cheap", "moderate", "expensive"]
    choices = [candidate for candidate in prices if candidate != price]
    return choices[index % len(choices)]


def _time(index: int) -> str:
    return ["18:00", "18:30", "19:00", "19:30", "20:00", "20:30"][index % 6]


def _day(index: int) -> str:
    return ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][index % 7]


def _people(index: int) -> int:
    return [1, 2, 3, 4, 5, 6, 8][index % 7]


def _user(category: str, split: str, index: int, record: dict[str, Any] | None, ref: str | None = None) -> str:
    name = _field(record or {}, "name", "that restaurant")
    food = _field(record or {}, "food", "restaurant")
    area = _field(record or {}, "area", "centre")
    price = _field(record or {}, "pricerange", "moderate")
    price_word = _word(PRICE_WORDS, price, index)
    area_word = _word(AREA_WORDS, area, index)
    filler = _filler(split, index)
    challenge = split == "challenge"
    restaurant_phrase = with_article(f"{price_word} {food} restaurant")
    place_phrase = with_article(f"{price_word} {food} place")
    price_place_phrase = with_article(f"{price_word} place")
    other_price = _other_price(price, index)
    other_area = _other_area(area, index)
    other_price_word = _word(PRICE_WORDS, other_price, index)
    other_area_word = _word(AREA_WORDS, other_area, index)
    other_place_phrase = with_article(f"{other_price_word} {food} place")
    price_phrase = with_article(f"{price_word} price")
    other_price_phrase = with_article(f"{other_price_word} price")
    templates = {
        "greeting": [f"hello {filler}", f"hi there {filler}", f"good afternoon {filler}", f"hiya need restaurant help {filler}"],
        "thanks": [f"thanks {filler}", f"thank you for the help {filler}", f"that's helpful {filler}", f"cheers for that {filler}"],
        "goodbye": [f"goodbye {filler}", f"bye for now {filler}", f"that's all goodbye {filler}", f"end the restaurant chat {filler}"],
        "exact_recommendation": [
            f"find {place_phrase} in the {area_word} {filler}",
            f"show me {food} food around {area_word} that is {price_word} {filler}",
            f"recommend {restaurant_phrase} in {area_word} {filler}",
        ],
        "partial_match": [
            f"need {other_place_phrase} in the {other_area_word} {filler}",
            f"anything exact for {food} in {other_area_word} at {other_price_phrase} {filler}",
        ],
        "no_exact_match": [
            f"can you find exactly {with_article(f'{other_price_word} {food} restaurant')} in the {other_area_word} {filler}",
            f"I want an exact match for {food} in the {other_area_word} {filler}",
        ],
        "no_result": [
            f"find a moon cafe with violin service {filler}",
            f"show me a restaurant with hovercraft parking {filler}",
            f"look for a midnight underwater restaurant {filler}",
        ],
        "missing_food": [f"I need somewhere to eat in the {area_word} {filler}", f"find {price_place_phrase} in {area_word} {filler}"],
        "missing_area": [f"find {restaurant_phrase} {filler}", f"I fancy {food} at {price_phrase} {filler}"],
        "missing_price": [f"find {food} food in the {area_word} {filler}", f"show me {food} places around {area_word} {filler}"],
        "missing_multiple_search": [f"I need a restaurant {filler}", f"can you suggest somewhere to eat {filler}"],
        "list_results": [f"list matching {food} restaurants in the {area_word} {filler}", f"show several {price_word} restaurants {area_word} {filler}"],
        "alternative_suggestions": [f"show me another option like {name} {filler}", f"any alternatives to {name} {filler}"],
        "address_info": [f"what is the address for {name} {filler}", f"where is {name} located {filler}"],
        "phone_postcode_info": [f"what phone and postcode do you have for {name} {filler}", f"give me contact details for {name} {filler}"],
        "cuisine_help": [f"what cuisines can I search for {filler}", f"which food categories are loaded {filler}"],
        "booking_missing_day": [f"book {name} at {_time(index)} for {_people_text(index)} {filler}", f"reserve {name} for {_people_text(index)} at {_time(index)} {filler}"],
        "booking_missing_time": [f"book {name} on {_day(index)} for {_people_text(index)} {filler}", f"reserve {name} {_day(index)} for {_people_text(index)} {filler}"],
        "booking_missing_people": [f"book {name} on {_day(index)} at {_time(index)} {filler}", f"reserve {name} {_day(index)} {_time(index)} {filler}"],
        "booking_missing_multiple": [f"book {name} {filler}", f"I want a booking at {name} {filler}"],
        "booking_confirmation": [f"book {name} on {_day(index)} at {_time(index)} for {_people_text(index)} {filler}", f"make booking {ref} for {name} {filler}"],
        "booking_reschedule": [f"move booking {ref} to {_day(index + 1)} at {_time(index + 1)} {filler}", f"change {ref} for {name} to {_day(index + 1)} {_time(index + 1)} {filler}"],
        "booking_cancellation": [f"cancel booking {ref} {filler}", f"please cancel {ref} for {name} {filler}"],
        "booking_info": [f"what is booking {ref} {filler}", f"remind me about {ref} {filler}"],
        "booking_list": [f"show my booking records {filler}", f"list the current booking records {filler}"],
        "unsupported_hotel": [f"find me a hotel {filler}", f"book a hotel room {filler}"],
        "unsupported_train": [f"what are the train times {filler}", f"find a train to London {filler}"],
        "unsupported_taxi": [f"book a taxi for me {filler}", f"can you call a cab {filler}"],
        "unsupported_payment": [f"can I pay by card through this chat {filler}", f"take my payment for {name} {filler}"],
        "unsupported_review": [f"what are the reviews for {name} {filler}", f"is {name} highly rated {filler}"],
        "cautious_halal": [f"is {name} halal {filler}", f"can you verify halal status for {name} {filler}"],
        "cautious_allergy": [f"is {name} safe for allergies {filler}", f"can you confirm allergy safety at {name} {filler}"],
        "cautious_live_availability": [f"is there a live table available at {name} {_time(index)} {filler}", f"check live availability for {name} {filler}"],
    }
    options = templates[category]
    text = options[index % len(options)]
    if challenge:
        text = text.replace("restaurant", "resturant" if index % 2 == 0 else "restaurant")
        text = text.replace("please", "pls")
        if index % 3 == 0:
            text = text.replace("?", "")
        if index % 5 == 0:
            text = f"quick one {text}"
    return " ".join(text.split())


def _select_records(records: list[dict[str, Any]], count: int, start: int = 0) -> list[dict[str, Any]]:
    if not records:
        return []
    return [records[(start + offset) % len(records)] for offset in range(count)]


def _constraint_state(constraints: dict[str, str]) -> str:
    return _state(
        food=constraints.get("food"),
        area=constraints.get("area"),
        pricerange=constraints.get("pricerange"),
    )


def _record_matches(record: dict[str, Any], constraints: dict[str, str]) -> bool:
    for key, value in constraints.items():
        if not value:
            continue
        if _field(record, key).casefold() != str(value).casefold():
            return False
    return True


def _matching_records(records: list[dict[str, Any]], constraints: dict[str, str]) -> list[dict[str, Any]]:
    return [record for record in records if _record_matches(record, constraints)]


def _rotate_records(records: list[dict[str, Any]], count: int, start: int) -> list[dict[str, Any]]:
    if not records:
        return []
    return [records[(start + offset) % len(records)] for offset in range(min(count, len(records)))]


def _constraint_candidates(record: dict[str, Any]) -> list[dict[str, str]]:
    food = _field(record, "food")
    area = _field(record, "area")
    price = _field(record, "pricerange")
    return [
        {key: value for key, value in candidate.items() if value}
        for candidate in [
            {"food": food, "area": area, "pricerange": price},
            {"food": food, "area": area},
            {"food": food, "pricerange": price},
            {"area": area, "pricerange": price},
            {"food": food},
            {"area": area},
            {"pricerange": price},
        ]
        if any(candidate.values())
    ]


def _matching_choice(
    records: list[dict[str, Any]],
    record: dict[str, Any],
    *,
    index: int,
    minimum: int,
    exclude_selected: bool = False,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    excluded_id = restaurant_id(record) if exclude_selected else ""
    for constraints in _constraint_candidates(record):
        matches = [
            candidate
            for candidate in _matching_records(records, constraints)
            if not excluded_id or restaurant_id(candidate) != excluded_id
        ]
        if len(matches) >= minimum:
            return constraints, _rotate_records(matches, 3, index)
    return {}, []


def _available_records(
    records: list[dict[str, Any]],
    *,
    index: int,
    exclude_record: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    excluded_id = restaurant_id(exclude_record) if exclude_record else ""
    pool = [record for record in records if not excluded_id or restaurant_id(record) != excluded_id]
    return _rotate_records(pool or records, 3, index)


def _row_from_parts(
    *,
    split: str,
    category: str,
    intent: str,
    user: str,
    output: str,
    evidence_records: list[dict[str, Any]] | None = None,
    state: str = "",
    missing_slots: list[str] | None = None,
    known_restaurants: list[dict[str, Any]] | None = None,
) -> Example:
    evidence_records = evidence_records or []
    item = {
        "instruction": INSTRUCTION,
        "input": build_response_input(
            intent=intent,
            user=user,
            state=state,
            evidence_records=evidence_records,
            missing_slots=missing_slots,
        ),
        "output": output,
    }
    validation = validate_generated_response(
        output,
        evidence_records=evidence_records,
        known_restaurant_records=known_restaurants or evidence_records,
    )
    output_refs = set(BOOKING_REF_RE.findall(output))
    input_refs = set(BOOKING_REF_RE.findall(item["input"]))
    missing_refs = sorted(output_refs - input_refs)
    return Example(
        split=split,
        category=category,
        intent=intent,
        row=item,
        restaurant_ids=[restaurant_id(record) for record in evidence_records],
        evidence_present=bool(evidence_records),
        booking_state_present="booking_reference=" in state or category.startswith("booking"),
        safety_failure=None if validation.ok else validation.reason,
        booking_reference_failure=", ".join(missing_refs) if missing_refs else None,
    )


def make_example(
    *,
    split: str,
    category: str,
    index: int,
    records: list[dict[str, Any]],
    all_known: list[dict[str, Any]],
    rng: random.Random,
) -> Example:
    intent = CATEGORY_INTENTS[category]
    record = records[index % len(records)] if records else {}
    second = records[(index + 1) % len(records)] if records else record
    third = records[(index + 2) % len(records)] if records else record
    ref = _booking_reference(rng)
    user = _user(category, split, index, record, ref)
    name = _field(record, "name", "the selected restaurant")
    day = _day(index)
    time = _time(index)
    people = _people(index)

    if category == "greeting":
        return _row_from_parts(split=split, category=category, intent=intent, user=user, output="Hi - I can help find restaurants in the loaded MultiWOZ data and create booking records.", known_restaurants=all_known)
    if category == "thanks":
        return _row_from_parts(split=split, category=category, intent=intent, user=user, output="You're welcome. I can also help view, change or cancel the current booking record.", known_restaurants=all_known)
    if category == "goodbye":
        return _row_from_parts(split=split, category=category, intent=intent, user=user, output="Goodbye. Come back any time you want to search restaurants or manage a booking record.", known_restaurants=all_known)
    if category == "exact_recommendation":
        output = f"I found {_summary(record)}, which matches your request for {_constraint_phrase(record)}."
        return _row_from_parts(split=split, category=category, intent=intent, user=user, state=_state(food=_field(record, "food"), area=_field(record, "area"), pricerange=_field(record, "pricerange")), evidence_records=[record], output=output, known_restaurants=all_known)
    if category == "partial_match":
        requested_area = _other_area(_field(record, "area"), index)
        requested_price = _other_price(_field(record, "pricerange"), index)
        output = f"I could not find an exact match, but the closest option I have is {_summary(record)}. It may not match every requested preference."
        return _row_from_parts(split=split, category=category, intent=intent, user=user, state=_state(food=_field(record, "food"), area=requested_area, pricerange=requested_price), evidence_records=[record], output=output, known_restaurants=all_known)
    if category == "no_exact_match":
        requested_area = _other_area(_field(record, "area"), index)
        requested = restaurant_request_phrase(food=_field(record, "food"), area=requested_area)
        output = f"I do not have an exact match for {requested}, but {_field(record, 'name')} is the supplied closest option."
        return _row_from_parts(split=split, category=category, intent=intent, user=user, state=_state(food=_field(record, "food"), area=requested_area), evidence_records=[record], output=output, known_restaurants=all_known)
    if category == "no_result":
        output = "I could not find a matching restaurant record in the loaded data. Try another food type, area or price range."
        return _row_from_parts(split=split, category=category, intent=intent, user=user, state=_state(food="unsupported request"), output=output, known_restaurants=all_known)
    if category == "missing_food":
        return _row_from_parts(split=split, category=category, intent=intent, user=user, state=_state(area=_field(record, "area"), pricerange=_field(record, "pricerange")), missing_slots=["food"], output="Sure - what kind of food would you like me to search for?", known_restaurants=all_known)
    if category == "missing_area":
        return _row_from_parts(split=split, category=category, intent=intent, user=user, state=_state(food=_field(record, "food"), pricerange=_field(record, "pricerange")), missing_slots=["area"], output="Sure - which area would you like me to search in?", known_restaurants=all_known)
    if category == "missing_price":
        return _row_from_parts(split=split, category=category, intent=intent, user=user, state=_state(food=_field(record, "food"), area=_field(record, "area")), missing_slots=["pricerange"], output="Sure - what price range would you prefer?", known_restaurants=all_known)
    if category == "missing_multiple_search":
        return _row_from_parts(split=split, category=category, intent=intent, user=user, missing_slots=["food", "area", "pricerange"], output="Please tell me your preferred food type, area and price range.", known_restaurants=all_known)
    if category == "list_results":
        constraints, chosen = _matching_choice(records, record, index=index, minimum=2)
        if chosen:
            output = "Matching restaurants: " + " ".join(f"{pos}. {_summary(item)}." for pos, item in enumerate(chosen, start=1))
            state = _constraint_state(constraints)
        else:
            chosen = _available_records(records, index=index)
            output = "Other available records: " + " ".join(f"{pos}. {_summary(item)}." for pos, item in enumerate(chosen, start=1)) + " These may not match every preference."
            state = _state(food=_field(record, "food"), area=_field(record, "area"), pricerange=_field(record, "pricerange"))
        return _row_from_parts(split=split, category=category, intent=intent, user=user, state=state, evidence_records=chosen, output=output, known_restaurants=all_known)
    if category == "alternative_suggestions":
        constraints, chosen = _matching_choice(records, record, index=index + 1, minimum=1, exclude_selected=True)
        if chosen:
            output = "Other matching options: " + " ".join(f"{pos}. {_summary(item)}." for pos, item in enumerate(chosen, start=1))
            state = _constraint_state(constraints)
        else:
            chosen = _available_records(records, index=index + 1, exclude_record=record)
            output = "Other available records: " + " ".join(f"{pos}. {_summary(item)}." for pos, item in enumerate(chosen, start=1)) + " These may not match every preference."
            state = _state(food=_field(record, "food"), area=_field(record, "area"), pricerange=_field(record, "pricerange"))
        return _row_from_parts(split=split, category=category, intent=intent, user=user, state=state, evidence_records=chosen, output=output, known_restaurants=all_known)
    if category == "address_info":
        output = f"{name} is in the {_field(record, 'area')} area. Address: {_field(record, 'address')}. Postcode: {_field(record, 'postcode')}."
        return _row_from_parts(split=split, category=category, intent=intent, user=user, evidence_records=[record], output=output, known_restaurants=all_known)
    if category == "phone_postcode_info":
        output = f"For {name}, the postcode is {_field(record, 'postcode')} and the phone number is {_field(record, 'phone')}."
        return _row_from_parts(split=split, category=category, intent=intent, user=user, evidence_records=[record], output=output, known_restaurants=all_known)
    if category == "cuisine_help":
        cuisines = sorted({_field(item, "food") for item in all_known if _field(item, "food")})
        display = ", ".join(cuisine.title() for cuisine in cuisines[index % max(len(cuisines), 1) :][:8])
        if not display:
            display = ", ".join(cuisine.title() for cuisine in cuisines[:8])
        output = f"Available cuisine categories include: {display}."
        return _row_from_parts(split=split, category=category, intent=intent, user=user, state=_state(supported_cuisines=display), output=output, known_restaurants=all_known)
    if category == "booking_missing_day":
        output = f"Great, I can create a booking record for {name}. To finish it, I still need the day."
        return _row_from_parts(split=split, category=category, intent=intent, user=user, state=_state(restaurant=name, time=time, people=people), evidence_records=[record], missing_slots=["day"], output=output, known_restaurants=all_known)
    if category == "booking_missing_time":
        output = f"Great, I can create a booking record for {name}. To finish it, I still need the time."
        return _row_from_parts(split=split, category=category, intent=intent, user=user, state=_state(restaurant=name, day=day, people=people), evidence_records=[record], missing_slots=["time"], output=output, known_restaurants=all_known)
    if category == "booking_missing_people":
        output = f"Great, I can create a booking record for {name}. To finish it, I still need the number of people."
        return _row_from_parts(split=split, category=category, intent=intent, user=user, state=_state(restaurant=name, day=day, time=time), evidence_records=[record], missing_slots=["people"], output=output, known_restaurants=all_known)
    if category == "booking_missing_multiple":
        output = f"Great, I can create a booking record for {name}. To finish it, I still need the day, time and number of people."
        return _row_from_parts(split=split, category=category, intent=intent, user=user, state=_state(restaurant=name), evidence_records=[record], missing_slots=["day", "time", "people"], output=output, known_restaurants=all_known)
    if category == "booking_confirmation":
        state = _state(booking_reference=ref, restaurant=name, day=day, time=time, people=people)
        output = f"Great, I have created booking record {ref} for {name} on {day} at {time} for {people_text(people)}."
        return _row_from_parts(split=split, category=category, intent=intent, user=user, state=state, evidence_records=[record], output=output, known_restaurants=all_known)
    if category == "booking_reschedule":
        new_day = _day(index + 1)
        new_time = _time(index + 1)
        state = _state(booking_reference=ref, restaurant=name, day=new_day, time=new_time, people=people)
        output = f"Done, I have updated booking {ref} for {name} to {new_day} at {new_time} for {people_text(people)}."
        return _row_from_parts(split=split, category=category, intent=intent, user=user, state=state, evidence_records=[record], output=output, known_restaurants=all_known)
    if category == "booking_cancellation":
        state = _state(booking_reference=ref, restaurant=name)
        output = f"Done - I have cancelled booking {ref}."
        return _row_from_parts(split=split, category=category, intent=intent, user=user, state=state, evidence_records=[record], output=output, known_restaurants=all_known)
    if category == "booking_info":
        state = _state(booking_reference=ref, restaurant=name, day=day, time=time, people=people, status="confirmed")
        output = f"Booking {ref} is confirmed for {name}: {day} at {time} for {people_text(people)}."
        return _row_from_parts(split=split, category=category, intent=intent, user=user, state=state, evidence_records=[record], output=output, known_restaurants=all_known)
    if category == "booking_list":
        ref2 = _booking_reference(rng)
        records_for_list = [record, second] if second else [record]
        state = _state(booking_reference=ref, second_booking_reference=ref2, restaurant=name)
        output = f"Current session booking records: 1. {ref}: {name} on {day} at {time} for {people_text(people)}. 2. {ref2}: {_field(second, 'name', name)} on {_day(index + 1)} at {_time(index + 1)} for {_people_text(index + 1)}."
        return _row_from_parts(split=split, category=category, intent=intent, user=user, state=state, evidence_records=records_for_list, output=output, known_restaurants=all_known)
    unsupported_outputs = {
        "unsupported_hotel": "I can only help with MultiWOZ restaurant search and restaurant booking records. I cannot book hotels in this demo.",
        "unsupported_train": "I can only help with restaurant search and booking records. I cannot provide train times.",
        "unsupported_taxi": "I can only help with restaurant search and booking records. I cannot book taxis.",
        "unsupported_payment": "I cannot process payments or handle card details. I can only create local booking records for the demonstration.",
        "unsupported_review": "I do not have review or rating data in the loaded restaurant records.",
        "cautious_halal": "I cannot verify halal status from the loaded restaurant records.",
        "cautious_allergy": "I cannot confirm allergy safety. Please contact the restaurant directly.",
        "cautious_live_availability": "I cannot check live table availability, but I can create a local booking record for the demonstration.",
    }
    evidence = [record] if category in {"cautious_halal", "cautious_allergy", "cautious_live_availability"} and record else []
    state = _state(restaurant=name) if evidence else ""
    return _row_from_parts(split=split, category=category, intent=intent, user=user, state=state, evidence_records=evidence, output=unsupported_outputs[category], known_restaurants=all_known)


def _usable_restaurants(restaurants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    usable = [
        record
        for record in restaurants
        if _field(record, "name") and _field(record, "food") and _field(record, "area") and _field(record, "pricerange")
    ]
    return usable or restaurants


def split_restaurants(restaurants: list[dict[str, Any]], rng: random.Random) -> tuple[dict[str, list[dict[str, Any]]], bool, list[str]]:
    records = sorted(_usable_restaurants(restaurants), key=restaurant_id)
    rng.shuffle(records)
    warnings: list[str] = []
    if len(records) >= 6:
        eval_size = max(1, min(len(records) // 5, 24))
        challenge_size = max(1, min(len(records) // 5, 24))
        train_size = max(1, len(records) - eval_size - challenge_size)
        splits = {
            "train": records[:train_size],
            "eval": records[train_size : train_size + eval_size],
            "challenge": records[train_size + eval_size : train_size + eval_size + challenge_size],
        }
        return splits, True, warnings
    if len(records) >= 3:
        splits = {"train": records[:-2], "eval": [records[-2]], "challenge": [records[-1]]}
        warnings.append("Small restaurant source: disjoint restaurant splits were possible but have limited diversity.")
        return splits, True, warnings
    warnings.append("Too few restaurant records for disjoint restaurant-level splits; records are reused across splits.")
    return {"train": records, "eval": records, "challenge": records}, False, warnings


def _category_schedule(count: int, rng: random.Random) -> list[str]:
    categories = [CATEGORIES[index % len(CATEGORIES)] for index in range(count)]
    rng.shuffle(categories)
    return categories


def generate_split(
    *,
    split: str,
    count: int,
    records: list[dict[str, Any]],
    all_known: list[dict[str, Any]],
    rng: random.Random,
) -> list[Example]:
    if count < 0:
        raise ValueError("Split counts must be non-negative.")
    if not records:
        raise ValueError(f"No restaurant records available for {split} split.")
    examples = []
    category_counts: Counter[str] = Counter()
    seen_users: set[str] = set()
    for category in _category_schedule(count, rng):
        category_index = category_counts[category]
        category_counts[category] += 1
        attempt = 0
        while True:
            example = make_example(
                split=split,
                category=category,
                index=category_index + attempt * 997,
                records=records,
                all_known=all_known,
                rng=rng,
            )
            user_norm = extract_user_norm(example.row["input"])
            if user_norm not in seen_users:
                seen_users.add(user_norm)
                examples.append(example)
                break
            attempt += 1
            if attempt > 20:
                raise RuntimeError(f"Could not create a unique user message for {split}/{category}.")
    return examples


def extract_user(input_text: str) -> str:
    for line in input_text.splitlines():
        if line.startswith("User:"):
            return line.split(":", 1)[1].strip()
    return ""


def extract_user_norm(input_text: str) -> str:
    return normalise_text(extract_user(input_text))


def _row_key(row: dict[str, str]) -> str:
    return json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _split_overlap(first: list[Example], second: list[Example], selector) -> int:
    return len({selector(item) for item in first} & {selector(item) for item in second})


def _duplicates(values: list[str]) -> int:
    counts = Counter(values)
    return sum(count - 1 for count in counts.values() if count > 1)


def _restaurant_ids(examples: Iterable[Example]) -> set[str]:
    ids: set[str] = set()
    for example in examples:
        ids.update(example.restaurant_ids)
    return ids


def build_report(
    *,
    examples: dict[str, list[Example]],
    seed: int,
    requested_counts: dict[str, int],
    restaurants: list[dict[str, Any]],
    restaurant_splits: dict[str, list[dict[str, Any]]],
    restaurant_split_separation: bool,
    source_path: Path,
    sample_data: bool,
    warnings: list[str],
) -> dict[str, Any]:
    split_reports: dict[str, Any] = {}
    for split, split_examples in examples.items():
        rows = [example.row for example in split_examples]
        input_lengths = [len(row["input"].split()) for row in rows]
        output_lengths = [len(row["output"].split()) for row in rows]
        split_reports[split] = {
            "actual_count": len(rows),
            "intent_counts": dict(Counter(example.intent for example in split_examples)),
            "response_category_counts": dict(Counter(example.category for example in split_examples)),
            "unique_restaurant_count": len(_restaurant_ids(split_examples)),
            "restaurant_counts": dict(Counter(rid for example in split_examples for rid in example.restaurant_ids)),
            "evidence_present_count": sum(int(example.evidence_present) for example in split_examples),
            "booking_state_present_count": sum(int(example.booking_state_present) for example in split_examples),
            "unique_normalised_user_messages": len({extract_user_norm(row["input"]) for row in rows}),
            "unique_inputs": len({row["input"] for row in rows}),
            "duplicate_rows": _duplicates([_row_key(row) for row in rows]),
            "duplicate_input_output_pairs": _duplicates([row["input"] + "\n" + row["output"] for row in rows]),
            "average_input_word_count": round(statistics.mean(input_lengths), 2) if input_lengths else 0.0,
            "average_output_word_count": round(statistics.mean(output_lengths), 2) if output_lengths else 0.0,
            "minimum_input_word_count": min(input_lengths) if input_lengths else 0,
            "maximum_input_word_count": max(input_lengths) if input_lengths else 0,
            "minimum_output_word_count": min(output_lengths) if output_lengths else 0,
            "maximum_output_word_count": max(output_lengths) if output_lengths else 0,
        }

    train = examples["train"]
    eval_examples = examples["eval"]
    challenge = examples["challenge"]
    overlaps = {
        "train_eval_input_overlap": _split_overlap(train, eval_examples, lambda item: item.row["input"]),
        "train_challenge_input_overlap": _split_overlap(train, challenge, lambda item: item.row["input"]),
        "eval_challenge_input_overlap": _split_overlap(eval_examples, challenge, lambda item: item.row["input"]),
        "train_eval_user_overlap": _split_overlap(train, eval_examples, lambda item: extract_user_norm(item.row["input"])),
        "train_challenge_user_overlap": _split_overlap(train, challenge, lambda item: extract_user_norm(item.row["input"])),
        "eval_challenge_user_overlap": _split_overlap(eval_examples, challenge, lambda item: extract_user_norm(item.row["input"])),
        "train_eval_row_overlap": _split_overlap(train, eval_examples, lambda item: _row_key(item.row)),
        "train_challenge_row_overlap": _split_overlap(train, challenge, lambda item: _row_key(item.row)),
        "eval_challenge_row_overlap": _split_overlap(eval_examples, challenge, lambda item: _row_key(item.row)),
        "train_eval_restaurant_overlap": len(_restaurant_ids(train) & _restaurant_ids(eval_examples)),
        "train_challenge_restaurant_overlap": len(_restaurant_ids(train) & _restaurant_ids(challenge)),
        "eval_challenge_restaurant_overlap": len(_restaurant_ids(eval_examples) & _restaurant_ids(challenge)),
    }
    safety_failures = [
        {"split": example.split, "category": example.category, "reason": example.safety_failure, "output": example.row["output"]}
        for split_examples in examples.values()
        for example in split_examples
        if example.safety_failure
    ]
    booking_failures = [
        {"split": example.split, "category": example.category, "missing_references": example.booking_reference_failure}
        for split_examples in examples.values()
        for example in split_examples
        if example.booking_reference_failure
    ]
    limitations = list(warnings)
    if sample_data:
        limitations.append("Sample data is useful for smoke validation but has limited restaurant diversity.")
    if not restaurant_split_separation:
        limitations.append("Restaurant-level disjoint splitting was not achieved because too few source records were available.")
    limitations.append("Responses are synthetic deterministic templates, so they may favour the system's target response style.")
    limitations.append("Automatic safety checks do not replace human evaluation of trained model outputs.")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "requested_counts": requested_counts,
        "actual_counts": {split: len(items) for split, items in examples.items()},
        "source_data_path": repo_relative(source_path),
        "sample_data_used": sample_data,
        "source_restaurant_record_count": len(restaurants),
        "restaurant_split_sizes": {split: len(items) for split, items in restaurant_splits.items()},
        "restaurant_level_disjoint_split_achieved": restaurant_split_separation,
        "splits": split_reports,
        "overlap_counts": overlaps,
        "safety_validation_failure_count": len(safety_failures),
        "safety_validation_failures": safety_failures,
        "booking_reference_grounding_failure_count": len(booking_failures),
        "booking_reference_grounding_failures": booking_failures,
        "limitations": limitations,
    }


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def build_dataset(
    restaurants: list[dict[str, Any]],
    *,
    train_count: int = DEFAULT_TRAIN_COUNT,
    eval_count: int = DEFAULT_EVAL_COUNT,
    challenge_count: int = DEFAULT_CHALLENGE_COUNT,
    seed: int = DEFAULT_SEED,
    sample_data: bool = False,
    source_path: Path | None = None,
) -> BuildResult:
    rng = random.Random(seed)
    restaurants = _usable_restaurants(restaurants)
    if not restaurants:
        raise ValueError("No restaurant records available for response generation.")
    restaurant_splits, split_separation, warnings = split_restaurants(restaurants, rng)
    examples = {
        "train": generate_split(split="train", count=train_count, records=restaurant_splits["train"], all_known=restaurants, rng=rng),
        "eval": generate_split(split="eval", count=eval_count, records=restaurant_splits["eval"], all_known=restaurants, rng=rng),
        "challenge": generate_split(split="challenge", count=challenge_count, records=restaurant_splits["challenge"], all_known=restaurants, rng=rng),
    }
    report = build_report(
        examples=examples,
        seed=seed,
        requested_counts={"train": train_count, "eval": eval_count, "challenge": challenge_count},
        restaurants=restaurants,
        restaurant_splits=restaurant_splits,
        restaurant_split_separation=split_separation,
        source_path=source_path or Path("unknown"),
        sample_data=sample_data,
        warnings=warnings,
    )
    if report["safety_validation_failure_count"] or report["booking_reference_grounding_failure_count"]:
        raise ValueError("Unsafe response dataset examples were generated; see report details.")
    return BuildResult(
        train_rows=[example.row for example in examples["train"]],
        eval_rows=[example.row for example in examples["eval"]],
        challenge_rows=[example.row for example in examples["challenge"]],
        examples=examples,
        report=report,
        warnings=warnings,
    )


def build_rows(restaurants: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Compatibility helper used by older local scripts."""

    return build_dataset(restaurants, train_count=24, eval_count=6, challenge_count=3).train_rows


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for item in rows:
            file.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_report_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def write_report_md(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Response Generation Dataset Report",
        "",
        f"- Seed: `{report['seed']}`",
        f"- Source data: `{report['source_data_path']}`",
        f"- Sample data used: `{report['sample_data_used']}`",
        f"- Source restaurant records: `{report['source_restaurant_record_count']}`",
        f"- Restaurant-level disjoint split achieved: `{report['restaurant_level_disjoint_split_achieved']}`",
        "",
        "## Split Counts",
        "",
        "| Split | Requested | Actual | Unique Restaurants | Unique Users | Unique Inputs |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ("train", "eval", "challenge"):
        split_report = report["splits"][split]
        lines.append(
            f"| {split} | {report['requested_counts'][split]} | {report['actual_counts'][split]} | "
            f"{split_report['unique_restaurant_count']} | {split_report['unique_normalised_user_messages']} | "
            f"{split_report['unique_inputs']} |"
        )
    lines.extend(["", "## Intent Counts", ""])
    for split in ("train", "eval", "challenge"):
        lines.append(f"### {split}")
        for key, value in sorted(report["splits"][split]["intent_counts"].items()):
            lines.append(f"- `{key}`: {value}")
        lines.append("")
    lines.extend(["## Response Category Counts", ""])
    for split in ("train", "eval", "challenge"):
        lines.append(f"### {split}")
        for key, value in sorted(report["splits"][split]["response_category_counts"].items()):
            lines.append(f"- `{key}`: {value}")
        lines.append("")
    lines.extend(
        [
            "## Leakage And Safety",
            "",
            f"- Train/eval input overlap: `{report['overlap_counts']['train_eval_input_overlap']}`",
            f"- Train/challenge input overlap: `{report['overlap_counts']['train_challenge_input_overlap']}`",
            f"- Eval/challenge input overlap: `{report['overlap_counts']['eval_challenge_input_overlap']}`",
            f"- Train/eval user-message overlap: `{report['overlap_counts']['train_eval_user_overlap']}`",
            f"- Train/challenge user-message overlap: `{report['overlap_counts']['train_challenge_user_overlap']}`",
            f"- Eval/challenge user-message overlap: `{report['overlap_counts']['eval_challenge_user_overlap']}`",
            f"- Safety-validation failures: `{report['safety_validation_failure_count']}`",
            f"- Booking-reference grounding failures: `{report['booking_reference_grounding_failure_count']}`",
            "",
            "## Limitations",
            "",
        ]
    )
    for limitation in report["limitations"]:
        lines.append(f"- {limitation}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_source_restaurants(*, sample_data: bool) -> tuple[list[dict[str, Any]], Path, bool, list[str]]:
    settings = get_settings()
    warnings: list[str] = []
    if sample_data:
        return load_restaurants(settings, use_sample=True), settings.sample_data_path, True, warnings
    if settings.processed_restaurant_path.exists():
        return load_restaurants(settings, use_sample=False), settings.processed_restaurant_path, False, warnings
    warnings.append("Processed restaurant data was unavailable; fell back to bundled sample restaurants.")
    return load_restaurants(settings, use_sample=True), settings.sample_data_path, True, warnings


def print_summary(result: BuildResult, train_output: Path, eval_output: Path, challenge_output: Path) -> None:
    report = result.report
    print(f"Generated training examples: {report['actual_counts']['train']}")
    print(f"Generated evaluation examples: {report['actual_counts']['eval']}")
    print(f"Generated challenge examples: {report['actual_counts']['challenge']}")
    for split in ("train", "eval", "challenge"):
        print(f"Unique restaurants in {split}: {report['splits'][split]['unique_restaurant_count']}")
    print(f"Restaurant-level split separation possible: {report['restaurant_level_disjoint_split_achieved']}")
    duplicate_counts = {
        split: {
            "duplicate_rows": report["splits"][split]["duplicate_rows"],
            "duplicate_input_output_pairs": report["splits"][split]["duplicate_input_output_pairs"],
        }
        for split in ("train", "eval", "challenge")
    }
    print(f"Duplicate counts: {json.dumps(duplicate_counts, sort_keys=True)}")
    print(f"Overlap counts: {json.dumps(report['overlap_counts'], sort_keys=True)}")
    print(f"Safety-validation failures: {report['safety_validation_failure_count']}")
    print(f"Booking-reference grounding failures: {report['booking_reference_grounding_failure_count']}")
    print(f"Training output: {repo_relative(train_output)}")
    print(f"Evaluation output: {repo_relative(eval_output)}")
    print(f"Challenge output: {repo_relative(challenge_output)}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    restaurants, source_path, sample_used, warnings = load_source_restaurants(sample_data=args.sample_data)
    result = build_dataset(
        restaurants,
        train_count=args.train_count,
        eval_count=args.eval_count,
        challenge_count=args.challenge_count,
        seed=args.seed,
        sample_data=sample_used,
        source_path=source_path,
    )
    if warnings:
        result.report["limitations"] = warnings + result.report["limitations"]
    write_jsonl(args.train_output, result.train_rows)
    write_jsonl(args.eval_output, result.eval_rows)
    write_jsonl(args.challenge_output, result.challenge_rows)
    write_report_json(args.report_json, result.report)
    write_report_md(args.report_md, result.report)
    print_summary(result, args.train_output, args.eval_output, args.challenge_output)
    print(f"Dataset report JSON: {repo_relative(args.report_json)}")
    print(f"Dataset report Markdown: {repo_relative(args.report_md)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
