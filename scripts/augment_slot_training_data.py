"""Build balanced deterministic slot-extraction instruction data."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from restaurant_assistant.slot_extraction import ALLOWED_INTENTS, ALLOWED_SLOT_KEYS

try:
    from .check_data_leakage import load_records, normalize_text
except ImportError:
    from check_data_leakage import load_records, normalize_text


DEFAULT_INPUT = ROOT / "data" / "training" / "slot_instruction_examples.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "training" / "slot_instruction_examples_augmented.jsonl"
DEFAULT_EVAL = ROOT / "data" / "evaluation" / "slot_eval_cases.jsonl"

REQUIRED_INTENTS = (
    "search",
    "list",
    "book",
    "reschedule",
    "cancel",
    "correct",
    "booking_info",
    "booking_list",
    "restaurant_info",
    "filter_info",
    "cuisine_help",
    "dish_preference",
    "distance_info",
    "table_view",
    "greeting",
    "thanks",
    "unsupported",
)

FOODS = ("italian", "chinese", "thai", "indian", "lebanese", "british", "japanese", "vegetarian")
AREAS = ("centre", "north", "south", "east", "west")
PRICES = ("cheap", "moderate", "expensive")
PRICE_PHRASES = {"cheap": "cheap", "moderate": "reasonable", "expensive": "upmarket"}
DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
TIMES = ("12:30", "17:45", "18:00", "18:30", "19:00", "19:30", "20:00", "20:30")
PEOPLE = (2, 3, 4, 5, 6, 8)
REFERENCES = (
    "BK-7GHT92",
    "SIM-A1B2C3",
    "BK-Q2W3E4",
    "BK-9Z8Y7X",
    "SIM-C4D5E6",
    "BK-M6N7P8",
    "BK-R4S5T6",
    "SIM-U7V8W9",
)

CUISINE_GROUPS = (
    ("Middle Eastern", ["lebanese", "turkish", "mediterranean"]),
    ("South Asian", ["indian"]),
    ("East Asian", ["chinese", "cantonese", "japanese", "korean", "asian oriental"]),
    ("Southeast Asian", ["thai", "vietnamese", "asian oriental"]),
    ("North African", ["african"]),
    ("West African", ["african"]),
)

DISHES = (
    ("curry", ["indian", "thai"]),
    ("noodles", ["chinese", "thai", "vietnamese"]),
    ("pizza", ["italian"]),
    ("sushi", ["japanese"]),
    ("mezze", ["lebanese", "mediterranean", "turkish"]),
    ("tapas", ["spanish"]),
)


def compact_output(intent: str, slots: dict[str, Any]) -> str:
    """Return the exact compact target format used by training."""
    return json.dumps({"intent": intent, "slots": slots}, ensure_ascii=True, separators=(",", ":"))


def example(text: str, intent: str, slots: dict[str, Any] | None = None) -> dict[str, str]:
    return {"text": " ".join(text.split()), "output": compact_output(intent, slots or {})}


def _empty_intent_candidates(intent: str, phrases: tuple[str, ...]) -> Iterator[dict[str, str]]:
    suffixes = (
        "",
        " please",
        " pls",
        " if possible",
        " for me",
        " right now",
        " in this chat",
        " when you can",
        " today",
        " on here",
    )
    for phrase in phrases:
        for suffix in suffixes:
            yield example(f"{phrase}{suffix}", intent)


def _search_candidates() -> Iterator[dict[str, str]]:
    templates = (
        "Find a {price} {food} resturant in the {area}",
        "I need somewhere {price} for {food} near the {area}",
        "Could you look for {food} food in the {area}, {price} pls",
        "Show me a {price} place serving {food} around the {area}",
        "Somewhere nice for {food} near the {area}, {price} would be ideal",
        "Any {price} {food} options on the {area} side of town?",
    )
    for index in range(120):
        food = FOODS[index % len(FOODS)]
        area = AREAS[(index // len(FOODS)) % len(AREAS)]
        price = PRICES[(index // (len(FOODS) * len(AREAS))) % len(PRICES)]
        template = templates[index % len(templates)]
        yield example(
            template.format(food=food, area=area, price=PRICE_PHRASES[price]),
            "search",
            {"food": food, "area": area, "pricerange": price},
        )
        if index < len(CUISINE_GROUPS) * len(AREAS):
            group, candidates = CUISINE_GROUPS[index % len(CUISINE_GROUPS)]
            group_area = AREAS[(index // len(CUISINE_GROUPS)) % len(AREAS)]
            yield example(
                f"Find {group} food near the {group_area} please",
                "search",
                {"cuisine_group": group, "food_candidates": candidates, "area": group_area},
            )


def _list_candidates() -> Iterator[dict[str, str]]:
    templates = (
        "List {food} restaurants in the {area}",
        "Show every {food} place around the {area}",
        "What {food} options are there on the {area} side?",
        "Can I see all {food} resturants near the {area} pls",
        "Give me the full list of {food} places in the {area}",
        "Which {food} restaurants do you have around {area} town?",
    )
    for index in range(100):
        food = FOODS[index % len(FOODS)]
        area = AREAS[(index // len(FOODS)) % len(AREAS)]
        yield example(
            templates[index % len(templates)].format(food=food, area=area),
            "list",
            {"food": food, "area": area},
        )


def _book_candidates() -> Iterator[dict[str, str]]:
    timed_templates = (
        "Book a table for {people} on {day} at {spoken_time}",
        "Can you reserve it {day}, {spoken_time}, party of {people}",
        "Table for {people} at {spoken_time} on {day} pls",
        "Please book for {people} people this {day} at {spoken_time}",
        "Need a tbale on {day} at {spoken_time} for {people}",
    )
    food_templates = (
        "Book a {price} {food} place for a party of {people}",
        "Reserve somewhere {price} serving {food} for {people} people",
        "Can you book {food}, {price}, for {people} pls",
        "I need a table for {people} at a {price} {food} resturant",
        "Find and book {food} food for {people}, budget is {price}",
    )
    for index in range(80):
        people = PEOPLE[index % len(PEOPLE)]
        if index % 2 == 0:
            day = DAYS[(index // 2) % len(DAYS)]
            time = TIMES[(index // (2 * len(DAYS))) % len(TIMES)]
            spoken_time = time.lstrip("0")
            yield example(
                timed_templates[index % len(timed_templates)].format(
                    people=people, day=day, spoken_time=spoken_time
                ),
                "book",
                {"day": day, "time": time, "people": people},
            )
        else:
            food = FOODS[(index // 2) % len(FOODS)]
            price = PRICES[(index // (2 * len(FOODS))) % len(PRICES)]
            yield example(
                food_templates[index % len(food_templates)].format(
                    people=people, food=food, price=PRICE_PHRASES[price]
                ),
                "book",
                {"food": food, "pricerange": price, "people": people},
            )
    for people in PEOPLE:
        yield example(
            f"Book it tmrw at 6pm for a party of {people}",
            "book",
            {"relative_day": "tomorrow", "time": "18:00", "people": people},
        )


def _reschedule_candidates() -> Iterator[dict[str, str]]:
    templates = (
        "Move booking {reference} to {day} at {spoken_time}",
        "Reschedule {reference} for {day}, {spoken_time} pls",
        "Change {reference} to {spoken_time} on {day}",
        "Can you shift booking {reference} to {day} at {spoken_time}?",
        "Update {reference}: {day}, {spoken_time}",
        "I need {reference} moved to {day} at {spoken_time}",
    )
    for index in range(96):
        reference = REFERENCES[index % len(REFERENCES)]
        day = DAYS[(index // len(REFERENCES)) % len(DAYS)]
        time = TIMES[(index // (len(REFERENCES) * len(DAYS))) % len(TIMES)]
        yield example(
            templates[index % len(templates)].format(
                reference=reference, day=day, spoken_time=time.lstrip("0")
            ),
            "reschedule",
            {"booking_reference": reference, "day": day, "time": time},
        )


def _reference_candidates(intent: str, phrases: tuple[str, ...]) -> Iterator[dict[str, str]]:
    for reference in REFERENCES:
        for phrase in phrases:
            yield example(
                phrase.format(reference=reference),
                intent,
                {"booking_reference": reference},
            )


def _correct_candidates() -> Iterator[dict[str, str]]:
    corrections = [
        (f"Actually make the area {area}", {"area": area}) for area in AREAS
    ] + [
        (f"Sorry, I meant {food} food", {"food": food}) for food in FOODS
    ] + [
        (f"Change that to {price}, not the previous price", {"pricerange": price})
        for price in PRICES
    ] + [
        (f"No, the booking should be on {day}", {"day": day}) for day in DAYS
    ] + [
        (f"I meant {time.lstrip('0')}, please correct the time", {"time": time})
        for time in TIMES
    ]
    prefixes = ("", "Wait, ", "Correction: ", "Sorry, ", "My mistake, ")
    for text, slots in corrections:
        for prefix in prefixes:
            yield example(f"{prefix}{text}", "correct", slots)


def _distance_candidates() -> Iterator[dict[str, str]]:
    templates = (
        "How far is it from the {area}?",
        "Is that close to the {area} side?",
        "Tell me the distance from {area} town",
        "How near is the restaurant to the {area}?",
        "Would it be a long walk from the {area}?",
        "Distance info for the {area} please",
        "Is the place anywhere near the {area}?",
        "Roughly how far from the {area} is it?",
        "Could you check how close it is to {area}?",
        "I need somewhere not far from the {area}",
        "How far away is that from {area} Cambridge?",
        "Near the {area} or miles away?",
    )
    for area in AREAS:
        for template in templates:
            yield example(template.format(area=area), "distance_info", {"area": area})


def _dish_candidates() -> Iterator[dict[str, str]]:
    templates = (
        "I fancy {dish}, what cuisines should I try?",
        "Where could I get {dish}?",
        "Suggest food types if I want {dish}",
        "I'm craving {dish} tonight",
        "Which restaurant cuisine is best for {dish}?",
        "Could you help me find {dish} pls",
        "I'd like a dish such as {dish}",
        "What places might serve {dish}?",
        "Find somewhere for {dish} around town",
        "Any ideas if I want {dish}?",
    )
    for dish, candidates in DISHES:
        for template in templates:
            yield example(
                template.format(dish=dish),
                "dish_preference",
                {"dish": dish, "food_candidates": candidates},
            )


def _unsupported_candidates() -> Iterator[dict[str, str]]:
    requests = (
        "book me a taxi",
        "find an emergency dentist",
        "reserve a hotel room",
        "check the next train",
        "where can I buy fireworks",
        "give me a refund",
        "order a takeaway",
        "take my payment",
    )
    wrappers = (
        "{request}",
        "Can you {request}?",
        "Please {request} for me",
        "I need you to {request} pls",
        "Could this assistant {request}?",
        "Quickly {request}",
        "Help me {request} today",
        "Is it possible to {request} on here?",
    )
    for wrapper in wrappers:
        for request in requests:
            yield example(wrapper.format(request=request), "unsupported")


def synthetic_candidates() -> Iterator[dict[str, str]]:
    """Yield a deterministic pool with more candidates than the target needs."""
    yield from _search_candidates()
    yield from _list_candidates()
    yield from _book_candidates()
    yield from _reschedule_candidates()
    yield from _reference_candidates(
        "cancel",
        (
            "Cancel booking {reference}",
            "Please call off {reference}",
            "I no longer need {reference}",
            "Can you cancel {reference} pls?",
            "Remove reservation {reference}",
            "Scrap my booking {reference}",
            "I'd like {reference} cancelled",
            "Cancel {reference} for me today",
        ),
    )
    yield from _correct_candidates()
    yield from _reference_candidates(
        "booking_info",
        (
            "Show details for {reference}",
            "What are the booking details for {reference}?",
            "Open reservation {reference}",
            "Tell me about booking {reference}",
            "Can I check {reference} pls?",
            "Which table is linked to {reference}?",
            "Look up my reference {reference}",
            "I need the info for {reference}",
        ),
    )
    yield from _empty_intent_candidates(
        "booking_list",
        (
            "Show my bookings",
            "List all my reservations",
            "What tables have I booked?",
            "Open my booking history",
            "Can I see my current reservations?",
            "Display every booking I have",
        ),
    )
    yield from _empty_intent_candidates(
        "restaurant_info",
        (
            "Tell me about the restaurant",
            "Show the restaurant details",
            "What do you know about that place?",
            "Give me more information about it",
            "Open the details for this resturant",
            "Can I see the restaurant information?",
        ),
    )
    yield from _empty_intent_candidates(
        "filter_info",
        (
            "What filters are active?",
            "Show my current search filters",
            "Which preferences did I set?",
            "Tell me the filters being used",
            "What is this list filtered by?",
            "Display the active criteria",
        ),
    )
    yield from _empty_intent_candidates(
        "cuisine_help",
        (
            "Help me choose a cuisine",
            "What kind of food should I try?",
            "I cannot decide on a cuisine",
            "Suggest a type of food",
            "Can you explain the cuisine choices?",
            "Give me some cuisine ideas",
        ),
    )
    yield from _dish_candidates()
    yield from _distance_candidates()
    yield from _empty_intent_candidates(
        "table_view",
        (
            "Show the results as a table",
            "Switch to table view",
            "Put those restaurants in a table",
            "Can I see a tabular list?",
            "Display the options in rows",
            "Use the table layout",
        ),
    )
    yield from _empty_intent_candidates(
        "greeting",
        (
            "Hello restaurant assistant",
            "Hi there, can we get started?",
            "Good morning assistant",
            "Good afternoon, anyone there?",
            "Hey, nice to meet you",
            "Evening, I need some help",
        ),
    )
    yield from _empty_intent_candidates(
        "thanks",
        (
            "Thanks for that",
            "Thank you for your help",
            "Cheers, that is all",
            "Much appreciated",
            "Brilliant, thanks",
            "Lovely, thank you",
        ),
    )
    yield from _unsupported_candidates()


def _canonicalize_source_record(record: dict[str, Any], *, path: Path, line_number: int) -> dict[str, str]:
    text = str(record.get("text") or record.get("input") or "").strip()
    if not text:
        raise ValueError(f"Missing text at {path}:{line_number}")
    if "output" in record:
        raw_target = record["output"]
        target = json.loads(raw_target) if isinstance(raw_target, str) else raw_target
    else:
        target = {"intent": record.get("intent"), "slots": record.get("slots", {})}
    if not isinstance(target, dict):
        raise ValueError(f"Target is not an object at {path}:{line_number}")
    return example(text, str(target.get("intent") or ""), target.get("slots") or {})


def validate_records(
    records: Iterable[dict[str, Any]],
    *,
    eval_file: Path,
) -> tuple[Counter[str], Counter[str]]:
    """Validate schema, target vocabulary, deduplication and hold-out isolation."""
    eval_texts = {
        normalize_text(str(record.get("text") or record.get("input") or ""))
        for record in load_records(eval_file)
    }
    seen: set[tuple[str, str]] = set()
    intent_counts: Counter[str] = Counter()
    slot_counts: Counter[str] = Counter()
    for line_number, record in enumerate(records, start=1):
        if not isinstance(record, dict) or "text" not in record or "output" not in record:
            raise ValueError(f"Row {line_number} must contain text and output")
        text = str(record["text"]).strip()
        output = record["output"]
        if not text or not isinstance(output, str):
            raise ValueError(f"Row {line_number} has invalid text or output")
        if not output.startswith("{") or not output.endswith("}"):
            raise ValueError(f"Row {line_number} output must retain JSON object braces")
        target = json.loads(output)
        if not isinstance(target, dict) or set(target) != {"intent", "slots"}:
            raise ValueError(f"Row {line_number} output must contain only intent and slots")
        intent = target["intent"]
        slots = target["slots"]
        if intent not in ALLOWED_INTENTS:
            raise ValueError(f"Row {line_number} contains unsupported intent {intent!r}")
        if not isinstance(slots, dict) or not set(slots).issubset(ALLOWED_SLOT_KEYS):
            raise ValueError(f"Row {line_number} contains unsupported slot keys")
        if output != compact_output(intent, slots):
            raise ValueError(f"Row {line_number} output is not compact canonical JSON")
        normalized = normalize_text(text)
        if normalized in eval_texts:
            raise ValueError(f"Hold-out text leakage detected at row {line_number}: {text!r}")
        dedupe_key = (normalized, output)
        if dedupe_key in seen:
            raise ValueError(f"Duplicate normalized text/output at row {line_number}: {text!r}")
        seen.add(dedupe_key)
        intent_counts[intent] += 1
        slot_counts.update(slots.keys())
    return intent_counts, slot_counts


def build_augmented_records(
    input_file: Path,
    eval_file: Path,
    *,
    target_per_intent: int = 55,
    minimum_additions_per_intent: int = 20,
) -> list[dict[str, str]]:
    """Canonicalize originals and add balanced, non-leaking synthetic rows."""
    eval_texts = {
        normalize_text(str(record.get("text") or record.get("input") or ""))
        for record in load_records(eval_file)
    }
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    intent_counts: Counter[str] = Counter()
    for line_number, source in enumerate(load_records(input_file), start=1):
        record = _canonicalize_source_record(source, path=input_file, line_number=line_number)
        normalized = normalize_text(record["text"])
        if normalized in eval_texts:
            raise ValueError(f"Original training text overlaps hold-out data: {record['text']!r}")
        key = (normalized, record["output"])
        if key in seen:
            continue
        seen.add(key)
        records.append(record)
        intent_counts[json.loads(record["output"])["intent"]] += 1

    required_additions = {
        intent: max(minimum_additions_per_intent, target_per_intent - intent_counts[intent])
        for intent in REQUIRED_INTENTS
    }
    additions: Counter[str] = Counter()
    for candidate in synthetic_candidates():
        target = json.loads(candidate["output"])
        intent = target["intent"]
        if additions[intent] >= required_additions[intent]:
            continue
        normalized = normalize_text(candidate["text"])
        key = (normalized, candidate["output"])
        if normalized in eval_texts or key in seen:
            continue
        seen.add(key)
        records.append(candidate)
        additions[intent] += 1

    missing = {
        intent: required_additions[intent] - additions[intent]
        for intent in REQUIRED_INTENTS
        if additions[intent] < required_additions[intent]
    }
    if missing:
        raise ValueError(f"Not enough unique deterministic candidates: {missing}")
    validate_records(records, eval_file=eval_file)
    return records


def write_augmented_file(records: Iterable[dict[str, str]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=True, separators=(",", ":")) for record in records]
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate balanced slot instruction data.")
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--target-per-intent", type=int, default=55)
    parser.add_argument("--minimum-additions-per-intent", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = build_augmented_records(
        args.input_file,
        args.eval_file,
        target_per_intent=args.target_per_intent,
        minimum_additions_per_intent=args.minimum_additions_per_intent,
    )
    if not 800 <= len(records) <= 1500:
        raise ValueError(f"Expected 800-1500 augmented rows, got {len(records)}")
    intent_counts, slot_counts = validate_records(records, eval_file=args.eval_file)
    write_augmented_file(records, args.output_file)
    print(f"Wrote {len(records)} rows to {args.output_file}")
    print("Intent distribution:")
    for intent, count in sorted(intent_counts.items()):
        print(f"  {intent}: {count}")
    print("Slot-key distribution:")
    for slot, count in sorted(slot_counts.items()):
        print(f"  {slot}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
