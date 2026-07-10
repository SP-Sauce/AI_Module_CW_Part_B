"""Build strict JSON slot/intent training data for the restaurant assistant."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from restaurant_assistant.slot_extraction import STRICT_MODEL_INTENTS, STRICT_MODEL_SLOT_KEYS

try:
    from .check_data_leakage import load_records, normalize_text
except ImportError:
    from check_data_leakage import load_records, normalize_text


DEFAULT_TRAIN_OUTPUT = ROOT / "data" / "training" / "slot_train_strict.jsonl"
DEFAULT_DEV_OUTPUT = ROOT / "data" / "training" / "slot_dev_strict.jsonl"
DEFAULT_EVAL_FILES = (
    ROOT / "data" / "evaluation" / "slot_eval_cases.jsonl",
    ROOT / "data" / "evaluation" / "slot_challenge_cases.jsonl",
)

SEED = 6062026
FOODS = [
    "italian",
    "chinese",
    "thai",
    "indian",
    "lebanese",
    "british",
    "japanese",
    "turkish",
    "vegetarian",
    "mediterranean",
    "korean",
    "french",
    "spanish",
    "asian oriental",
]
AREAS = ["centre", "north", "south", "east", "west"]
AREA_SURFACES = {
    "centre": ["centre", "center", "central", "city centre", "city center", "middle of town"],
    "north": ["north", "north side", "northern part"],
    "south": ["south", "south side", "southern part"],
    "east": ["east", "east side", "eastern part"],
    "west": ["west", "west side", "western part"],
}
PRICES = ["cheap", "moderate", "expensive"]
PRICE_SURFACES = {
    "cheap": ["cheap", "budget", "affordable", "low cost", "not pricey", "not expensive"],
    "moderate": ["moderate", "reasonable", "mid range", "not too expensive", "fairly priced"],
    "expensive": ["expensive", "fancy", "high end", "upmarket", "pricey"],
}
DISH_ALIASES = {
    "spaghetti": "italian",
    "pasta": "italian",
    "curry": "indian",
    "sushi": "japanese",
    "noodles": "chinese",
    "kebab": "turkish",
    "veggie": "vegetarian",
}
CUISINE_GROUPS = {
    "Middle Eastern": ["lebanese", "turkish", "mediterranean"],
    "South Asian": ["indian"],
    "East Asian": ["chinese", "cantonese", "japanese", "korean", "asian oriental"],
    "Southeast Asian": ["thai", "vietnamese", "asian oriental"],
}
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "today", "tomorrow"]
TIMES = ["12:00", "12:30", "17:30", "18:00", "18:30", "19:00", "19:30", "20:00", "20:30", "21:00"]
PEOPLE = [1, 2, 3, 4, 5, 6, 8]
REFERENCES = ["BK-A1B2C3", "BK-Q7W8E9", "BK-M4N5P6", "SIM-Z8L84T", "SIM-ROFNYN", "BK-X9Y8Z7"]
RESTAURANTS = [
    "pizza hut city centre",
    "efes restaurant",
    "anatolia",
    "kohinoor",
    "the gandhi",
    "nandos city centre",
    "ask restaurant",
    "meze bar",
    "hakka",
    "ali baba",
]
UNSUPPORTED_TOPICS = [
    "hotel",
    "train",
    "taxi",
    "flights",
    "weather",
    "general knowledge",
    "medical advice",
    "legal advice",
    "financial advice",
]


def compact_target(target: dict[str, Any]) -> str:
    """Return the canonical strict target string."""

    encoded = json.dumps(target, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    decoded = json.loads(encoded)
    if decoded != target:
        raise ValueError(f"Target failed round-trip validation: {target!r}")
    return encoded


def row(user_input: str, intent: str, slots: dict[str, Any] | None = None) -> dict[str, str]:
    """Build one validated training row."""

    clean_input = " ".join(user_input.split())
    if intent not in STRICT_MODEL_INTENTS:
        raise ValueError(f"Unsupported strict intent: {intent}")
    clean_slots = dict(slots or {})
    unsupported = set(clean_slots) - STRICT_MODEL_SLOT_KEYS
    if unsupported:
        raise ValueError(f"Unsupported strict slot keys: {sorted(unsupported)}")
    target = compact_target({"intent": intent, "slots": clean_slots})
    json.loads(target)
    return {"input": clean_input, "target": target}


def _choice(items: list[str], index: int) -> str:
    return items[index % len(items)]


def _search_examples() -> Iterable[dict[str, str]]:
    templates = [
        "could you please find a {price_word} {food} resturant in the {area_word}",
        "i need {food} food {area_word} and {price_word} please",
        "find me {price_word} grub, {food}, around the {area_word}",
        "any {food} places in the {area_word} that are {price_word}",
        "mate can you look for {food} somewhere {price_word} near {area_word}",
        "{price_word} {food} {area_word}",
        "show me a {price_word} place for {food} in {area_word}",
        "looking for {food}, {price_word}, {area_word}, ta",
    ]
    index = 0
    for food in FOODS:
        for area in AREAS:
            for price in PRICES:
                template = _choice(templates, index)
                yield row(
                    template.format(
                        food=food,
                        area_word=_choice(AREA_SURFACES[area], index),
                        price_word=_choice(PRICE_SURFACES[price], index),
                    ),
                    "search",
                    {"food": food, "area": area, "pricerange": price},
                )
                index += 1
    for dish, food in DISH_ALIASES.items():
        for area in AREAS:
            yield row(
                f"i fancy {dish}, find somewhere in the {AREA_SURFACES[area][0]}",
                "search",
                {"food": food, "area": area},
            )
            yield row(
                f"{dish} please not too pricey around {AREA_SURFACES[area][-1]}",
                "search",
                {"food": food, "area": area, "pricerange": "moderate"},
            )
    for group, candidates in CUISINE_GROUPS.items():
        for area in AREAS:
            yield row(
                f"could you find {group.lower()} restaurants in the {AREA_SURFACES[area][0]}",
                "search",
                {"cuisine_group": group, "food_candidates": candidates, "area": area},
            )
    for price in PRICES:
        for price_word in PRICE_SURFACES[price]:
            yield row(f"what is a {price_word} restaurant you can find", "search", {"pricerange": price})
    for phrase in ["i need a restaurant", "find somewhere to eat", "can you help me pick a place", "restaurants please"]:
        yield row(phrase, "search", {})


def _list_examples() -> Iterable[dict[str, str]]:
    templates = [
        "list all {food} resturants in the {area_word}",
        "show every {price_word} restaurant around {area_word}",
        "what {food} places are there",
        "can i see all matching restaurants please",
        "give me the full list for {food} {area_word}",
        "show me all {price_word} {food} options",
    ]
    index = 0
    for food in FOODS:
        for area in AREAS:
            yield row(
                _choice(templates, index).format(
                    food=food,
                    area_word=_choice(AREA_SURFACES[area], index),
                    price_word=_choice(PRICE_SURFACES[_choice(PRICES, index)], index),
                ),
                "list",
                {"food": food, "area": area},
            )
            index += 1
    for price in PRICES:
        for area in AREAS:
            yield row(
                f"could you list the {PRICE_SURFACES[price][1]} resturants near {AREA_SURFACES[area][1]}",
                "list",
                {"area": area, "pricerange": price},
            )
    for group, candidates in CUISINE_GROUPS.items():
        yield row(
            f"list all {group.lower()} places please",
            "list",
            {"cuisine_group": group, "food_candidates": candidates},
        )
    for phrase in ["can you give the list again", "show those options again", "all of them please", "more restaurants"]:
        yield row(phrase, "list", {})


def _restaurant_info_examples() -> Iterable[dict[str, str]]:
    for name in RESTAURANTS:
        yield row(f"what is the address of {name}", "restaurant_info", {})
        yield row(f"tell me about {name} please", "restaurant_info", {})
        yield row(f"phone number for {name}?", "restaurant_info", {})
    for phrase in [
        "what is the address",
        "tell me about it",
        "can i have the postcode",
        "details please",
        "where is that restaurant",
    ]:
        yield row(phrase, "restaurant_info", {})


def _booking_examples() -> Iterable[dict[str, str]]:
    index = 0
    for day in DAYS:
        for time in TIMES:
            people = _choice([str(item) for item in PEOPLE], index)
            surface_time = time.lstrip("0")
            yield row(
                f"book it for {day} at {surface_time} for {people} people",
                "book",
                {"day": day, "time": time, "people": int(people)},
            )
            index += 1
    for name in RESTAURANTS:
        yield row(f"reserve {name} for 2 people", "book", {"people": 2})
        yield row(f"book a table at {name} tomorrow 7pm", "book", {"day": "tomorrow", "time": "19:00"})
    for food in FOODS[:8]:
        yield row(f"book me a {food} restaurant for four", "book", {"food": food, "people": 4})
    for phrase in ["can you book it", "reserve a table please", "i want to make a booking", "book that one mate"]:
        yield row(phrase, "book", {})


def _update_examples() -> Iterable[dict[str, str]]:
    for reference in REFERENCES:
        for day in DAYS[:7]:
            yield row(
                f"move booking {reference} to {day}",
                "reschedule",
                {"booking_reference": reference, "day": day},
            )
        for time in TIMES[2:]:
            yield row(
                f"change {reference} to {time.lstrip('0')}",
                "reschedule",
                {"booking_reference": reference, "time": time},
            )
    for people in PEOPLE:
        yield row(f"actually make it for {people} people", "reschedule", {"people": people})
    for phrase in [
        "reschedule the booking",
        "change the time please",
        "i need to update it",
        "move it to another day",
        "no i meant friday",
    ]:
        slots = {"day": "friday"} if "friday" in phrase else {}
        yield row(phrase, "reschedule", slots)


def _cancel_examples() -> Iterable[dict[str, str]]:
    for reference in REFERENCES:
        yield row(f"cancel booking {reference}", "cancel", {"booking_reference": reference})
        yield row(f"please delete {reference}", "cancel", {"booking_reference": reference})
        yield row(f"i no longer need reservation {reference}", "cancel", {"booking_reference": reference})
    for phrase in ["cancel it", "delete my booking", "scrap the reservation", "call off the table please"]:
        yield row(phrase, "cancel", {})


def _booking_list_examples() -> Iterable[dict[str, str]]:
    for phrase in [
        "show my bookings",
        "list all my reservations",
        "what bookings do i have",
        "booking history please",
        "can i see every table i booked",
        "as a table show my bookings",
    ]:
        yield row(phrase, "booking_list", {})


def _assistant_action_examples() -> Iterable[dict[str, str]]:
    for phrase in ["show another option", "any alternative restaurants", "different place please"]:
        yield row(phrase, "alternative", {})
    for reference in REFERENCES[:3]:
        yield row(f"what is booking {reference}", "booking_info", {"booking_reference": reference})
        yield row(f"status for reservation {reference}", "booking_info", {"booking_reference": reference})
    for area in AREAS:
        yield row(f"actually make the area {area}", "correct", {"area": area})
    for price in PRICES:
        yield row(f"no make it {price} instead", "correct", {"pricerange": price})
    for phrase in ["what cuisines can you search", "which food types do you support", "suggest some cuisine types"]:
        yield row(phrase, "cuisine_help", {})
    for phrase in ["next week", "the following week", "next week please"]:
        yield row(phrase, "date_clarification", {})
    for dish, food in DISH_ALIASES.items():
        yield row(f"i feel like {dish}", "dish_preference", {"dish": dish, "food_candidates": [food]})
    for area in AREAS:
        yield row(f"how far is it from the {area} side", "distance_info", {"area": area})
    for phrase in ["which areas can i filter by", "what price filters are available", "show the supported filters"]:
        yield row(phrase, "filter_info", {})
    for phrase in ["show results as a table", "display those options in table format", "table view please"]:
        yield row(phrase, "table_view", {})
    for phrase in ["hmm", "not sure", "maybe later", "just thinking"]:
        yield row(phrase, "unknown", {})


def _small_talk_examples() -> Iterable[dict[str, str]]:
    for phrase in ["hello", "hi there", "good morning", "hey mate", "hiya", "hello restaurant assistant"]:
        yield row(phrase, "greeting", {})
    for phrase in ["thanks", "thank you", "cheers", "ta", "nice one thanks", "much appreciated"]:
        yield row(phrase, "thanks", {})
    for phrase in ["bye", "goodbye", "see you", "see ya later", "farewell", "that is all goodbye"]:
        yield row(phrase, "thanks", {})


def _unsupported_examples() -> Iterable[dict[str, str]]:
    wrappers = [
        "{topic}",
        "can you help with {topic}",
        "please book me a {topic}",
        "i need {topic} information",
        "what is the latest {topic}",
        "give me advice about {topic}",
    ]
    for topic in UNSUPPORTED_TOPICS:
        for wrapper in wrappers:
            yield row(wrapper.format(topic=topic), "unsupported", {})


def candidate_rows() -> list[dict[str, str]]:
    base_rows: list[dict[str, str]] = []
    for builder in [
        _search_examples,
        _list_examples,
        _restaurant_info_examples,
        _booking_examples,
        _update_examples,
        _cancel_examples,
        _booking_list_examples,
        _assistant_action_examples,
        _small_talk_examples,
        _unsupported_examples,
    ]:
        base_rows.extend(builder())

    rows: list[dict[str, str]] = []
    for item in base_rows:
        rows.extend(_with_noise_variants(item))
    return rows


def _with_noise_variants(item: dict[str, str]) -> list[dict[str, str]]:
    text = item["input"]
    target = item["target"]
    variants = [
        text,
        f"{text} please",
        f"{text} ta",
        f"{text} mate",
    ]
    lowered = text.casefold()
    if not lowered.startswith(("can ", "could ", "please ", "what ", "show ", "list ", "hello", "hi ", "thanks", "thank")):
        variants.append(f"could you {text}")
    if "restaurant" in lowered:
        variants.append(text.replace("restaurants", "resturants").replace("restaurant", "resturant"))
    if "resturant" in lowered:
        variants.append(text.replace("resturants", "resurants").replace("resturant", "resurant"))
    if "centre" in lowered:
        variants.append(text.replace("centre", "center"))
    if "please" not in lowered:
        variants.append(f"please {text}")
    return [{"input": " ".join(variant.split()), "target": target} for variant in variants]


def load_eval_texts(eval_files: Iterable[Path]) -> set[str]:
    texts: set[str] = set()
    for path in eval_files:
        if not path.exists():
            continue
        for record in load_records(path):
            text = str(record.get("text") or record.get("input") or "").strip()
            if text:
                texts.add(normalize_text(text))
    return texts


def build_rows(*, total_examples: int, seed: int, eval_files: Iterable[Path]) -> list[dict[str, str]]:
    if not 1500 <= total_examples <= 3000:
        raise ValueError("--examples must be between 1500 and 3000")
    rng = random.Random(seed)
    eval_texts = load_eval_texts(eval_files)
    pool = candidate_rows()
    seen_inputs: set[str] = set()
    unique: list[dict[str, str]] = []
    for item in pool:
        normalized = normalize_text(item["input"])
        if normalized in eval_texts or normalized in seen_inputs:
            continue
        json.loads(item["target"])
        seen_inputs.add(normalized)
        unique.append(item)

    if len(unique) < total_examples:
        raise ValueError(f"Only generated {len(unique)} unique non-leaking rows; need {total_examples}")

    by_intent: dict[str, list[dict[str, str]]] = {intent: [] for intent in sorted(STRICT_MODEL_INTENTS)}
    for item in unique:
        intent = json.loads(item["target"])["intent"]
        by_intent[intent].append(item)
    missing_intents = [intent for intent, items in by_intent.items() if not items]
    if missing_intents:
        raise ValueError(f"No rows generated for intents: {missing_intents}")

    selected: list[dict[str, str]] = []
    per_intent_floor = max(12, total_examples // (len(by_intent) * 3))
    for intent, items in by_intent.items():
        rng.shuffle(items)
        selected.extend(items[: min(per_intent_floor, len(items))])

    remaining = [item for item in unique if item not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: total_examples - len(selected)])
    rng.shuffle(selected)
    return selected[:total_examples]


def split_rows(rows: list[dict[str, str]], *, dev_ratio: float, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rng = random.Random(seed + 1)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    dev_count = max(1, round(len(shuffled) * dev_ratio))
    return shuffled[dev_count:], shuffled[:dev_count]


def validate_rows(rows: Iterable[dict[str, str]], *, eval_files: Iterable[Path]) -> dict[str, Any]:
    eval_texts = load_eval_texts(eval_files)
    seen: set[str] = set()
    intent_counts: Counter[str] = Counter()
    slot_counts: Counter[str] = Counter()
    count = 0
    for line_number, item in enumerate(rows, start=1):
        count += 1
        if set(item) != {"input", "target"}:
            raise ValueError(f"Row {line_number} must contain only input and target")
        normalized = normalize_text(item["input"])
        if normalized in eval_texts:
            raise ValueError(f"Evaluation fixture text leaked into generated data: {item['input']!r}")
        if normalized in seen:
            raise ValueError(f"Duplicate generated input: {item['input']!r}")
        seen.add(normalized)
        target = json.loads(item["target"])
        if item["target"] != compact_target(target):
            raise ValueError(f"Target is not canonical compact JSON at row {line_number}")
        intent = target.get("intent")
        slots = target.get("slots")
        if intent not in STRICT_MODEL_INTENTS:
            raise ValueError(f"Unsupported intent at row {line_number}: {intent!r}")
        if not isinstance(slots, dict) or set(slots) - STRICT_MODEL_SLOT_KEYS:
            raise ValueError(f"Unsupported slots at row {line_number}: {slots!r}")
        intent_counts[intent] += 1
        slot_counts.update(slots.keys())
    return {"count": count, "intent_counts": dict(intent_counts), "slot_counts": dict(slot_counts)}


def write_jsonl(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(item, ensure_ascii=False, separators=(",", ":")) for item in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build strict JSON slot/intent train/dev data.")
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--dev-output", type=Path, default=DEFAULT_DEV_OUTPUT)
    parser.add_argument("--examples", type=int, default=2200)
    parser.add_argument("--dev-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--eval-file", action="append", type=Path, dest="eval_files")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    eval_files = tuple(args.eval_files) if args.eval_files else DEFAULT_EVAL_FILES
    rows = build_rows(total_examples=args.examples, seed=args.seed, eval_files=eval_files)
    train_rows, dev_rows = split_rows(rows, dev_ratio=args.dev_ratio, seed=args.seed)
    train_summary = validate_rows(train_rows, eval_files=eval_files)
    dev_summary = validate_rows(dev_rows, eval_files=eval_files)
    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.dev_output, dev_rows)
    print(f"Wrote {train_summary['count']} training rows to {args.train_output}")
    print(f"Wrote {dev_summary['count']} dev rows to {args.dev_output}")
    print("Training intent distribution:")
    for intent, count in sorted(train_summary["intent_counts"].items()):
        print(f"  {intent}: {count}")
    print("Dev intent distribution:")
    for intent, count in sorted(dev_summary["intent_counts"].items()):
        print(f"  {intent}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
