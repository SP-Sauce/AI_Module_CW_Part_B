"""Generate a deterministic held-out challenge fixture for slot extraction."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from restaurant_assistant.slot_extraction import ALLOWED_INTENTS, ALLOWED_SLOT_KEYS

try:
    from .check_data_leakage import load_records, normalize_text
except ImportError:
    from check_data_leakage import load_records, normalize_text


OUTPUT = ROOT / "data" / "evaluation" / "slot_challenge_cases.jsonl"
TRAINING_FILES = (
    ROOT / "data" / "training" / "slot_instruction_examples.jsonl",
    ROOT / "data" / "training" / "slot_instruction_examples_augmented.jsonl",
)
MAIN_FIXTURE = ROOT / "data" / "evaluation" / "slot_eval_cases.jsonl"


def case(text: str, intent: str, slots: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"text": text, "intent": intent, "slots": slots or {}}


def build_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = [
        case("Need summat cheap and Italian out east, ta", "search", {"food": "italian", "area": "east", "pricerange": "cheap"}),
        case("Any upmarket Japanese grub near the north?", "search", {"food": "japanese", "area": "north", "pricerange": "expensive"}),
        case("reasonable veggie resturant in west pls", "search", {"food": "vegetarian", "area": "west", "pricerange": "moderate"}),
        case("Could I get Thai food somewhere south and cheap?", "search", {"food": "thai", "area": "south", "pricerange": "cheap"}),
        case("I'm after British cooking in the centre, nothing pricey", "search", {"food": "british", "area": "centre", "pricerange": "cheap"}),
        case("Find Chinese around east Cambridge at a reasonable price", "search", {"food": "chinese", "area": "east", "pricerange": "moderate"}),
        case("Lebanese, north side, make it upmarket", "search", {"food": "lebanese", "area": "north", "pricerange": "expensive"}),
        case("Show me Indian food down south on a budget", "search", {"food": "indian", "area": "south", "pricerange": "cheap"}),
        case("Find me a resturant, not sure what kind yet", "search"),
        case("I just want somewhere nice to eat pls", "search"),
        case("Could you search for food around town?", "search"),
        case("Need a place for dinner, any ideas?", "search"),
        case("List the cheap Chinese choices on the west side", "list", {"food": "chinese", "area": "west", "pricerange": "cheap"}),
        case("all Italian options east of town please", "list", {"food": "italian", "area": "east"}),
        case("What Thai resturants are in the centre?", "list", {"food": "thai", "area": "centre"}),
        case("show every moderate Indian place in north Cambridge", "list", {"food": "indian", "area": "north", "pricerange": "moderate"}),
        case("Any Lebanese listings for the south?", "list", {"food": "lebanese", "area": "south"}),
        case("Fancy something Middle-Eastern around west Cambridge", "list", {"cuisine_group": "Middle Eastern", "food_candidates": ["lebanese", "turkish", "mediterranean"], "area": "west"}),
        case("What South-Asian choices are there in the east?", "list", {"cuisine_group": "South Asian", "food_candidates": ["indian"], "area": "east"}),
        case("Show East-Asian food near the north side", "list", {"cuisine_group": "East Asian", "food_candidates": ["chinese", "cantonese", "japanese", "korean", "asian oriental"], "area": "north"}),
        case("Any Southeast-Asian places down south?", "list", {"cuisine_group": "Southeast Asian", "food_candidates": ["thai", "vietnamese", "asian oriental"], "area": "south"}),
        case("West-African cuisine around central Cambridge please", "list", {"cuisine_group": "West African", "food_candidates": ["african"], "area": "centre"}),
        case("Craving curry but dunno which cuisine", "dish_preference", {"dish": "curry", "food_candidates": ["indian", "thai"]}),
        case("Where might I find proper noodles?", "dish_preference", {"dish": "noodles", "food_candidates": ["chinese", "thai", "vietnamese"]}),
        case("Pizza sounds good, suggest a cuisine", "dish_preference", {"dish": "pizza", "food_candidates": ["italian"]}),
        case("I fancy mezze tonight, what food type is that?", "dish_preference", {"dish": "mezze", "food_candidates": ["lebanese", "mediterranean", "turkish"]}),
        case("Got anywhere doing sushi?", "dish_preference", {"dish": "sushi", "food_candidates": ["japanese"]}),
    ]

    booking_specs = [
        ("tmrw", "tomorrow", "18:00", 2),
        ("today", "today", "20:00", 4),
        ("tomorrow", "tomorrow", "19:30", 6),
        ("today", "today", "12:30", 3),
        ("tmrw", "tomorrow", "17:45", 5),
        ("today", "today", "21:00", 8),
    ]
    for index, (spoken_day, relative_day, time, people) in enumerate(booking_specs, start=1):
        cases.append(
            case(
                f"book a tbale {spoken_day} at {time.lstrip('0')} for a party of {people}, challenge {index}",
                "book",
                {"relative_day": relative_day, "time": time, "people": people},
            )
        )
    cases.extend(
        [
            case("Can you start a booking for tomorrow?", "book", {"relative_day": "tomorrow"}),
            case("I'd like to reserve a table but haven't picked a time", "book"),
            case("shift BK-C7D8E9 to monday at 6:30pm", "reschedule", {"booking_reference": "BK-C7D8E9", "day": "monday", "time": "18:30"}),
            case("SIM-F1G2H3 needs moving to thursday, 20:00", "reschedule", {"booking_reference": "SIM-F1G2H3", "day": "thursday", "time": "20:00"}),
            case("pls change BK-J4K5L6 to saturday noon", "reschedule", {"booking_reference": "BK-J4K5L6", "day": "saturday", "time": "12:00"}),
            case("rescheduel SIM-M7N8P9 for friday at quarter past seven", "reschedule", {"booking_reference": "SIM-M7N8P9", "day": "friday", "time": "19:15"}),
            case("Move my booking to Tuesday, I don't know the ref", "reschedule", {"day": "tuesday"}),
            case("cancle BK-Q1R2S3 right away", "cancel", {"booking_reference": "BK-Q1R2S3"}),
            case("please scrap reservation SIM-T4U5V6", "cancel", {"booking_reference": "SIM-T4U5V6"}),
            case("I no longer need BK-W7X8Y9", "cancel", {"booking_reference": "BK-W7X8Y9"}),
            case("cancel my current table, ref SIM-Z1A2B3", "cancel", {"booking_reference": "SIM-Z1A2B3"}),
            case("Can you cancel it? I lost the reference", "cancel"),
            case("What's the address for the place you just showed?", "restaurant_info"),
            case("Does that resturant have a phone number?", "restaurant_info"),
            case("Tell me the postcode for my selected place", "restaurant_info"),
            case("More details about this restaurant pls", "restaurant_info"),
            case("look up booking BK-D4E5F6", "booking_info", {"booking_reference": "BK-D4E5F6"}),
            case("What time is SIM-G7H8J9 booked for?", "booking_info", {"booking_reference": "SIM-G7H8J9"}),
            case("show details of BK-K1L2M3", "booking_info", {"booking_reference": "BK-K1L2M3"}),
            case("I need info on my current booking", "booking_info"),
            case("what reservations are in this session?", "booking_list"),
            case("show all the tables I've booked here", "booking_list"),
            case("booking history for this chat pls", "booking_list"),
            case("remind me which filters are on", "filter_info"),
            case("what criteria narrowed these results?", "filter_info"),
            case("put the current choices into a tbale", "table_view"),
            case("switch these booking results to table format", "table_view"),
        ]
    )

    unsupported = (
        ("order me a taxi to the station", "taxi"),
        ("find a hotel with late checkout", "hotel"),
        ("when is the next train to London?", "train"),
        ("need a dentist open tonight", "dentist"),
        ("take payment for this booking", "payment"),
        ("I want a refund on my meal", "refund"),
        ("place a takeaway order for curry", "takeaway"),
        ("where can I buy fireworks nearby?", "fireworks"),
        ("book me a flight for the weekend", "flight"),
        ("store my credit card for later", "credit card"),
    )
    cases.extend(case(text, "unsupported") for text, _ in unsupported)
    cases.extend(
        [
            case("hiya, you alright?", "greeting"),
            case("morning bot, let's find some food", "greeting"),
            case("cheers mate, sorted", "thanks"),
            case("ta very much, that's everything", "thanks"),
            case("How far is that from the west side of Cambridge?", "distance_info", {"area": "west"}),
            case("Is this place walkable from the centre?", "distance_info", {"area": "centre"}),
            case("distance from the north, roughly?", "distance_info", {"area": "north"}),
            case("Actually west, not east", "correct", {"area": "west"}),
            case("Oops make that expensive rather than cheap", "correct", {"pricerange": "expensive"}),
            case("No, I said 8pm not 8am", "correct", {"time": "20:00"}),
        ]
    )
    return cases


def validate_cases(cases: list[dict[str, Any]]) -> None:
    if not 75 <= len(cases) <= 100:
        raise ValueError(f"Expected 75-100 challenge cases, got {len(cases)}")
    known_texts: set[str] = set()
    for path in (*TRAINING_FILES, MAIN_FIXTURE):
        if path.exists():
            known_texts.update(
                normalize_text(str(record.get("text") or record.get("input") or ""))
                for record in load_records(path)
            )
    seen: set[str] = set()
    for index, record in enumerate(cases, start=1):
        if set(record) != {"text", "intent", "slots"}:
            raise ValueError(f"Invalid challenge schema at row {index}")
        if record["intent"] not in ALLOWED_INTENTS:
            raise ValueError(f"Unsupported challenge intent at row {index}")
        if not isinstance(record["slots"], dict) or not set(record["slots"]).issubset(ALLOWED_SLOT_KEYS):
            raise ValueError(f"Unsupported challenge slots at row {index}")
        normalized = normalize_text(record["text"])
        if normalized in known_texts:
            raise ValueError(f"Challenge leakage at row {index}: {record['text']!r}")
        if normalized in seen:
            raise ValueError(f"Duplicate challenge text at row {index}: {record['text']!r}")
        seen.add(normalized)


def main() -> None:
    cases = build_cases()
    validate_cases(cases)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True, separators=(",", ":")) for record in cases) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(cases)} challenge cases to {OUTPUT}")


if __name__ == "__main__":
    main()
