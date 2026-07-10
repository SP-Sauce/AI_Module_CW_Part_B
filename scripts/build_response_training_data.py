"""Build supervised examples for grounded response generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from restaurant_assistant.config import get_settings
from restaurant_assistant.data_loader import load_restaurants
from restaurant_assistant.llm_generator import validate_generated_response


DEFAULT_TRAIN_OUTPUT = ROOT / "data" / "training" / "response_generation_examples.jsonl"
DEFAULT_EVAL_OUTPUT = ROOT / "data" / "evaluation" / "response_generation_eval.jsonl"
INSTRUCTION = "Generate a short grounded restaurant assistant response using only the provided evidence."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build response-generation JSONL examples.")
    parser.add_argument("--sample-data", action="store_true", help="Use bundled sample restaurants.")
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--eval-output", type=Path, default=DEFAULT_EVAL_OUTPUT)
    return parser


def _field(record: dict[str, Any], key: str, fallback: str = "") -> str:
    return str(record.get(key) or fallback).strip()


def _evidence(record: dict[str, Any]) -> str:
    fields = [
        ("name", "name"),
        ("food", "food"),
        ("area", "area"),
        ("pricerange", "pricerange"),
        ("address", "address"),
        ("postcode", "postcode"),
        ("phone", "phone"),
    ]
    return "; ".join(f"{label}={_field(record, key)}" for label, key in fields if _field(record, key))


def _state(**slots: Any) -> str:
    return ", ".join(f"{key}={value}" for key, value in slots.items() if value not in (None, "", []))


def _input(
    *,
    intent: str,
    user: str,
    state: str = "",
    evidence: str = "",
    missing_slots: Iterable[str] = (),
) -> str:
    parts = [
        "Task: Generate a grounded restaurant assistant response.",
        f"Intent: {intent}",
        f"User: {user}",
    ]
    if state:
        parts.append(f"State: {state}")
    if evidence:
        parts.append(f"Evidence: {evidence}")
    parts.append(f"Missing slots: {list(missing_slots)}")
    parts.append("Response:")
    return "\n".join(parts)


def row(
    *,
    intent: str,
    user: str,
    output: str,
    record: dict[str, Any] | None = None,
    state: str = "",
    evidence: str = "",
    missing_slots: Iterable[str] = (),
) -> dict[str, str]:
    evidence_text = evidence or (_evidence(record or {}) if record else "")
    validation = validate_generated_response(
        output,
        evidence_records=[record] if record else [],
        known_restaurant_records=[record] if record else [],
    )
    if not validation.ok:
        raise ValueError(f"Unsafe generated output for {intent}: {validation.reason}: {output}")
    return {
        "instruction": INSTRUCTION,
        "input": _input(
            intent=intent,
            user=user,
            state=state,
            evidence=evidence_text,
            missing_slots=missing_slots,
        ),
        "output": output,
    }


def _pick_records(restaurants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    usable = [
        record
        for record in restaurants
        if record.get("name") and record.get("food") and record.get("area") and record.get("pricerange")
    ]
    return usable[:12] if usable else restaurants[:12]


def build_rows(restaurants: list[dict[str, Any]]) -> list[dict[str, str]]:
    records = _pick_records(restaurants)
    if not records:
        raise ValueError("No restaurant records available for response examples.")
    first = records[0]
    second = records[1 % len(records)]
    third = records[2 % len(records)]
    rows: list[dict[str, str]] = []

    rows.append(
        row(
            intent="greeting",
            user="hello",
            output="Hi - I can help find restaurants in the loaded MultiWOZ data and create booking records.",
        )
    )

    for record in records[:6]:
        name = _field(record, "name")
        food = _field(record, "food")
        area = _field(record, "area")
        price = _field(record, "pricerange")
        rows.append(
            row(
                intent="search",
                user=f"Need {price} {food} in the {area}",
                state=_state(food=food, area=area, pricerange=price),
                record=record,
                output=(
                    f"I found {name} ({price} {food}) in the {area} area, which matches your request for "
                    f"a {price} {food} restaurant in the {area} area."
                ),
            )
        )

    rows.append(
        row(
            intent="search",
            user=f"Find a cheap {_field(first, 'food')} restaurant in the west",
            state=_state(food=_field(first, "food"), area="west", pricerange="cheap"),
            record=first,
            output=(
                f"I could not find an exact match, but the closest option I have is {_field(first, 'name')} "
                f"({_field(first, 'pricerange')} {_field(first, 'food')}) in the {_field(first, 'area')} area."
            ),
        )
    )

    list_names = ", ".join(_field(record, "name") for record in records[:3])
    rows.append(
        row(
            intent="list",
            user="show me the matching restaurants",
            evidence=" | ".join(_evidence(record) for record in records[:3]),
            output=f"Matching restaurants: {list_names}.",
        )
    )

    rows.extend(
        [
            row(
                intent="search",
                user="I need a restaurant",
                state="",
                output="Sure - what kind of food would you like me to search for?",
                missing_slots=["food"],
            ),
            row(
                intent="search",
                user="Find Italian food",
                state=_state(food="italian"),
                output="Sure - which area would you like me to search in?",
                missing_slots=["area"],
            ),
            row(
                intent="search",
                user="Find Italian in the centre",
                state=_state(food="italian", area="centre"),
                output="Sure - what price range would you prefer?",
                missing_slots=["pricerange"],
            ),
        ]
    )

    rows.append(
        row(
            intent="restaurant_info",
            user=f"What is the address for {_field(second, 'name')}?",
            record=second,
            output=(
                f"{_field(second, 'name')} is in the {_field(second, 'area')} area. "
                f"Address: {_field(second, 'address')}. Postcode: {_field(second, 'postcode')}. "
                f"Phone: {_field(second, 'phone')}."
            ),
        )
    )

    rows.append(
        row(
            intent="book",
            user=f"Book {_field(first, 'name')}",
            state=_state(restaurant=_field(first, "name")),
            record=first,
            missing_slots=["day", "time", "people"],
            output=(
                f"Great, I can create a booking record for {_field(first, 'name')}. "
                "To finish it, I still need the day, time and number of people."
            ),
        )
    )

    rows.append(
        row(
            intent="book",
            user=f"Book it for Friday at 19:00 for 2",
            state=_state(restaurant=_field(first, "name"), day="friday", time="19:00", people=2),
            record=first,
            output=(
                f"Great, I have created a booking record for {_field(first, 'name')} on Friday at 19:00 "
                "for 2 people. Your reference is BK-EXAMPLE."
            ),
        )
    )

    rows.append(
        row(
            intent="reschedule",
            user="Move it to Saturday at 20:00",
            state=_state(restaurant=_field(first, "name"), day="saturday", time="20:00", people=2),
            record=first,
            output=f"Done, I have updated booking BK-EXAMPLE for {_field(first, 'name')} to Saturday at 20:00 for 2 people.",
        )
    )

    rows.append(
        row(
            intent="cancel",
            user="Cancel it",
            output="Done - I have cancelled that booking record.",
        )
    )

    rows.append(
        row(
            intent="booking_info",
            user="What is my booking?",
            state=_state(restaurant=_field(first, "name"), day="friday", time="19:00", people=2),
            record=first,
            output=f"Booking BK-EXAMPLE is confirmed for {_field(first, 'name')}: Friday at 19:00 for 2 people.",
        )
    )

    rows.append(
        row(
            intent="booking_list",
            user="Show my bookings",
            evidence=f"name={_field(first, 'name')} | name={_field(third, 'name')}",
            output=f"Current session booking records: 1. BK-EXAMPLE: {_field(first, 'name')} on Friday at 19:00 for 2 people.",
        )
    )

    rows.append(
        row(
            intent="unsupported",
            user="Can you book me a hotel?",
            output=(
                "I can only help with MultiWOZ restaurant search and restaurant booking records. "
                "Try asking for a food type, area or price range."
            ),
        )
    )

    cuisines = sorted({_field(record, "food") for record in records if _field(record, "food")})
    rows.append(
        row(
            intent="cuisine_help",
            user="What cuisines can I search?",
            output="Available cuisine categories include: " + ", ".join(cuisine.title() for cuisine in cuisines[:8]) + ".",
        )
    )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for item in rows:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    restaurants = load_restaurants(settings, use_sample=args.sample_data)
    rows = build_rows(restaurants)
    eval_rows = rows[::4]
    train_rows = [row for index, row in enumerate(rows) if index % 4 != 0]
    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.eval_output, eval_rows)
    print(f"Saved {len(train_rows)} training examples to {args.train_output}")
    print(f"Saved {len(eval_rows)} evaluation examples to {args.eval_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
