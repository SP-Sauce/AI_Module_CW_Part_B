from collections import Counter

from scripts.check_data_leakage import load_records, normalize_text
from scripts.generate_slot_challenge_fixture import (
    MAIN_FIXTURE,
    OUTPUT,
    TRAINING_FILES,
    validate_cases,
)


def test_challenge_fixture_is_valid_and_has_required_size():
    records = load_records(OUTPUT)

    validate_cases(records)
    assert 75 <= len(records) <= 100
    assert all(set(record) == {"text", "intent", "slots"} for record in records)


def test_challenge_fixture_is_separate_from_training_and_main_evaluation():
    challenge_texts = {
        normalize_text(record["text"])
        for record in load_records(OUTPUT)
    }
    existing_texts = set()
    for path in (*TRAINING_FILES, MAIN_FIXTURE):
        existing_texts.update(
            normalize_text(record["text"])
            for record in load_records(path)
        )

    assert challenge_texts.isdisjoint(existing_texts)


def test_challenge_fixture_covers_requested_domains():
    records = load_records(OUTPUT)
    intents = Counter(record["intent"] for record in records)
    text = " ".join(record["text"].casefold() for record in records)

    for intent in (
        "search",
        "list",
        "book",
        "reschedule",
        "cancel",
        "restaurant_info",
        "booking_info",
        "booking_list",
        "unsupported",
        "dish_preference",
    ):
        assert intents[intent] > 0
    for topic in ("taxi", "hotel", "train", "dentist", "payment", "refund", "takeaway"):
        assert topic in text
