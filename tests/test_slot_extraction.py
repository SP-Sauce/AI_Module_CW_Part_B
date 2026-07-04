import json
from pathlib import Path

from restaurant_assistant.slot_extraction import extract_slots


def test_slot_fixture_cases():
    cases = json.loads(Path("tests/fixtures/slot_cases.json").read_text(encoding="utf-8"))

    for case in cases:
        result = extract_slots(case["text"])
        assert result.intent == case["intent"]
        for key, expected in case["slots"].items():
            assert result.slots[key] == expected


def test_unsupported_values_are_not_added():
    result = extract_slots("Find me a Martian restaurant in the moon district")

    assert "food" not in result.slots
    assert "area" not in result.slots


def test_transcript_typo_and_booking_intents():
    cases = [
        ("I need a cheap italian resturant in the city centre", "search", {"area": "centre"}),
        ("can you tell me any othe rresutrant nearby?", "alternative", {}),
        ("resechdeule the booking", "reschedule", {}),
        ("I want to reschedule this booking: SIM-OGTV7J. change it to 10:00", "reschedule", {"time": "10:00"}),
        ("cancel SIM-OGTV7J.", "cancel", {"booking_reference": "SIM-OGTV7J"}),
        ("I want to buy a gun", "unsupported", {}),
    ]

    for text, expected_intent, expected_slots in cases:
        result = extract_slots(text)
        assert result.intent == expected_intent
        for key, expected_value in expected_slots.items():
            assert result.slots[key] == expected_value


def test_unsupported_area_is_reported():
    result = extract_slots("give me a list of resturants in the countryside")

    assert result.intent == "list"
    assert result.unsupported_slots == {"area": "countryside"}


def test_am_time_is_not_converted_to_pm():
    result = extract_slots("book it for this sunday, around 9am for 4 people")

    assert result.slots["time"] == "09:00"


def test_time_is_not_misread_as_people_count():
    result = extract_slots("I want to book pizza express for 12:00 this monday for 4 people")

    assert result.intent == "book"
    assert result.slots["time"] == "12:00"
    assert result.slots["people"] == 4


def test_midnight_and_booking_reference_intents():
    midnight = extract_slots("I want to book pizza express for midnight this monday for 4 people")
    info = extract_slots("tell me about SIM-Z8L84T booking.")
    cancel = extract_slots("SIM-ROFNYN can I cancle this booking")

    assert midnight.slots["time"] == "00:00"
    assert info.intent == "booking_info"
    assert cancel.intent == "cancel"
    assert cancel.slots["booking_reference"] == "SIM-ROFNYN"


def test_list_alternative_and_food_typo_intents():
    indian_list = extract_slots("I want to see the resturants that are indian in the area")
    more = extract_slots("anymore resturants?")
    mediterranean = extract_slots("I want to see a list of the mediteranian resutrants in the area")

    assert indian_list.intent == "list"
    assert indian_list.slots["food"] == "indian"
    assert more.intent == "alternative"
    assert mediterranean.intent == "list"
    assert mediterranean.slots["food"] == "mediterranean"


def test_action_intent_wins_over_booking_info():
    reschedule = extract_slots("tell me about SIM-6ZP1SQ, are you able to reschduele to the next tuesday?")
    cancel = extract_slots("tell me about SIM-6ZP1SQ, are you able to cancel?")

    assert reschedule.intent == "reschedule"
    assert reschedule.slots["day"] == "tuesday"
    assert cancel.intent == "cancel"


def test_relative_day_slots_are_extracted():
    tomorrow = extract_slots("book it for tomorrow at 7pm for 2 people")
    day_after = extract_slots("move it to the day after")
    day_after_tomorrow = extract_slots("book it for day after tomorrow at 8pm for 3 people")

    assert tomorrow.slots["relative_day"] == "tomorrow"
    assert day_after.intent == "reschedule"
    assert day_after.slots["relative_day"] == "day_after"
    assert day_after_tomorrow.slots["relative_day"] == "day_after_tomorrow"
