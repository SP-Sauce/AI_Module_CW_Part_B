import json

import pytest

from restaurant_assistant.slot_extraction import (
    OptionalLLMSlotExtractor,
    adapter_slot_prompt,
    parse_llm_json_output,
    repair_llm_json_output,
)


class FakeText2TextPipeline:
    def __init__(self, payload):
        self.payload = payload
        self.generation_kwargs = None

    def __call__(self, prompt, **generation_kwargs):
        self.generation_kwargs = generation_kwargs
        generated_text = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        return [{"generated_text": generated_text}]


def test_optional_llm_slot_extractor_uses_llm_and_rule_guards():
    extractor = OptionalLLMSlotExtractor("fake-slot-model")
    extractor._pipeline = FakeText2TextPipeline(
        {
            "intent": "book",
            "slots": {
                "food": "martian",
                "area": "moon",
                "day": "friday",
                "time": "7pm",
                "people": 2,
            },
        }
    )

    result = extractor.extract("Can you book an Italian restaurant in the centre for Friday at 7pm for 2 people?")

    assert result.used_llm is True
    assert result.llm_attempted is True
    assert result.llm_parse_success is True
    assert result.llm_raw_output is not None
    assert extractor._pipeline.generation_kwargs == {
        "max_new_tokens": 128,
        "do_sample": False,
        "num_beams": 1,
        "early_stopping": True,
        "repetition_penalty": 1.2,
    }
    assert result.intent == "book"
    assert result.slots["food"] == "italian"
    assert result.slots["area"] == "centre"
    assert result.slots["day"] == "friday"
    assert result.slots["time"] == "19:00"
    assert result.slots["people"] == 2
    assert "martian" not in result.slots.values()
    assert "moon" not in result.slots.values()


def test_optional_llm_slot_extractor_keeps_unsupported_safety_intent():
    extractor = OptionalLLMSlotExtractor("fake-slot-model")
    extractor._pipeline = FakeText2TextPipeline({"intent": "search", "slots": {"area": "centre"}})

    result = extractor.extract("I want to buy a gun")

    assert result.used_llm is True
    assert result.intent == "unsupported"


def test_optional_llm_slot_extractor_keeps_explicit_booking_action():
    extractor = OptionalLLMSlotExtractor("fake-slot-model")
    extractor._pipeline = FakeText2TextPipeline(
        {
            "intent": "dish_preference",
            "slots": {"dish": "curry", "food_candidates": ["indian", "thai"]},
        }
    )

    result = extractor.extract("Book the golden curry for tomorrow, 6pm, 4 people")

    assert result.used_llm is True
    assert result.intent == "book"
    assert result.slots["relative_day"] == "tomorrow"
    assert result.slots["time"] == "18:00"
    assert result.slots["people"] == 4


def test_adapter_prompt_uses_strict_answer_marker_without_examples():
    prompt = adapter_slot_prompt("hello")

    assert "Task: Extract the restaurant assistant intent and slots." in prompt
    assert "Return only one valid minified JSON object." in prompt
    assert "Allowed intents: search, list, restaurant_info, book, update_booking" in prompt
    assert prompt.endswith("User: hello\nJSON:")
    assert '{"intent":' not in prompt


def test_json_parser_uses_text_after_last_answer_marker():
    raw_output = (
        'JSON: {"intent":"wrong","slots":{}}\n'
        "repeated prompt\n"
        'JSON: commentary {"intent":"search","slots":{"area":"centre"}} trailing {broken'
    )

    assert parse_llm_json_output(raw_output) == {
        "intent": "search",
        "slots": {"area": "centre"},
    }


def test_json_parser_skips_objects_without_required_keys():
    raw_output = 'metadata {"debug":true} then {"intent":"thanks","slots":{}}'

    assert parse_llm_json_output(raw_output) == {"intent": "thanks", "slots": {}}


def test_parse_failure_preserves_llm_diagnostics():
    extractor = OptionalLLMSlotExtractor("fake-slot-model")
    extractor._pipeline = FakeText2TextPipeline("not-json")

    result = extractor.extract("hello")

    assert result.used_llm is False
    assert result.llm_attempted is True
    assert result.llm_parse_success is False
    assert result.llm_raw_output == "not-json"
    assert "Raw output" in result.errors[0]


def test_json_parser_error_contains_bounded_raw_preview():
    with pytest.raises(ValueError, match="Raw output") as exc_info:
        parse_llm_json_output("x" * 500)

    assert len(str(exc_info.value)) < 400


@pytest.mark.parametrize(
    ("fragment", "expected_slots"),
    [
        ('"slots": "area": "centre"', {"area": "centre"}),
        ('"slots": "food": "thai"', {"food": "thai"}),
        ('"slots": "time": "6"', {"time": "6"}),
        ('"slots": "people": 4', {"people": 4}),
        (
            '"slots": "booking_reference": "BK-X9Y8Z7"',
            {"booking_reference": "BK-X9Y8Z7"},
        ),
    ],
)
def test_json_repair_handles_known_slot_fragments(fragment, expected_slots):
    repaired, repaired_output = repair_llm_json_output(f'"intent": "book", {fragment}')

    assert repaired == {"intent": "book", "slots": expected_slots}
    assert json.loads(repaired_output) == repaired


def test_json_repair_keeps_first_valid_duplicate_and_ignores_disallowed_keys():
    repaired, _ = repair_llm_json_output(
        '"intent":"search","slots":"area":oops,"area":"north","area":"south",'
        '"password":"secret","food":"thai"'
    )

    assert repaired == {
        "intent": "search",
        "slots": {"area": "north", "food": "thai"},
    }


def test_json_repair_keeps_already_valid_structured_output():
    raw_output = 'prefix {"intent":"thanks","slots":{}} trailing text'

    repaired, repaired_output = repair_llm_json_output(raw_output)

    assert repaired == {"intent": "thanks", "slots": {}}
    assert repaired_output == '{"intent":"thanks","slots":{}}'


def test_optional_extractor_tracks_repaired_output_without_hiding_parse_failure():
    extractor = OptionalLLMSlotExtractor("fake-slot-model")
    extractor._pipeline = FakeText2TextPipeline(
        '"intent": "list", "slots": "area": "city centre"'
    )

    result = extractor.extract("Show restaurants in the centre")

    assert result.used_llm is True
    assert result.llm_parse_success is False
    assert result.llm_repair_success is True
    assert result.llm_repair_weak is False
    assert result.llm_intent_trusted is True
    assert result.llm_slots_trusted is True
    assert json.loads(result.llm_repaired_output) == {
        "intent": "list",
        "slots": {"area": "city centre"},
    }
    assert result.slots["area"] == "centre"
    assert result.errors and "valid intent/slots JSON" in result.errors[0]


def test_weak_dangling_slots_repair_uses_rule_intent():
    extractor = OptionalLLMSlotExtractor("fake-slot-model")
    extractor._pipeline = FakeText2TextPipeline('"intent":"list","slots":')

    result = extractor.extract("Show every restaurant in the east")

    assert result.llm_repair_success is True
    assert result.llm_repair_weak is True
    assert result.llm_intent_trusted is False
    assert result.llm_slots_trusted is False
    assert result.intent == "list"
    assert result.slots == {"area": "east"}


def test_conflicting_repaired_intent_cannot_override_confident_rule_intent():
    extractor = OptionalLLMSlotExtractor("fake-slot-model")
    extractor._pipeline = FakeText2TextPipeline(
        '"intent":"list","slots":"area":"city centre"'
    )

    result = extractor.extract("I need a cheap Chinese restaurant in the centre")

    assert result.llm_repair_success is True
    assert result.llm_repair_weak is True
    assert result.llm_intent_trusted is False
    assert result.intent == "search"
    assert result.slots == {
        "food": "chinese",
        "area": "centre",
        "pricerange": "cheap",
    }


def test_invalid_repaired_intent_is_weak():
    extractor = OptionalLLMSlotExtractor("fake-slot-model")
    extractor._pipeline = FakeText2TextPipeline(
        '"intent":"browse","slots":"area":"east"'
    )

    result = extractor.extract("Show every restaurant in the east")

    assert result.llm_repair_weak is True
    assert result.llm_intent_trusted is False
    assert result.intent == "list"


def test_complete_repaired_slots_object_can_support_non_conflicting_intent():
    extractor = OptionalLLMSlotExtractor("fake-slot-model")
    extractor._pipeline = FakeText2TextPipeline(
        '"intent":"list","slots":{"area":"east"}'
    )

    result = extractor.extract("Show every restaurant in the east")

    assert result.llm_parse_success is False
    assert result.llm_repair_success is True
    assert result.llm_repair_weak is False
    assert result.llm_intent_trusted is True
    assert result.intent == "list"


def test_guarded_rule_intent_is_preserved_even_after_strict_parse():
    extractor = OptionalLLMSlotExtractor("fake-slot-model")
    extractor._pipeline = FakeText2TextPipeline({"intent": "list", "slots": {}})

    result = extractor.extract("hello")

    assert result.llm_parse_success is True
    assert result.llm_intent_trusted is False
    assert result.intent == "greeting"


def test_llm_slots_fill_missing_values_but_do_not_override_rule_slots():
    extractor = OptionalLLMSlotExtractor("fake-slot-model")
    extractor._pipeline = FakeText2TextPipeline(
        {"intent": "search", "slots": {"area": "west", "food": "thai"}}
    )

    result = extractor.extract("Find a restaurant in the east")

    assert result.llm_slots_trusted is True
    assert result.llm_meaningful_slot_contribution is True
    assert result.slots == {"area": "east", "food": "thai"}


@pytest.mark.parametrize(
    ("message", "raw_output", "expected_slots"),
    [
        (
            "What places do you have for south asian food?",
            '"intent":"restaurant_info","slots":"food":"indian"',
            {
                "cuisine_group": "South Asian",
                "food_candidates": ["indian"],
            },
        ),
        (
            "Book this restaurant for today at noon, table for two",
            '"intent":"book","slots":"day":"today","people":2',
            {"relative_day": "today", "time": "12:00", "people": 2},
        ),
        (
            "Are there middle eastern restaurants in the centre?",
            '"intent":"restaurant_info","slots":"food":"lebanese","area":"centre"',
            {
                "cuisine_group": "Middle Eastern",
                "food_candidates": ["lebanese", "turkish", "mediterranean"],
                "area": "centre",
            },
        ),
        (
            "Any east asian options around town?",
            '"intent":"filter_info","slots":"food":"asian oriental","area":"town"',
            {
                "cuisine_group": "East Asian",
                "food_candidates": [
                    "chinese",
                    "cantonese",
                    "japanese",
                    "korean",
                    "asian oriental",
                ],
            },
        ),
    ],
)
def test_semantic_duplicate_llm_slots_do_not_reduce_rule_exactness(
    message,
    raw_output,
    expected_slots,
):
    extractor = OptionalLLMSlotExtractor("fake-slot-model")
    extractor._pipeline = FakeText2TextPipeline(raw_output)

    result = extractor.extract(message)

    assert result.slots == expected_slots
    assert result.llm_meaningful_slot_contribution is False


@pytest.mark.parametrize("area", ["Cambridge", "town", "around town", "city"])
def test_vague_llm_area_values_are_not_merged(area):
    extractor = OptionalLLMSlotExtractor("fake-slot-model")
    extractor._pipeline = FakeText2TextPipeline(
        {"intent": "search", "slots": {"area": area}}
    )

    result = extractor.extract("Find me a restaurant")

    assert "area" not in result.slots
    assert result.llm_slots_trusted is False


def test_invalid_llm_price_and_booking_reference_are_not_merged():
    extractor = OptionalLLMSlotExtractor("fake-slot-model")
    extractor._pipeline = FakeText2TextPipeline(
        {
            "intent": "search",
            "slots": {
                "pricerange": "less than",
                "booking_reference": "BOOKING-123",
            },
        }
    )

    result = extractor.extract("Find a restaurant")

    assert result.slots == {}
    assert result.llm_slots_trusted is False
