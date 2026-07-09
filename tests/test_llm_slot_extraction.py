import json

import pytest

from restaurant_assistant.slot_extraction import (
    OptionalLLMSlotExtractor,
    adapter_slot_prompt,
    parse_llm_json_output,
)


class FakeText2TextPipeline:
    def __init__(self, payload):
        self.payload = payload

    def __call__(self, prompt, max_new_tokens=96, do_sample=False):
        return [{"generated_text": json.dumps(self.payload)}]


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

    assert "Return exactly one JSON object." in prompt
    assert 'Use only keys "intent" and "slots".' in prompt
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
    assert result.llm_raw_output == '"not-json"'
    assert "Raw output" in result.errors[0]


def test_json_parser_error_contains_bounded_raw_preview():
    with pytest.raises(ValueError, match="Raw output") as exc_info:
        parse_llm_json_output("x" * 500)

    assert len(str(exc_info.value)) < 400
