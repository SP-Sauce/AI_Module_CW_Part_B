import json

from restaurant_assistant.slot_extraction import OptionalLLMSlotExtractor


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
