from restaurant_assistant.dialogue_state import DialogueState
from restaurant_assistant.llm_generator import GroundedResponseGenerator
from restaurant_assistant.ranking import RankedRestaurant


def test_fallback_generator_does_not_invent_missing_details():
    generator = GroundedResponseGenerator(enable_llm=False)
    state = DialogueState(food="italian", area="south", pricerange="cheap")
    ranked = [
        RankedRestaurant(
            record={
                "name": "Test Restaurant",
                "food": "italian",
                "area": "south",
                "pricerange": "cheap",
                "address": "1 Test Street",
            },
            score=9.0,
            matched_constraints=["food", "area", "pricerange"],
            missing_unmatched_constraints=[],
            explanation="matched food, area, pricerange; tf-idf similarity 0.500",
            similarity=0.5,
        )
    ]

    result = generator.generate("Find cheap Italian in the south", state, ranked)

    assert "Test Restaurant" in result.text
    assert "1 Test Street" in result.text
    assert "Phone:" not in result.text
    assert not result.used_llm


def test_generator_uses_llm_pipeline_when_enabled():
    generator = GroundedResponseGenerator(enable_llm=True, model_name="fake-generator")

    class FakeText2TextPipeline:
        def __call__(self, prompt, max_new_tokens=120, do_sample=False):
            return [{"generated_text": "I found Test Restaurant from the provided evidence."}]

    generator._pipeline = FakeText2TextPipeline()
    state = DialogueState(food="italian", area="south", pricerange="cheap")
    ranked = [
        RankedRestaurant(
            record={"name": "Test Restaurant", "food": "italian", "area": "south", "pricerange": "cheap"},
            score=9.0,
            matched_constraints=["food", "area", "pricerange"],
            missing_unmatched_constraints=[],
            explanation="matched food, area, pricerange; tf-idf similarity 0.500",
            similarity=0.5,
        )
    ]

    result = generator.generate("Find cheap Italian in the south", state, ranked)

    assert result.used_llm is True
    assert result.mode == "transformers:fake-generator"
    assert result.text == "I found Test Restaurant from the provided evidence."


def test_generator_uses_template_for_missing_booking_slots_even_when_llm_enabled():
    generator = GroundedResponseGenerator(enable_llm=True, model_name="fake-generator")

    class FailingPipeline:
        def __call__(self, prompt, max_new_tokens=120, do_sample=False):
            raise AssertionError("Missing-slot clarifications should not call the LLM.")

    generator._pipeline = FailingPipeline()
    state = DialogueState(
        selected_restaurant={
            "name": "pipasha restaurant",
            "food": "indian",
            "area": "east",
            "pricerange": "expensive",
        }
    )

    result = generator.generate(
        "I would like to book pipasha restaurant",
        state,
        intent="book",
        missing_slots=["day", "time", "people"],
    )

    assert result.used_llm is False
    assert result.mode == "template"
    assert "To complete the booking for pipasha restaurant" in result.text
    assert '"user":' not in result.text


def test_generator_rejects_prompt_leak_and_falls_back_to_template():
    generator = GroundedResponseGenerator(enable_llm=True, model_name="fake-generator")

    class LeakyPipeline:
        def __call__(self, prompt, max_new_tokens=120, do_sample=False):
            return [
                {
                    "generated_text": (
                        '"user": "I want to book a restaurant", '
                        '"assistant": "Matching restaurants: 1. Test Restaurant", '
                        '"timestamp": "2026-07-06T17:35:08+01:00"'
                    )
                }
            ]

    generator._pipeline = LeakyPipeline()
    state = DialogueState(food="italian", area="south", pricerange="cheap")
    ranked = [
        RankedRestaurant(
            record={"name": "Test Restaurant", "food": "italian", "area": "south", "pricerange": "cheap"},
            score=9.0,
            matched_constraints=["food", "area", "pricerange"],
            missing_unmatched_constraints=[],
            explanation="matched food, area, pricerange; tf-idf similarity 0.500",
            similarity=0.5,
        )
    ]

    result = generator.generate("Find cheap Italian in the south", state, ranked)

    assert result.used_llm is False
    assert result.mode == "template"
    assert result.text == "I found Test Restaurant (cheap italian) in the south area."
