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

