from restaurant_assistant.dialogue_state import DialogueState
from restaurant_assistant.ranking import rank_candidates
from restaurant_assistant.retrieval import RetrievedRestaurant


def test_ranking_prioritises_better_matches():
    state = DialogueState(food="italian", area="south", pricerange="cheap")
    candidates = [
        RetrievedRestaurant(
            {"name": "Exact", "food": "italian", "area": "south", "pricerange": "cheap"},
            0.1,
        ),
        RetrievedRestaurant(
            {"name": "Partial", "food": "italian", "area": "centre", "pricerange": "moderate"},
            0.9,
        ),
    ]

    ranked = rank_candidates(candidates, state, top_k=2)

    assert ranked[0].record["name"] == "Exact"
    assert ranked[0].matched_constraints == ["food", "area", "pricerange"]
    assert "area" in ranked[1].missing_unmatched_constraints

