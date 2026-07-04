from restaurant_assistant.data_loader import load_sample_restaurants
from restaurant_assistant.dialogue_state import DialogueState
from restaurant_assistant.retrieval import RestaurantRetriever


def test_retrieval_returns_matching_restaurant():
    restaurants = load_sample_restaurants()
    retriever = RestaurantRetriever().fit(restaurants)
    state = DialogueState(food="italian", area="south", pricerange="cheap")

    results = retriever.search("cheap Italian restaurant in the south", state, top_k=3)

    assert results
    assert results[0].record["name"] == "Pizza Hut Cherry Hinton"
    assert results[0].record["food_norm"] == "italian"


def test_retrieve_by_constraints_filters_exact_slots():
    restaurants = load_sample_restaurants()
    retriever = RestaurantRetriever().fit(restaurants)
    state = DialogueState(food="chinese", area="centre", pricerange="moderate")

    results = retriever.retrieve_by_constraints(state, top_k=3)

    assert len(results) == 1
    assert results[0].record["name"] == "Yippee Noodle Bar"

