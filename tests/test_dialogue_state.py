from restaurant_assistant.dialogue_state import DialogueState


def test_dialogue_state_updates_and_missing_slots():
    state = DialogueState()
    state.update_slots({"food": "Italian", "area": "center", "pricerange": "budget"})

    assert state.food == "italian"
    assert state.area == "centre"
    assert state.pricerange == "cheap"
    assert state.missing_search_slots() == []
    assert state.missing_booking_slots(include_restaurant=False) == ["day", "time", "people"]


def test_dialogue_state_reset_clears_session_data():
    state = DialogueState(food="italian", area="south", booking_status="confirmed", booking_reference="BK-ABC123")
    state.add_turn("hello", "hi")

    state.reset()

    assert state.food is None
    assert state.booking_status == "none"
    assert state.booking_reference is None
    assert state.conversation_history == []
