from restaurant_assistant.booking import BookingManager
from restaurant_assistant.dialogue_state import DialogueState


def test_simulated_booking_confirmation_reschedule_and_cancel():
    state = DialogueState(
        day="friday",
        time="19:00",
        people=2,
        selected_restaurant={"name": "Pizza Hut Cherry Hinton"},
    )
    manager = BookingManager()

    created = manager.create_booking(state)
    assert created.success
    assert state.booking_reference is not None
    assert state.booking_reference.startswith("SIM-")
    assert "simulated booking" in created.message.lower()
    assert "not a live restaurant booking" in created.message.lower()

    state.update_slots({"day": "saturday"})
    rescheduled = manager.reschedule_booking(state)
    assert rescheduled.success
    assert "saturday" in rescheduled.message.lower()

    cancelled = manager.cancel_booking(state)
    assert cancelled.success
    assert state.booking_status == "cancelled"


def test_booking_reports_missing_slots():
    state = DialogueState(selected_restaurant={"name": "Aromi"})
    result = BookingManager().create_booking(state)

    assert not result.success
    assert result.missing_slots == ["day", "time", "people"]

