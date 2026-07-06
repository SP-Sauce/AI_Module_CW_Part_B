from restaurant_assistant.booking import BookingManager
from restaurant_assistant.dialogue_state import DialogueState


def test_booking_confirmation_reschedule_and_cancel():
    state = DialogueState(
        day="friday",
        booking_date="2026-07-10",
        time="19:00",
        people=2,
        selected_restaurant={"name": "Pizza Hut Cherry Hinton"},
    )
    manager = BookingManager()

    created = manager.create_booking(state)
    assert created.success
    assert state.booking_reference is not None
    assert state.booking_reference.startswith("BK-")
    assert "booking record" in created.message.lower()
    assert "Friday 10 July 2026" in created.message

    state.update_slots({"day": "saturday", "booking_date": "2026-07-11"})
    rescheduled = manager.reschedule_booking(state)
    assert rescheduled.success
    assert "Saturday 11 July 2026" in rescheduled.message

    cancelled = manager.cancel_booking(state)
    assert cancelled.success
    assert state.booking_status == "cancelled"


def test_booking_reports_missing_slots():
    state = DialogueState(selected_restaurant={"name": "Aromi"})
    result = BookingManager().create_booking(state)

    assert not result.success
    assert result.missing_slots == ["day", "time", "people"]
