from restaurant_assistant.booking import BookingManager
from restaurant_assistant.dialogue_state import DialogueState
from restaurant_assistant.storage import BookingStore


def test_booking_store_persists_create_reschedule_and_cancel(tmp_path):
    store = BookingStore(tmp_path / "bookings.sqlite3")
    session_id = "session-1"
    state = DialogueState(
        day="friday",
        booking_date="2026-07-10",
        time="19:00",
        people=2,
        selected_restaurant={
            "name": "Pizza Hut Cherry Hinton",
            "food": "italian",
            "area": "south",
            "pricerange": "cheap",
            "address": "G4 Cambridge Leisure Park",
            "postcode": "CB1 7DY",
            "phone": "01223 323737",
        },
    )
    manager = BookingManager(store=store, session_id=session_id)

    created = manager.create_booking(state)

    assert created.success
    bookings = store.list_bookings(session_id)
    assert len(bookings) == 1
    assert bookings[0]["reference"] == state.booking_reference
    assert bookings[0]["booking_date"] == "2026-07-10"
    assert bookings[0]["status"] == "confirmed"

    state.update_slots({"day": "saturday", "booking_date": "2026-07-11"})
    manager.reschedule_booking(state)

    bookings = store.list_bookings(session_id)
    assert bookings[0]["booking_date"] == "2026-07-11"
    assert bookings[0]["day"] == "saturday"

    manager.cancel_booking(state)

    bookings = store.list_bookings(session_id)
    assert bookings[0]["status"] == "cancelled"
    assert store.get_booking(session_id, state.booking_reference)["status"] == "cancelled"
    assert store.get_booking("other-session", state.booking_reference) is None


def test_booking_store_saves_chat_turns(tmp_path):
    store = BookingStore(tmp_path / "bookings.sqlite3")
    store.save_turn("session-1", "hello", "Hello. I can help find MultiWOZ restaurant records.")

    turns = store.list_turns("session-1")

    assert turns == [
        {
            "user_message": "hello",
            "assistant_message": "Hello. I can help find MultiWOZ restaurant records.",
            "created_at": turns[0]["created_at"],
        }
    ]

    transcript = store.export_session_text("session-1")

    assert "Session: session-1" in transcript
    assert "You: hello" in transcript
    assert "Assistant: Hello. I can help find MultiWOZ restaurant records." in transcript


def test_booking_store_users_and_scoped_history(tmp_path):
    store = BookingStore(tmp_path / "bookings.sqlite3")
    user = store.create_user("DemoUser", "hash", display_name="Demo User")
    other = store.create_user("OtherUser", "hash", display_name="Other User")

    store.ensure_session("session-1", user_id=user["id"])
    store.ensure_session("session-2", user_id=other["id"])
    store.save_turn("session-1", "hello", "Hello.")
    store.save_turn("session-2", "hi", "Hello.")

    user_sessions = store.list_user_sessions(user["id"])
    other_sessions = store.list_user_sessions(other["id"])

    assert store.get_user_by_username("demouser")["display_name"] == "Demo User"
    assert [session["session_id"] for session in user_sessions] == ["session-1"]
    assert [session["session_id"] for session in other_sessions] == ["session-2"]


def test_booking_store_admin_snapshot_includes_metrics(tmp_path):
    store = BookingStore(tmp_path / "bookings.sqlite3")
    store.save_turn(
        "session-1",
        "book it",
        "Please provide: time.",
        metadata={"intent": "book", "effective_intent": "book", "slots": {"food": "italian"}},
        latency_ms=12.5,
    )
    state = DialogueState(
        day="friday",
        booking_date="2026-07-10",
        time="19:00",
        people=2,
        selected_restaurant={"name": "Pizza Hut Cherry Hinton", "food": "italian"},
    )
    BookingManager(store=store, session_id="session-1").create_booking(state)

    turns = store.list_turns("session-1", include_metadata=True)
    snapshot = store.admin_snapshot()

    assert turns[0]["intent"] == "book"
    assert turns[0]["slots"] == {"food": "italian"}
    assert snapshot["summary"]["total_sessions"] == 1
    assert snapshot["summary"]["active_sessions"] == 1
    assert snapshot["summary"]["closed_sessions"] == 0
    assert snapshot["summary"]["total_turns"] == 1
    assert snapshot["summary"]["total_bookings"] == 1
    assert snapshot["summary"]["average_latency_ms"] == 12.5
    assert ("book", 1) in snapshot["intent_counts"]


def test_booking_store_closes_and_deletes_sessions(tmp_path):
    store = BookingStore(tmp_path / "bookings.sqlite3")
    store.save_turn("session-1", "hello", "Hello.")
    store.save_turn("session-2", "hi", "Hello.")

    closed_count = store.close_sessions(["session-1"])
    closed = store.get_session("session-1")
    snapshot = store.admin_snapshot()

    assert closed_count == 1
    assert closed["status"] == "closed"
    assert closed["closed_at"]
    assert snapshot["summary"]["active_sessions"] == 1
    assert snapshot["summary"]["closed_sessions"] == 1

    closed_count = store.close_all_sessions()
    snapshot = store.admin_snapshot()

    assert closed_count == 1
    assert snapshot["summary"]["active_sessions"] == 0
    assert snapshot["summary"]["closed_sessions"] == 2

    store.delete_session("session-1")

    assert store.get_session("session-1") is None
    assert store.list_turns("session-1") == []
