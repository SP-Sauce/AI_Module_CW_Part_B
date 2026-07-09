from restaurant_assistant.assistant import RestaurantAssistant
from restaurant_assistant.storage import BookingStore


def test_integration_search_book_reschedule_cancel():
    assistant = RestaurantAssistant(use_sample=True)

    search = assistant.process("I need a cheap Italian restaurant in the south")
    booked = assistant.process("Book it for Friday at 7pm for 2 people")
    moved = assistant.process("Move it to Saturday at 8pm")
    cancelled = assistant.process("Cancel it")

    assert "Pizza Hut Cherry Hinton" in search.response
    assert "your reference" in booked.response.casefold()
    assert "updated booking" in moved.response.casefold()
    assert "cancelled booking" in cancelled.response.casefold()


def test_integration_vague_search_requests_clarification():
    assistant = RestaurantAssistant(use_sample=True)

    result = assistant.process("I need a restaurant")

    assert "Please tell me your preferred" in result.response


def test_integration_unsupported_request_returns_safe_scope_response():
    assistant = RestaurantAssistant(use_sample=True)

    result = assistant.process("Please book me a taxi to the station")

    assert "only help with MultiWOZ restaurant search" in result.response
    assert "taxi booking" not in result.response.casefold()


def test_integration_restaurant_info_is_grounded_in_selected_record():
    assistant = RestaurantAssistant(use_sample=True)
    assistant.process("I need a cheap Italian restaurant in the south")

    result = assistant.process("What is the address of that restaurant?")

    assert "Pizza Hut Cherry Hinton" in result.response
    assert "G4 Cambridge Leisure Park" in result.response
    assert "01223 323737" in result.response


def test_integration_booking_list_and_details_are_session_only(tmp_path):
    store = BookingStore(tmp_path / "bookings.sqlite3")
    owner = RestaurantAssistant(
        use_sample=True,
        booking_store=store,
        session_id="owner-session",
    )
    outsider = RestaurantAssistant(
        use_sample=True,
        booking_store=store,
        session_id="other-session",
    )
    owner.process("I need a cheap Italian restaurant in the south")
    owner.process("Book it for Friday at 7pm for 2 people")
    reference = owner.state.booking_reference

    owner_list = owner.process("List my bookings")
    owner_details = owner.process(f"Tell me about booking {reference}")
    outsider_list = outsider.process("List my bookings")
    outsider_details = outsider.process(f"Tell me about booking {reference}")

    assert reference in owner_list.response
    assert reference in owner_details.response
    assert reference not in outsider_list.response
    assert "cannot find booking reference" in outsider_details.response
