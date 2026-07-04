from restaurant_assistant.assistant import RestaurantAssistant
from datetime import datetime
from zoneinfo import ZoneInfo


def test_assistant_search_book_reschedule_cancel_flow():
    assistant = RestaurantAssistant(use_sample=True)

    search = assistant.process("I need a cheap Italian restaurant in the south")
    assert "Pizza Hut Cherry Hinton" in search.response

    booked = assistant.process("Can you book it for Friday at 7pm for 2 people?")
    assert "simulated reference" in booked.response.lower()

    moved = assistant.process("Move it to Saturday")
    assert "updated the simulated booking" in moved.response.lower()

    cancelled = assistant.process("Cancel it")
    assert "cancelled the simulated booking" in cancelled.response.lower()


def test_assistant_handles_reported_transcript_edges():
    restaurants = [
        {
            "source_id": "r1",
            "name": "pizza hut city centre",
            "food": "italian",
            "area": "centre",
            "pricerange": "cheap",
            "address": "Regent Street City Centre",
            "postcode": "cb21ab",
            "phone": "01223323737",
        },
        {
            "source_id": "r2",
            "name": "ask restaurant",
            "food": "italian",
            "area": "centre",
            "pricerange": "cheap",
            "address": "12 Bridge Street",
            "postcode": "cb21uf",
            "phone": "01223364000",
        },
        {
            "source_id": "r3",
            "name": "kohinoor",
            "food": "indian",
            "area": "centre",
            "pricerange": "cheap",
            "address": "74 Mill Road City Centre",
            "postcode": "cb12as",
            "phone": "01223323639",
        },
    ]
    assistant = RestaurantAssistant(restaurants=restaurants)

    first = assistant.process("I need a cheap italian resturant in the city centre")
    assert "pizza hut city centre" in first.response

    alternative = assistant.process("can you tell me any othe rresutrant nearby?")
    assert "ask restaurant" in alternative.response
    assert "pizza hut city centre" not in alternative.response

    unsupported_area = assistant.process("give me a list of resturants in the countryside")
    assert "only supports these areas" in unsupported_area.response
    assert "kohinoor" not in unsupported_area.response

    off_topic = assistant.process("I want to buy a gun")
    assert "only help with MultiWOZ restaurant search" in off_topic.response
    assert "pizza hut city centre" not in off_topic.response


def test_assistant_booking_correction_and_reschedule_details():
    assistant = RestaurantAssistant(use_sample=True)

    assistant.process("I need a cheap Italian restaurant in the south")
    booked = assistant.process("can you book it for this sunday, around 9am for 4 people")
    assert "09:00" in booked.response

    correction = assistant.process("i said 9am no pm")
    assert "09:00" in correction.response

    vague = assistant.process("resechdeule the booking")
    assert "Please give a new day, time or number of people" in vague.response

    updated = assistant.process("I want to reschedule this booking. change it to 10:00")
    assert "10:00" in updated.response


def test_assistant_does_not_confuse_time_with_people_or_rebook_on_info_cancel():
    restaurants = [
        {
            "source_id": "r1",
            "name": "pizza express",
            "food": "italian",
            "area": "centre",
            "pricerange": "moderate",
            "address": "Regent Street",
            "postcode": "cb21ab",
            "phone": "01223323737",
        }
    ]
    assistant = RestaurantAssistant(restaurants=restaurants)

    assistant.process("I want to see a list of restaurant for italian cuisine")
    booked = assistant.process("I want to book pizza express for 12:00 this monday for 4 people")
    reference = assistant.state.booking_reference

    assert reference is not None
    assert "12:00" in booked.response
    assert "for 4 people" in booked.response

    info = assistant.process(f"tell me about {reference} booking.")
    assert "created a simulated booking" not in info.response.lower()
    assert reference in info.response
    assert "for 4 people" in info.response
    assert assistant.state.booking_reference == reference

    correction = assistant.process("I booked it for 4 people")
    assert reference in correction.response
    assert "for 4 people" in correction.response

    cancelled = assistant.process(f"{reference} can I cancle this booking")
    assert "cancelled the simulated booking" in cancelled.response.lower()
    assert assistant.state.booking_reference == reference
    assert assistant.state.booking_status == "cancelled"


def test_assistant_clears_stale_restaurant_context_between_searches_and_bookings():
    restaurants = [
        {
            "source_id": "r1",
            "name": "panahar",
            "food": "indian",
            "area": "centre",
            "pricerange": "expensive",
            "address": "8 Norfolk Street",
            "postcode": "cb12lf",
            "phone": "01223355012",
        },
        {
            "source_id": "r2",
            "name": "the golden curry",
            "food": "indian",
            "area": "centre",
            "pricerange": "expensive",
            "address": "10 Curry Road",
            "postcode": "cb12xx",
            "phone": "01223000000",
        },
        {
            "source_id": "r3",
            "name": "efes restaurant",
            "food": "turkish",
            "area": "centre",
            "pricerange": "moderate",
            "address": "King Street",
            "postcode": "cb11aa",
            "phone": "01223111111",
        },
        {
            "source_id": "r4",
            "name": "anatolia",
            "food": "turkish",
            "area": "centre",
            "pricerange": "moderate",
            "address": "Bridge Street",
            "postcode": "cb12bb",
            "phone": "01223222222",
        },
        {
            "source_id": "r5",
            "name": "la margherita",
            "food": "mediterranean",
            "area": "centre",
            "pricerange": "moderate",
            "address": "Magdalene Street",
            "postcode": "cb13cc",
            "phone": "01223333333",
        },
        {
            "source_id": "r6",
            "name": "clowns cafe",
            "food": "italian",
            "area": "centre",
            "pricerange": "expensive",
            "address": "King Street",
            "postcode": "cb14dd",
            "phone": "01223444444",
        },
    ]
    assistant = RestaurantAssistant(restaurants=restaurants)

    indian = assistant.process("I want to see the resturants that are indian in the area")
    assert "Matching restaurants" in indian.response
    assert "panahar" in indian.response

    more = assistant.process("anymore resturants?")
    assert "the golden curry" in more.response

    mediterranean = assistant.process("I want to see a list of the mediteranian resutrants in the area")
    assert "la margherita" in mediterranean.response
    assert "panahar" not in mediterranean.response

    turkish = assistant.process("I want to see a list of the turkish resutrants in the area")
    assert "efes restaurant" in turkish.response

    pending = assistant.process("I want to book efes resturant for tomorrow, for 5 people")
    assert "efes restaurant" in pending.response
    assert "time" in pending.response.lower()
    assert "panahar" not in pending.response

    booked = assistant.process("I want to book efes resturant for tomorrow, midnight for 5 people")
    reference = assistant.state.booking_reference
    assert "efes restaurant" in booked.response
    assert "00:00" in booked.response

    moved = assistant.process(f"tell me about {reference}, are you able to reschduele to the next tuesday?")
    assert "updated the simulated booking" in moved.response.lower()
    assert "tuesday" in moved.response.lower()

    cancelled = assistant.process(f"tell me about {reference}, are you able to cancel?")
    assert "cancelled the simulated booking" in cancelled.response.lower()

    new_booking = assistant.process(
        "book for me the most expensive italian resturant for 5 people at 7pm. provide the details of that resturant."
    )
    assert "clowns cafe" in new_booking.response
    assert "day" in new_booking.response.lower()
    assert "panahar" not in new_booking.response


def test_assistant_resolves_relative_days_from_turn_timestamp():
    fixed_turn_time = datetime(2026, 7, 4, 12, 30, tzinfo=ZoneInfo("Europe/London"))
    assistant = RestaurantAssistant(use_sample=True, clock=lambda: fixed_turn_time)

    assistant.process("I need a cheap Italian restaurant in the south")
    booked = assistant.process("book it for tomorrow at 7pm for 2 people", debug=True)

    assert "sunday" in booked.response.lower()
    assert booked.debug["turn_timestamp"].startswith("2026-07-04T12:30:00")
    assert booked.debug["relative_day_resolution"]["relative_day"] == "tomorrow"
    assert booked.debug["relative_day_resolution"]["resolved_day"] == "sunday"

    moved = assistant.process("move it to the day after")

    assert "monday" in moved.response.lower()


def test_assistant_completes_booking_details_across_turns():
    fixed_turn_time = datetime(2026, 7, 4, 12, 30, tzinfo=ZoneInfo("Europe/London"))
    restaurants = [
        {
            "source_id": "r1",
            "name": "pizza express",
            "food": "italian",
            "area": "centre",
            "pricerange": "moderate",
            "address": "Regent Street",
            "postcode": "cb21db",
            "phone": "01223324033",
        },
        {
            "source_id": "r2",
            "name": "clowns cafe",
            "food": "italian",
            "area": "centre",
            "pricerange": "expensive",
            "address": "54 King Street",
            "postcode": "cb11ln",
            "phone": "01223355711",
        },
    ]
    assistant = RestaurantAssistant(restaurants=restaurants, clock=lambda: fixed_turn_time)

    first = assistant.process("i want to book somthing for tomorrow")
    assert "choose a restaurant" in first.response.lower()

    assistant.process("italian resurant")
    assistant.process("anymore?")

    booked = assistant.process("clowns cafe for 7pm for 5 people please")
    assert "clowns cafe" in booked.response
    assert "sunday" in booked.response.lower()
    assert "19:00" in booked.response
    assert "for 5 people" in booked.response

    assistant.reset()
    assistant.process("italian resurant")
    assistant.process("I want to book clowns cafe for 7pm, for 5 people")
    completed = assistant.process("tomorrow, 7pm, 5 people")
    assert "clowns cafe" in completed.response
    assert "sunday" in completed.response.lower()
    assert "19:00" in completed.response
