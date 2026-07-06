from restaurant_assistant.assistant import RestaurantAssistant
from restaurant_assistant.storage import BookingStore
from datetime import datetime
from zoneinfo import ZoneInfo


def test_assistant_search_book_reschedule_cancel_flow():
    assistant = RestaurantAssistant(use_sample=True)

    search = assistant.process("I need a cheap Italian restaurant in the south")
    assert "Pizza Hut Cherry Hinton" in search.response

    booked = assistant.process("Can you book it for Friday at 7pm for 2 people?")
    assert "your reference" in booked.response.lower()

    moved = assistant.process("Move it to Saturday")
    assert "updated booking" in moved.response.lower()

    cancelled = assistant.process("Cancel it")
    assert "cancelled booking" in cancelled.response.lower()


def test_assistant_handles_distance_limit_and_full_chinese_list():
    restaurants = [
        {"source_id": "c1", "name": "tang chinese", "food": "chinese", "area": "centre", "pricerange": "expensive"},
        {
            "source_id": "c2",
            "name": "the good luck chinese food takeaway",
            "food": "chinese",
            "area": "south",
            "pricerange": "expensive",
        },
        {"source_id": "c3", "name": "hakka", "food": "chinese", "area": "north", "pricerange": "expensive"},
        {"source_id": "c4", "name": "charlie chan", "food": "chinese", "area": "centre", "pricerange": "cheap"},
        {"source_id": "c5", "name": "rice house", "food": "chinese", "area": "centre", "pricerange": "cheap"},
        {"source_id": "c6", "name": "peking restaurant", "food": "chinese", "area": "south", "pricerange": "expensive"},
    ]
    assistant = RestaurantAssistant(restaurants=restaurants)

    chinese = assistant.process("I am wondering if you could list some chinese resturants?")
    distance = assistant.process("how far is hakka from the south?")
    full_list = assistant.process("wait why is pecking not in the list?")

    assert "tang chinese" in chinese.response
    assert "exact distance or travel-time data" in distance.response
    assert "hakka is recorded in the north area" in distance.response
    assert "I found peking restaurant" not in distance.response
    assert assistant.state.area is None
    assert "peking restaurant" in full_list.response


def test_assistant_remembers_next_week_pending_booking_modifier():
    fixed_turn_time = datetime(2026, 7, 6, 12, 30, tzinfo=ZoneInfo("Europe/London"))
    restaurants = [
        {
            "source_id": "c3",
            "name": "hakka",
            "food": "chinese",
            "area": "north",
            "pricerange": "expensive",
            "address": "Milton Road Chesterton",
            "postcode": "cb41jy",
            "phone": "01223568988",
        }
    ]
    assistant = RestaurantAssistant(restaurants=restaurants, clock=lambda: fixed_turn_time)

    prompt = assistant.process("lets book hakka, for next week. for 2 people 4pm'")
    repeated = assistant.process("next week")
    booked = assistant.process("next tuesday")

    assert "Which day next week" in prompt.response
    assert "Which day next week" in repeated.response
    assert "Tuesday 14 July 2026" in booked.response
    assert "16:00" in booked.response
    assert "for 2 people" in booked.response


def test_assistant_books_typo_next_week_weekday_in_one_turn():
    fixed_turn_time = datetime(2026, 7, 6, 12, 30, tzinfo=ZoneInfo("Europe/London"))
    restaurants = [
        {
            "source_id": "p1",
            "name": "nandos",
            "food": "portuguese",
            "area": "centre",
            "pricerange": "cheap",
            "address": "33-34 Saint Andrews Street",
            "postcode": "cb23ar",
            "phone": "01223327908",
        }
    ]
    assistant = RestaurantAssistant(restaurants=restaurants, clock=lambda: fixed_turn_time)

    booked = assistant.process("sure nandos, next week thursdat, 4pm for 4 people")

    assert "Thursday 16 July 2026" in booked.response
    assert "16:00" in booked.response
    assert "for 4 people" in booked.response


def test_assistant_bulk_cancels_account_bookings_except_restaurant(tmp_path):
    fixed_turn_time = datetime(2026, 7, 6, 12, 30, tzinfo=ZoneInfo("Europe/London"))
    restaurants = [
        {
            "source_id": "p1",
            "name": "nandos",
            "food": "portuguese",
            "area": "centre",
            "pricerange": "cheap",
        },
        {
            "source_id": "c1",
            "name": "hakka",
            "food": "chinese",
            "area": "north",
            "pricerange": "expensive",
        },
        {
            "source_id": "m1",
            "name": "ali baba",
            "food": "lebanese",
            "area": "centre",
            "pricerange": "moderate",
        },
    ]
    store = BookingStore(tmp_path / "bookings.sqlite3")
    user = store.create_user("demo-user", "hash", display_name="Demo User")
    store.ensure_session("keep-session", user_id=user["id"])
    store.ensure_session("other-session", user_id=user["id"])

    keep = RestaurantAssistant(
        restaurants=restaurants,
        booking_store=store,
        session_id="keep-session",
        user_id=user["id"],
        clock=lambda: fixed_turn_time,
    )
    other = RestaurantAssistant(
        restaurants=restaurants,
        booking_store=store,
        session_id="other-session",
        user_id=user["id"],
        clock=lambda: fixed_turn_time,
    )
    keep.process("book nandos for next week thursday at 4pm for 4 people")
    kept_reference = keep.state.booking_reference
    other.process("book hakka for tomorrow at 5pm for 2 people")
    hakka_reference = other.state.booking_reference
    other.process("book ali baba for tomorrow at 8pm for 5 people")
    ali_reference = other.state.booking_reference

    cancelled = keep.process("can you cancel all my bookings apart from nandos?")
    bookings = {booking["reference"]: booking for booking in store.list_user_bookings(user["id"])}

    assert "I cancelled 2 booking records" in cancelled.response
    assert str(hakka_reference) in cancelled.response
    assert str(ali_reference) in cancelled.response
    assert str(kept_reference) in cancelled.response
    assert bookings[kept_reference]["status"] == "confirmed"
    assert bookings[hakka_reference]["status"] == "cancelled"
    assert bookings[ali_reference]["status"] == "cancelled"


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
    assert "created a booking record" not in info.response.lower()
    assert reference in info.response
    assert "for 4 people" in info.response
    assert assistant.state.booking_reference == reference

    correction = assistant.process("I booked it for 4 people")
    assert reference in correction.response
    assert "for 4 people" in correction.response

    cancelled = assistant.process(f"{reference} can I cancle this booking")
    assert "cancelled booking" in cancelled.response.lower()
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
    assert "updated booking" in moved.response.lower()
    assert "tuesday" in moved.response.lower()

    cancelled = assistant.process(f"tell me about {reference}, are you able to cancel?")
    assert "cancelled booking" in cancelled.response.lower()

    new_booking = assistant.process(
        "book for me the most expensive italian resturant for 5 people at 7pm. provide the details of that resturant."
    )
    assert "clowns cafe" in new_booking.response
    assert "day" in new_booking.response.lower()
    assert "panahar" not in new_booking.response


def test_assistant_answers_restaurant_details_and_broadens_cuisine_context():
    fixed_turn_time = datetime(2026, 7, 4, 12, 30, tzinfo=ZoneInfo("Europe/London"))
    restaurants = [
        {
            "source_id": "r1",
            "name": "efes restaurant",
            "food": "turkish",
            "area": "centre",
            "pricerange": "moderate",
            "address": "King Street City Centre",
            "postcode": "cb11ln",
            "phone": "01223500005",
        },
        {
            "source_id": "r2",
            "name": "anatolia",
            "food": "turkish",
            "area": "centre",
            "pricerange": "moderate",
            "address": "30 Bridge Street City Centre",
            "postcode": "cb21uj",
            "phone": "01223362372",
        },
        {
            "source_id": "r3",
            "name": "meze bar",
            "food": "turkish",
            "area": "centre",
            "pricerange": "expensive",
            "address": "196 Mill Road City Centre",
            "postcode": "cb13nf",
            "phone": "",
        },
        {
            "source_id": "r4",
            "name": "the oak bistro",
            "food": "british",
            "area": "centre",
            "pricerange": "moderate",
            "address": "6 Lensfield Road",
            "postcode": "cb21eg",
            "phone": "01223323361",
        },
        {
            "source_id": "r5",
            "name": "bedouin",
            "food": "african",
            "area": "centre",
            "pricerange": "moderate",
            "address": "100 Mill Road",
            "postcode": "cb12bd",
            "phone": "01223367660",
        },
    ]
    assistant = RestaurantAssistant(restaurants=restaurants, clock=lambda: fixed_turn_time)

    turkish = assistant.process("list all the turkish resturants")
    assert "meze bar" in turkish.response

    booked = assistant.process("i want to book meze bar for tomorrow, at 8pm for 10 people")
    assert "Sunday 5 July 2026" in booked.response

    address = assistant.process("what is the address?")
    assert "meze bar" in address.response
    assert "196 Mill Road City Centre" in address.response

    named_address = assistant.process("what is the address of the meze bar?")
    assert "196 Mill Road City Centre" in named_address.response

    booking_address = assistant.process("what is the address of my booking?")
    assert "meze bar" in booking_address.response
    assert "196 Mill Road City Centre" in booking_address.response

    moderate = assistant.process("can you list all resturants that will be in the moderate price range")
    assert "the oak bistro" in moderate.response
    assert "bedouin" in moderate.response
    assert "meze bar" not in moderate.response

    not_just = assistant.process(
        "can you list all resturants, not just turkish, that will be in the moderate price range"
    )
    assert "the oak bistro" in not_just.response
    assert "bedouin" in not_just.response
    assert "efes restaurant" not in not_just.response
    assert "anatolia" not in not_just.response

    other_cuisines = assistant.process("list resturants from other cusines")
    assert other_cuisines.response.startswith("Matching restaurants:")
    assert "the oak bistro" in other_cuisines.response


def test_assistant_handles_price_followups_area_filters_and_long_list_requests():
    restaurants = [
        {
            "source_id": "r1",
            "name": "cheap indian",
            "food": "indian",
            "area": "centre",
            "pricerange": "cheap",
            "address": "1 Cheap Street",
            "postcode": "cb11aa",
            "phone": "01223000001",
        },
        {
            "source_id": "r2",
            "name": "moderate italian",
            "food": "italian",
            "area": "centre",
            "pricerange": "moderate",
            "address": "2 Moderate Street",
            "postcode": "cb11bb",
            "phone": "01223000002",
        },
        {
            "source_id": "r3",
            "name": "moderate british",
            "food": "british",
            "area": "north",
            "pricerange": "moderate",
            "address": "3 North Road",
            "postcode": "cb11cc",
            "phone": "01223000003",
        },
        {
            "source_id": "r4",
            "name": "expensive french",
            "food": "french",
            "area": "west",
            "pricerange": "expensive",
            "address": "4 West Road",
            "postcode": "cb11dd",
            "phone": "01223000004",
        },
    ]
    assistant = RestaurantAssistant(restaurants=restaurants)

    all_restaurants = assistant.process("list all the resuratns in the area")
    assert "Matching restaurants" in all_restaurants.response
    assert "cheap indian" in all_restaurants.response

    price_followup = assistant.process("that are around 10-15")
    assert "moderate italian" in price_followup.response
    assert "moderate british" in price_followup.response
    assert "cheap indian" not in price_followup.response

    typo_price = assistant.process("list all resuratns that priced around £150-!5 pounds")
    assert "moderate italian" in typo_price.response

    assistant.process("cheap to moderately priced")
    moderate_list = assistant.process("moderatley priced resturants list all")
    assert "moderate italian" in moderate_list.response
    assert "cheap indian" not in moderate_list.response

    all_of_them = assistant.process("all of them")
    assert "moderate british" in all_of_them.response

    filters = assistant.process("what areas are there to filter through?")
    assert "centre, north, south, east and west" in filters.response

    near_me = assistant.process("list resturants near me")
    assert "only supports these areas" in near_me.response


def test_assistant_handles_cuisine_discovery_menu_items_and_negated_food():
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
            "name": "the cow pizza kitchen and bar",
            "food": "gastropub",
            "area": "centre",
            "pricerange": "moderate",
            "address": "Corn Exchange Street",
            "postcode": "cb23qf",
            "phone": "01223308000",
        },
        {
            "source_id": "r4",
            "name": "ali baba",
            "food": "lebanese",
            "area": "centre",
            "pricerange": "moderate",
            "address": "59 Hills Road City Centre",
            "postcode": "cb21nt",
            "phone": "01462432565",
        },
        {
            "source_id": "r5",
            "name": "kohinoor",
            "food": "indian",
            "area": "centre",
            "pricerange": "cheap",
            "address": "74 Mill Road City Centre",
            "postcode": "cb12as",
            "phone": "01223323639",
        },
        {
            "source_id": "r6",
            "name": "cote",
            "food": "french",
            "area": "centre",
            "pricerange": "expensive",
            "address": "21 Bridge Street",
            "postcode": "cb21uf",
            "phone": "01223311999",
        },
    ]
    assistant = RestaurantAssistant(restaurants=restaurants)

    help_response = assistant.process(
        "hello i am interested in booking a resturant but I am unsure what to eat. can you suggest some nice cuisines?"
    )
    assert "Available cuisines include" in help_response.response
    assert "Italian" in help_response.response
    assert "Lebanese" in help_response.response

    options = assistant.process("I want to know what food cuisines I can look for, then I will choose")
    assert "Available cuisines include" in options.response

    pizza = assistant.process("i like pizza")
    assert "pizza express" in pizza.response

    spaghetti = assistant.process("actually I like spaghetti")
    assert "italian" in spaghetti.response.lower()

    dish = assistant.process("no I like chicken and rice where can I find some resurants that serve that?")
    assert "dish-level menu data" in dish.response
    assert "Matching restaurants" in dish.response
    assert "kohinoor" in dish.response or "ali baba" in dish.response
    assert "pizza express" not in dish.response
    assert assistant.state.food is None

    cake = assistant.process("I like cake, can you suggest somewhere?")
    assert "dish-level menu data" in cake.response
    assert "cote" in cake.response

    arab = assistant.process("anything arab?")
    assert "ali baba" in arab.response
    assert "lebanese" in arab.response.lower()

    arabic = assistant.process("arabic food?")
    assert "ali baba" in arabic.response

    alternatives = assistant.process("no pizza, list other resurants in city center")
    assert "pizza express" not in alternatives.response
    assert "ask restaurant" not in alternatives.response
    assert "the cow pizza kitchen and bar" not in alternatives.response
    assert "ali baba" in alternatives.response or "kohinoor" in alternatives.response

    arab_list = assistant.process("list all arab resurants")
    assert "ali baba" in arab_list.response

    not_italian = assistant.process("not italion other cuisines")
    assert "pizza express" not in not_italian.response
    assert "ask restaurant" not in not_italian.response
    assert "kohinoor" in not_italian.response or "cote" in not_italian.response


def test_assistant_keeps_middle_eastern_and_price_filters_grounded():
    restaurants = [
        {
            "source_id": "r1",
            "name": "ali baba",
            "food": "lebanese",
            "area": "centre",
            "pricerange": "moderate",
            "address": "59 Hills Road City Centre",
            "postcode": "cb21nt",
            "phone": "01462432565",
        },
        {
            "source_id": "r2",
            "name": "efes restaurant",
            "food": "turkish",
            "area": "centre",
            "pricerange": "moderate",
            "address": "King Street City Centre",
            "postcode": "cb11aa",
            "phone": "01223111111",
        },
        {
            "source_id": "r3",
            "name": "the gardenia",
            "food": "mediterranean",
            "area": "centre",
            "pricerange": "cheap",
            "address": "2 Rose Street",
            "postcode": "cb12aa",
            "phone": "01223222222",
        },
        {
            "source_id": "r4",
            "name": "pipasha restaurant",
            "food": "indian",
            "area": "east",
            "pricerange": "expensive",
            "address": "4 East Road",
            "postcode": "cb13aa",
            "phone": "01223333333",
        },
        {
            "source_id": "r5",
            "name": "rajmahal",
            "food": "indian",
            "area": "east",
            "pricerange": "moderate",
            "address": "5 East Road",
            "postcode": "cb14aa",
            "phone": "01223444444",
        },
        {
            "source_id": "r6",
            "name": "royal standard",
            "food": "gastropub",
            "area": "east",
            "pricerange": "expensive",
            "address": "6 East Road",
            "postcode": "cb15aa",
            "phone": "01223555555",
        },
        {
            "source_id": "r7",
            "name": "pizza express",
            "food": "italian",
            "area": "centre",
            "pricerange": "moderate",
            "address": "Regent Street",
            "postcode": "cb21db",
            "phone": "01223324033",
        },
    ]
    assistant = RestaurantAssistant(restaurants=restaurants)

    arabic = assistant.process(
        "I am looking to book a resturant that serves dishes like lamb and rice, chicken and rice, or lamd mandhi. what arabic resturants are there?"
    )
    assert arabic.response.startswith("Matching restaurants:")
    assert "ali baba" in arabic.response
    assert "To complete the booking" not in arabic.response

    tell_assistant = RestaurantAssistant(restaurants=restaurants)
    tell_middle_eastern = tell_assistant.process("can you tell me about any middle eastern resturants?")
    assert "Middle Eastern is not a direct cuisine label" in tell_middle_eastern.response
    assert "efes restaurant" in tell_middle_eastern.response
    assert "pipasha restaurant" not in tell_middle_eastern.response

    middle_eastern = assistant.process("can you list other middle eastern resturants?")
    assert "Middle Eastern is not a direct cuisine label" in middle_eastern.response
    assert "efes restaurant" in middle_eastern.response
    assert "the gardenia" in middle_eastern.response
    assert "ali baba" not in middle_eastern.response
    assert "pipasha restaurant" not in middle_eastern.response
    assert "royal standard" not in middle_eastern.response

    mixed = assistant.process("any other middle eastern resurants like lebanese, egyption?")
    assert "efes restaurant" in mixed.response
    assert "pipasha restaurant" not in mixed.response

    lebanese = assistant.process("list all lebanese resurants please")
    assert "ali baba" in lebanese.response
    assert "efes restaurant" not in lebanese.response
    assert "rajmahal" not in lebanese.response

    egyptian = assistant.process("egyptian?")
    assert "do not have Egyptian" in egyptian.response

    moderate = assistant.process("can you list all moderatly priced resurants please?")
    assert "pizza express" in moderate.response
    assert "efes restaurant" in moderate.response
    assert "pipasha restaurant" not in moderate.response
    assert "royal standard" not in moderate.response


def test_assistant_lists_only_current_session_bookings(tmp_path):
    fixed_turn_time = datetime(2026, 7, 6, 12, 30, tzinfo=ZoneInfo("Europe/London"))
    restaurants = [
        {
            "source_id": "r1",
            "name": "shiraz restaurant",
            "food": "mediterranean",
            "area": "centre",
            "pricerange": "expensive",
            "address": "84 Regent Street City Centre",
            "postcode": "cb21dp",
            "phone": "01223307581",
        },
        {
            "source_id": "r2",
            "name": "pizza express",
            "food": "italian",
            "area": "centre",
            "pricerange": "moderate",
            "address": "Regent Street",
            "postcode": "cb21db",
            "phone": "01223324033",
        },
    ]
    store = BookingStore(tmp_path / "bookings.sqlite3")
    first = RestaurantAssistant(restaurants=restaurants, booking_store=store, session_id="session-one", clock=lambda: fixed_turn_time)
    second = RestaurantAssistant(restaurants=restaurants, booking_store=store, session_id="session-two", clock=lambda: fixed_turn_time)

    first.process("book shiraz restaurant for tomorrow at 10pm for 5 people")
    first_reference = first.state.booking_reference
    duplicate = first.process("can you create another booking for the same resturant, same people, same time")
    duplicate_reference = first.state.booking_reference
    second.process("book pizza express for tomorrow at 6pm for 2 people")
    second_reference = second.state.booking_reference

    assert duplicate_reference != first_reference
    assert "shiraz restaurant" in duplicate.response
    assert "Tuesday 7 July 2026" in duplicate.response
    assert "22:00" in duplicate.response
    assert "for 5 people" in duplicate.response

    listed = first.process("can you list all bookings")
    assert "Current session booking records" in listed.response
    assert first_reference in listed.response
    assert duplicate_reference in listed.response
    assert "shiraz restaurant" in listed.response
    assert second_reference not in listed.response
    assert "pizza express" not in listed.response

    question = first.process("what bookings are there")
    assert first_reference in question.response
    assert duplicate_reference in question.response
    assert second_reference not in question.response

    have_booked = first.process("what bookings have i booked?")
    assert first_reference in have_booked.response
    assert duplicate_reference in have_booked.response
    assert second_reference not in have_booked.response

    table_followup = first.process("as a tbale")
    assert "Current session booking records" in table_followup.response
    assert first_reference in table_followup.response
    assert duplicate_reference in table_followup.response
    assert second_reference not in table_followup.response

    thanks = first.process("thank you!")
    assert "You're welcome" in thanks.response


def test_assistant_resolves_booking_reference_only_within_current_session(tmp_path):
    fixed_turn_time = datetime(2026, 7, 6, 12, 30, tzinfo=ZoneInfo("Europe/London"))
    restaurants = [
        {
            "source_id": "r1",
            "name": "efes restaurant",
            "food": "turkish",
            "area": "centre",
            "pricerange": "moderate",
            "address": "King Street City Centre",
            "postcode": "cb11ln",
            "phone": "01223500005",
        }
    ]
    store = BookingStore(tmp_path / "bookings.sqlite3")
    owner = RestaurantAssistant(restaurants=restaurants, booking_store=store, session_id="owner", clock=lambda: fixed_turn_time)
    owner.process("book efes restaurant for tomorrow at 5pm for 5 people")
    reference = owner.state.booking_reference

    outsider = RestaurantAssistant(restaurants=restaurants, booking_store=store, session_id="outsider", clock=lambda: fixed_turn_time)
    info = outsider.process(f"can you tell me about {reference}")
    move = outsider.process(f"I want to reschedule {reference}")

    assert f"cannot find booking reference {reference} in this current session" in info.response
    assert f"cannot find booking reference {reference} in this current session" in move.response
    assert "efes restaurant" not in info.response

    restored = RestaurantAssistant(restaurants=restaurants, booking_store=store, session_id="owner", clock=lambda: fixed_turn_time)
    info_current = restored.process(f"can you tell me about {reference}")
    moved_current = restored.process(f"I want to reschedule {reference} to 6pm")

    assert reference in info_current.response
    assert "efes restaurant" in info_current.response
    assert reference in moved_current.response
    assert "18:00" in moved_current.response


def test_assistant_resolves_relative_days_from_turn_timestamp():
    fixed_turn_time = datetime(2026, 7, 4, 12, 30, tzinfo=ZoneInfo("Europe/London"))
    assistant = RestaurantAssistant(use_sample=True, clock=lambda: fixed_turn_time)

    assistant.process("I need a cheap Italian restaurant in the south")
    booked = assistant.process("book it for tomorrow at 7pm for 2 people", debug=True)

    assert "Sunday 5 July 2026" in booked.response
    assert booked.debug["turn_timestamp"].startswith("2026-07-04T12:30:00")
    assert booked.debug["relative_day_resolution"]["relative_day"] == "tomorrow"
    assert booked.debug["relative_day_resolution"]["resolved_day"] == "sunday"
    assert booked.debug["relative_day_resolution"]["booking_date"] == "2026-07-05"

    moved = assistant.process("move it to the day after")

    assert "Monday 6 July 2026" in moved.response

    next_week = assistant.process("reschedule it to next week Sunday")

    assert "Sunday 12 July 2026" in next_week.response
    assert assistant.state.booking_date == "2026-07-12"

    info = assistant.process("tell me about it")

    assert assistant.state.booking_reference in info.response
    assert "Sunday 12 July 2026" in info.response


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
    assert "Sunday 5 July 2026" in booked.response
    assert "19:00" in booked.response
    assert "for 5 people" in booked.response

    assistant.reset()
    assistant.process("italian resurant")
    assistant.process("I want to book clowns cafe for 7pm, for 5 people")
    completed = assistant.process("tomorrow, 7pm, 5 people")
    assert "clowns cafe" in completed.response
    assert "Sunday 5 July 2026" in completed.response
    assert "19:00" in completed.response
