import json
from pathlib import Path

from restaurant_assistant.slot_extraction import extract_slots


def test_slot_fixture_cases():
    cases = json.loads(Path("tests/fixtures/slot_cases.json").read_text(encoding="utf-8"))

    for case in cases:
        result = extract_slots(case["text"])
        assert result.intent == case["intent"]
        for key, expected in case["slots"].items():
            assert result.slots[key] == expected


def test_unsupported_values_are_not_added():
    result = extract_slots("Find me a Martian restaurant in the moon district")

    assert "food" not in result.slots
    assert "area" not in result.slots


def test_transcript_typo_and_booking_intents():
    cases = [
        ("I need a cheap italian resturant in the city centre", "search", {"area": "centre"}),
        ("can you tell me any othe rresutrant nearby?", "alternative", {}),
        ("resechdeule the booking", "reschedule", {}),
        ("I want to reschedule this booking: SIM-OGTV7J. change it to 10:00", "reschedule", {"time": "10:00"}),
        ("cancel SIM-OGTV7J.", "cancel", {"booking_reference": "SIM-OGTV7J"}),
        ("I want to buy a gun", "unsupported", {}),
    ]

    for text, expected_intent, expected_slots in cases:
        result = extract_slots(text)
        assert result.intent == expected_intent
        for key, expected_value in expected_slots.items():
            assert result.slots[key] == expected_value


def test_unsupported_area_is_reported():
    result = extract_slots("give me a list of resturants in the countryside")

    assert result.intent == "list"
    assert result.unsupported_slots == {"area": "countryside"}


def test_am_time_is_not_converted_to_pm():
    result = extract_slots("book it for this sunday, around 9am for 4 people")

    assert result.slots["time"] == "09:00"


def test_time_is_not_misread_as_people_count():
    result = extract_slots("I want to book pizza express for 12:00 this monday for 4 people")

    assert result.intent == "book"
    assert result.slots["time"] == "12:00"
    assert result.slots["people"] == 4


def test_midnight_and_booking_reference_intents():
    midnight = extract_slots("I want to book pizza express for midnight this monday for 4 people")
    info = extract_slots("tell me about SIM-Z8L84T booking.")
    list_bookings = extract_slots("can you list all bookings")
    question_bookings = extract_slots("what bookings are there")
    have_booked = extract_slots("what bookings have i booked?")
    cancel = extract_slots("SIM-ROFNYN can I cancle this booking")
    bk_cancel = extract_slots("cancel BK-ABC123")
    bulk_cancel = extract_slots("can you cancel all my bookings apart from nandos?")

    assert midnight.slots["time"] == "00:00"
    assert info.intent == "booking_info"
    assert list_bookings.intent == "booking_list"
    assert question_bookings.intent == "booking_list"
    assert have_booked.intent == "booking_list"
    assert cancel.intent == "cancel"
    assert cancel.slots["booking_reference"] == "SIM-ROFNYN"
    assert bk_cancel.intent == "cancel"
    assert bk_cancel.slots["booking_reference"] == "BK-ABC123"
    assert bulk_cancel.intent == "cancel"


def test_list_alternative_and_food_typo_intents():
    indian_list = extract_slots("I want to see the resturants that are indian in the area")
    more = extract_slots("anymore resturants?")
    mediterranean = extract_slots("I want to see a list of the mediteranian resutrants in the area")
    other_cuisines = extract_slots("list resturants from other cusines")
    not_just_turkish = extract_slots("list all resturants, not just turkish, in the moderate price range")

    assert indian_list.intent == "list"
    assert indian_list.slots["food"] == "indian"
    assert more.intent == "alternative"
    assert mediterranean.intent == "list"
    assert mediterranean.slots["food"] == "mediterranean"
    assert other_cuisines.intent == "list"
    assert not_just_turkish.intent == "list"


def test_thanks_is_extracted():
    result = extract_slots("thank you!")

    assert result.intent == "thanks"


def test_polite_tail_does_not_override_search_intent():
    result = extract_slots("Need summat cheap and Italian out east, ta")

    assert result.intent == "search"
    assert result.slots == {"food": "italian", "area": "east", "pricerange": "cheap"}


def test_greeting_typos_are_extracted():
    result = extract_slots("hellow")

    assert result.intent == "greeting"


def test_table_followup_typo_is_extracted():
    result = extract_slots("as a tbale")

    assert result.intent == "table_view"


def test_price_range_and_filter_info_intents_are_extracted():
    range_followup = extract_slots("that are around 10-15")
    typed_price = extract_slots("list all resuratns that priced around £150-!5 pounds")
    misspelled_moderate = extract_slots("moderatley priced resturants list all")
    all_of_them = extract_slots("all of them")
    areas = extract_slots("what areas are there to filter through?")
    near_me = extract_slots("list resturants near me")

    assert range_followup.intent == "list"
    assert range_followup.slots["pricerange"] == "moderate"
    assert typed_price.intent == "list"
    assert typed_price.slots["pricerange"] == "moderate"
    assert misspelled_moderate.intent == "list"
    assert misspelled_moderate.slots["pricerange"] == "moderate"
    assert all_of_them.intent == "list"
    assert areas.intent == "filter_info"
    assert near_me.unsupported_slots == {"area": "near me"}


def test_cuisine_help_aliases_and_menu_item_limits_are_extracted():
    cuisine_help = extract_slots(
        "hello i am interested in booking a resturant but I am unsure what to eat. can you suggest some nice cuisines?"
    )
    cuisine_options = extract_slots("I want to know what food cuisines I can look for, then I will choose")
    spaghetti = extract_slots("actually I like spaghetti")
    chicken_rice = extract_slots("no I like chicken and rice where can I find some resurants that serve that?")
    lamb_mandi = extract_slots("I like lamb and rice or lamd mandhi")
    cake = extract_slots("I like cake, can you suggest somewhere?")
    arab = extract_slots("anything arab?")
    arabic = extract_slots("arabic food?")
    middle_eastern = extract_slots("can you list other middle eastern resturants?")
    tell_middle_eastern = extract_slots("can you tell me about any middle eastern resturants?")
    south_asian = extract_slots("can you show me south asian restaurants?")
    south_asian_west = extract_slots("any south asian food in the west?")
    east_asian = extract_slots("list east asian restaurants please")
    southeast_asian = extract_slots("show me south east asian restaurants")
    mixed_booking_list = extract_slots(
        "I am looking to book a resturant that serves lamb and rice. what arabic resturants are there?"
    )
    duplicate_booking = extract_slots("can you create another booking for the same resturant, same people, same time")
    egyptian = extract_slots("egyption?")
    no_pizza = extract_slots("no pizza, list other resurants in city center")
    not_italian = extract_slots("not italion other cuisines")
    west_african = extract_slots("can tell about west african resutrants in the area?")
    north_american = extract_slots("can you find north american restaurants?")
    moroccan = extract_slots("im looking for morccan resturants")

    assert cuisine_help.intent == "cuisine_help"
    assert cuisine_options.intent == "cuisine_help"
    assert spaghetti.slots["food"] == "italian"
    assert chicken_rice.intent == "dish_preference"
    assert chicken_rice.slots["dish"] == "chicken and rice"
    assert "indian" in chicken_rice.slots["food_candidates"]
    assert "lebanese" in chicken_rice.slots["food_candidates"]
    assert lamb_mandi.intent == "dish_preference"
    assert "lebanese" in lamb_mandi.slots["food_candidates"]
    assert cake.intent == "dish_preference"
    assert cake.slots["dish"] == "cake or dessert"
    assert "british" in cake.slots["food_candidates"]
    assert arab.slots["cuisine_group"] == "Middle Eastern"
    assert arab.slots["food_candidates"] == ["lebanese", "turkish", "mediterranean"]
    assert arabic.slots["cuisine_group"] == "Middle Eastern"
    assert middle_eastern.intent == "alternative"
    assert middle_eastern.slots["cuisine_group"] == "Middle Eastern"
    assert middle_eastern.slots["food_candidates"] == ["lebanese", "turkish", "mediterranean"]
    assert "area" not in middle_eastern.slots
    assert tell_middle_eastern.intent == "list"
    assert tell_middle_eastern.slots["cuisine_group"] == "Middle Eastern"
    assert south_asian.intent == "list"
    assert south_asian.slots["cuisine_group"] == "South Asian"
    assert south_asian.slots["food_candidates"] == ["indian"]
    assert "area" not in south_asian.slots
    assert south_asian_west.slots["cuisine_group"] == "South Asian"
    assert south_asian_west.slots["area"] == "west"
    assert east_asian.slots["cuisine_group"] == "East Asian"
    assert "chinese" in east_asian.slots["food_candidates"]
    assert "area" not in east_asian.slots
    assert southeast_asian.slots["cuisine_group"] == "Southeast Asian"
    assert "thai" in southeast_asian.slots["food_candidates"]
    assert "vietnamese" in southeast_asian.slots["food_candidates"]
    assert mixed_booking_list.intent == "list"
    assert mixed_booking_list.slots["cuisine_group"] == "Middle Eastern"
    assert duplicate_booking.intent == "book"
    assert egyptian.unsupported_slots == {"food": "egyptian"}
    assert no_pizza.intent == "alternative"
    assert no_pizza.slots["food"] == "italian"
    assert no_pizza.slots["area"] == "centre"
    assert not_italian.intent == "list"
    assert not_italian.slots["food"] == "italian"
    assert west_african.intent == "list"
    assert west_african.slots["cuisine_group"] == "West African"
    assert west_african.slots["food_candidates"] == ["african"]
    assert "area" not in west_african.slots
    assert north_american.slots["food"] == "north american"
    assert "area" not in north_american.slots
    assert moroccan.unsupported_slots == {"food": "moroccan"}


def test_action_intent_wins_over_booking_info():
    reschedule = extract_slots("tell me about SIM-6ZP1SQ, are you able to reschduele to the next tuesday?")
    cancel = extract_slots("tell me about SIM-6ZP1SQ, are you able to cancel?")
    info = extract_slots("tell me about it")

    assert reschedule.intent == "reschedule"
    assert reschedule.slots["day"] == "tuesday"
    assert cancel.intent == "cancel"
    assert info.intent == "booking_info"


def test_booking_intent_wins_over_dish_words_in_restaurant_names():
    complete = extract_slots("book the golden curry for tomorrow, 6pm, 4 people")
    typo = extract_slots('book the resturant, "the goldern curry".')

    assert complete.intent == "book"
    assert complete.slots["relative_day"] == "tomorrow"
    assert complete.slots["time"] == "18:00"
    assert complete.slots["people"] == 4
    assert typo.intent == "book"


def test_restaurant_detail_intents_are_extracted():
    address = extract_slots("what is the address?")
    named_address = extract_slots("what is the address of meze bar?")
    booking_address = extract_slots("what is the address of my booking?")

    assert address.intent == "restaurant_info"
    assert named_address.intent == "restaurant_info"
    assert booking_address.intent == "restaurant_info"


def test_relative_day_slots_are_extracted():
    tomorrow = extract_slots("book it for tomorrow at 7pm for 2 people")
    day_after = extract_slots("move it to the day after")
    day_after_tomorrow = extract_slots("book it for day after tomorrow at 8pm for 3 people")
    next_week = extract_slots("next week")
    distance = extract_slots("how far is hakka from the south?")
    typo_weekday = extract_slots("sure nandos, next week thursdat, 4pm for 4 people")

    assert tomorrow.slots["relative_day"] == "tomorrow"
    assert day_after.intent == "reschedule"
    assert day_after.slots["relative_day"] == "day_after"
    assert day_after_tomorrow.slots["relative_day"] == "day_after_tomorrow"
    assert next_week.intent == "date_clarification"
    assert distance.intent == "distance_info"
    assert distance.slots["area"] == "south"
    assert typo_weekday.slots["day"] == "thursday"
    assert typo_weekday.slots["day_modifier"] == "next_week"
    assert typo_weekday.slots["time"] == "16:00"
    assert typo_weekday.slots["people"] == 4


def test_correction_area_price_and_walkable_distance_are_extracted():
    area = extract_slots("Actually west, not east")
    price = extract_slots("Oops make that expensive rather than cheap")
    not_pricey = extract_slots("I'm after British cooking in the centre, nothing pricey")
    walkable = extract_slots("Is this place walkable from the centre?")

    assert area.intent == "correct"
    assert area.slots == {"area": "west"}
    assert price.intent == "correct"
    assert price.slots == {"pricerange": "expensive"}
    assert not_pricey.slots["pricerange"] == "cheap"
    assert walkable.intent == "distance_info"
    assert walkable.slots["area"] == "centre"
