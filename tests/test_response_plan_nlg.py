from restaurant_assistant.assistant import RestaurantAssistant
from restaurant_assistant.nlg import contains_json_or_debug_leakage
from restaurant_assistant.slot_extraction import SlotExtractionResult


FORBIDDEN_TERMS = [
    "{",
    "}",
    "source_id",
    "name_norm",
    "food_norm",
    "area_norm",
    "pricerange_norm",
    "similarity",
    "score",
    "debug",
    "slot_model_name",
    "generation_mode",
]


def _restaurants():
    return [
        {
            "source_id": "r1",
            "name": "efes restaurant",
            "food": "turkish",
            "area": "centre",
            "pricerange": "moderate",
            "address": "King Street City Centre",
            "postcode": "cb11ln",
            "phone": "01223500005",
            "name_norm": "efes restaurant",
        },
        {
            "source_id": "r2",
            "name": "anatolia",
            "food": "turkish",
            "area": "centre",
            "pricerange": "expensive",
            "address": "30 Bridge Street City Centre",
            "postcode": "cb21uj",
            "phone": "01223362372",
            "name_norm": "anatolia",
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
        {
            "source_id": "r4",
            "name": "pizza hut city centre",
            "food": "italian",
            "area": "centre",
            "pricerange": "cheap",
            "address": "Regent Street City Centre",
            "postcode": "cb21ab",
            "phone": "01223323737",
        },
        {
            "source_id": "r5",
            "name": "the gardenia",
            "food": "mediterranean",
            "area": "east",
            "pricerange": "cheap",
            "address": "2 Rose Street",
            "postcode": "cb12aa",
            "phone": "01223222222",
        },
    ]


def _assert_no_leakage(text):
    lowered = text.casefold()
    assert not contains_json_or_debug_leakage(text)
    for term in FORBIDDEN_TERMS:
        assert term not in lowered


def test_no_json_or_debug_leakage_in_assistant_message_when_llm_leaks():
    assistant = RestaurantAssistant(restaurants=_restaurants(), enable_llm=False)
    assistant.generator.enable_llm = True

    class LeakyPipeline:
        def __call__(self, prompt, max_new_tokens=120, do_sample=False):
            return [
                {
                    "generated_text": (
                        '"name": "efes restaurant", "source_id": "r1", '
                        '"similarity": 0.99, "generation_mode": "debug"'
                    )
                }
            ]

    assistant.generator._pipeline = LeakyPipeline()

    result = assistant.process("I like spaghetti, what's a good restaurant for that?", debug=True)

    _assert_no_leakage(result.response)
    assert "pizza hut city centre" in result.response.lower()
    assert result.debug["generation_mode"] == "template"


def test_no_results_response_is_human_readable():
    assistant = RestaurantAssistant(restaurants=_restaurants())

    result = assistant.process("list me all Turkish restaurants in the east")

    _assert_no_leakage(result.response)
    assert "could not find restaurants" in result.response.lower()
    assert '"intent"' not in result.response


def test_list_response_is_human_readable():
    assistant = RestaurantAssistant(restaurants=_restaurants())

    result = assistant.process("can you list all the cheap restaurants in the centre?")

    _assert_no_leakage(result.response)
    assert result.response.startswith("Matching restaurants:")
    assert "kohinoor (cheap indian, centre)" in result.response
    assert "pizza hut city centre (cheap italian, centre)" in result.response


def test_standalone_cheap_restaurant_query_does_not_keep_moderate_or_expensive_context():
    assistant = RestaurantAssistant(restaurants=_restaurants())
    assistant.process("list all Turkish restaurants")

    result = assistant.process("what is a cheap restaurant?")

    _assert_no_leakage(result.response)
    assert "cheap" in result.response.lower()
    assert "efes restaurant" not in result.response.lower()
    assert "anatolia" not in result.response.lower()
    assert "moderate turkish" not in result.response.lower()
    assert "expensive turkish" not in result.response.lower()
    assert assistant.state.food is None
    assert assistant.state.pricerange == "cheap"


def test_spaghetti_maps_to_italian_recommendation():
    assistant = RestaurantAssistant(restaurants=_restaurants())

    result = assistant.process("I like spaghetti, whats a good resutrant for that?")

    _assert_no_leakage(result.response)
    assert "italian" in result.response.lower()
    assert "pizza hut city centre" in result.response.lower()


def test_conflicting_constraints_are_handled_safely_with_new_area_and_price():
    assistant = RestaurantAssistant(restaurants=_restaurants())
    assistant.process("show me Turkish restaurants in the centre")

    result = assistant.process("what is a cheap restaurant in the east?")

    _assert_no_leakage(result.response)
    assert "the gardenia" in result.response.lower()
    assert "efes restaurant" not in result.response.lower()
    assert assistant.state.food is None
    assert assistant.state.area == "east"
    assert assistant.state.pricerange == "cheap"


def test_table_view_formats_restaurant_results_cleanly():
    assistant = RestaurantAssistant(restaurants=_restaurants())
    assistant.process("can you list all the cheap restaurants in the centre?")

    result = assistant.process("as a table")

    _assert_no_leakage(result.response)
    assert result.response.startswith("Matching restaurants:")
    assert "| Name | Price | Cuisine | Area |" in result.response
    assert "| kohinoor | cheap | indian | centre |" in result.response


def test_malformed_llm_slot_output_falls_back_to_safe_extraction_and_nlg():
    assistant = RestaurantAssistant(restaurants=_restaurants(), enable_llm=False)

    class MalformedSlotExtractor:
        def extract(self, message):
            return SlotExtractionResult(
                intent="search",
                slots={"pricerange": "cheap"},
                used_llm=True,
                llm_attempted=True,
                llm_parse_success=False,
                errors=[
                    'LLM output did not contain a valid intent/slots JSON object. Raw output: '
                    '\'"intent":"restaurant_info","slots":\''
                ],
            )

    assistant.slot_extractor = MalformedSlotExtractor()

    result = assistant.process("what is a cheap restaurant?", debug=True)

    _assert_no_leakage(result.response)
    assert "cheap" in result.response.lower()
    assert result.debug["slot_extraction_errors"]
    assert result.debug["response_plan"]["dialogue_act"] in {"single_recommendation", "direct_message"}


def test_unsupported_taxi_hotel_train_requests_are_polite():
    assistant = RestaurantAssistant(restaurants=_restaurants())

    for message in [
        "please book me a taxi",
        "can you find a hotel?",
        "I need train times",
    ]:
        result = assistant.process(message)
        _assert_no_leakage(result.response)
        assert "only help with MultiWOZ restaurant search" in result.response
