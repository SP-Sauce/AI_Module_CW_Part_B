from dataclasses import replace

from restaurant_assistant.assistant import RestaurantAssistant
from restaurant_assistant.config import Settings
from restaurant_assistant.dialogue_state import DialogueState
from restaurant_assistant.llm_generator import GroundedResponseGenerator, validate_generated_response
from restaurant_assistant.ranking import RankedRestaurant
from scripts import run_web


def _ranked_restaurant():
    return RankedRestaurant(
        record={
            "name": "Test Restaurant",
            "food": "italian",
            "area": "east",
            "pricerange": "cheap",
            "address": "1 Test Street",
            "postcode": "CB1 1AA",
            "phone": "01223 111111",
        },
        score=1.0,
        matched_constraints=["food", "area", "pricerange"],
        missing_unmatched_constraints=[],
        explanation="test evidence",
        similarity=1.0,
    )


def test_run_web_enable_llm_alone_does_not_enable_response_llm():
    args = run_web.build_parser().parse_args(
        [
            "--enable-llm",
            "--slot-model-name",
            "models/slot-extractor-lora-strict",
        ]
    )
    settings = Settings()
    if args.enable_llm:
        settings = replace(settings, enable_llm=True)

    assert settings.enable_llm is True
    assert settings.enable_response_llm is False
    assert settings.response_model_name == "google/flan-t5-base"


def test_enable_response_llm_is_explicit():
    settings = Settings(enable_llm=True, enable_response_llm=True, response_model_name="google/flan-t5-base")
    assistant = RestaurantAssistant(settings=settings, use_sample=True, enable_llm=True)

    assert assistant.llm_enabled is True
    assert assistant.generator.enable_llm is True
    assert assistant.generator.model_name == "google/flan-t5-base"


def test_default_final_response_is_conversational_and_safe():
    assistant = RestaurantAssistant(restaurants=[_ranked_restaurant().record], enable_llm=False)

    result = assistant.process("Find cheap Italian in the east")

    assert "which matches your request" in result.response
    assert validate_generated_response(
        result.response,
        evidence_records=[_ranked_restaurant().record],
        known_restaurant_records=[_ranked_restaurant().record],
    ).ok


def test_generated_json_debug_leakage_is_rejected():
    generator = GroundedResponseGenerator(enable_llm=True, model_name="fake-generator")

    class LeakyPipeline:
        def __call__(self, prompt, max_new_tokens=120, do_sample=False):
            return [{"generated_text": '{"debug": true, "Evidence": []}'}]

    generator._pipeline = LeakyPipeline()
    state = DialogueState(food="italian", area="east", pricerange="cheap")
    result = generator.generate("Find cheap Italian in the east", state, [_ranked_restaurant()])

    assert result.used_llm is False
    assert result.rejected_reason == "json_or_debug_leakage"
    assert result.mode == "template"


def test_generated_invented_phone_address_and_postcode_are_rejected():
    evidence = [_ranked_restaurant().record]

    assert not validate_generated_response(
        "Test Restaurant is at 99 Fake Street. Phone: 01223 999999. Postcode: CB9 9ZZ.",
        evidence_records=evidence,
        known_restaurant_records=evidence,
    ).ok


def test_generated_unsupported_payment_dietary_and_availability_claims_are_rejected():
    evidence = [_ranked_restaurant().record]

    for text in [
        "Test Restaurant has live availability tonight.",
        "Test Restaurant accepts card payments online.",
        "Test Restaurant is halal and safe for allergies.",
    ]:
        result = validate_generated_response(text, evidence_records=evidence, known_restaurant_records=evidence)
        assert not result.ok
        assert result.reason == "unsupported_claim"


def test_trained_response_adapter_path_is_supported_if_present(tmp_path):
    adapter = tmp_path / "response-generator-lora"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    generator = GroundedResponseGenerator(enable_llm=True, model_name=str(adapter))

    class SafePipeline:
        def __call__(self, prompt, max_new_tokens=120, do_sample=False):
            return [{"generated_text": "I found Test Restaurant, which matches your request."}]

    generator._pipelines[f"trained_lora_response:{adapter}"] = SafePipeline()
    result = generator.generate(
        "Find cheap Italian in the east",
        DialogueState(food="italian", area="east", pricerange="cheap"),
        [_ranked_restaurant()],
    )

    assert result.used_llm is True
    assert result.final_response_mode == "trained_lora_response"


def test_response_llm_falls_back_to_template_when_generation_is_unsafe():
    generator = GroundedResponseGenerator(enable_llm=True, model_name="fake-generator")

    class UnsafePipeline:
        def __call__(self, prompt, max_new_tokens=120, do_sample=False):
            return [{"generated_text": "Test Restaurant has live availability tonight."}]

    generator._pipeline = UnsafePipeline()
    state = DialogueState(food="italian", area="east", pricerange="cheap")
    result = generator.generate("Find cheap Italian in the east", state, [_ranked_restaurant()])

    assert result.used_llm is False
    assert result.mode == "template"
    assert "which matches your request" in result.text
