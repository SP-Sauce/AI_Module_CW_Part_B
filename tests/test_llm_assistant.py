from restaurant_assistant import assistant as assistant_module
from restaurant_assistant.assistant import RestaurantAssistant
from restaurant_assistant.config import Settings
from restaurant_assistant.slot_extraction import SlotExtractionResult


class FakeLLMSlotExtractor:
    def __init__(self, model_name):
        self.model_name = model_name

    def extract(self, message):
        return SlotExtractionResult(
            intent="search",
            slots={"food": "italian", "area": "south", "pricerange": "cheap"},
            used_llm=True,
        )


def test_assistant_llm_mode_uses_persistent_slot_extractor(monkeypatch):
    monkeypatch.setattr(assistant_module, "OptionalLLMSlotExtractor", FakeLLMSlotExtractor)
    settings = Settings(enable_llm=True, slot_model_name="fake-slot-model")
    assistant = RestaurantAssistant(settings=settings, use_sample=True, enable_llm=True)
    assistant.generator.enable_llm = False

    result = assistant.process("Something vague that the fake model understands", debug=True)

    assert result.debug["slot_extraction_used_llm"] is True
    assert result.debug["slot_model_name"] == "fake-slot-model"
    assert "Pizza Hut Cherry Hinton" in result.response
