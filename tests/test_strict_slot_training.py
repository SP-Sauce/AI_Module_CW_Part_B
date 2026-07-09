import json

import pytest

from restaurant_assistant.slot_extraction import (
    OptionalLLMSlotExtractor,
    SlotExtractionResult,
    strict_parse_llm_json_output,
)
from scripts import build_slot_training_data as builder
from scripts import evaluate


def test_strict_parser_accepts_compact_json():
    parsed = strict_parse_llm_json_output(
        '{"intent":"search","slots":{"area":"centre","food":"italian","pricerange":"cheap"}}'
    )

    assert parsed == {
        "intent": "search",
        "slots": {"area": "centre", "food": "italian", "pricerange": "cheap"},
    }


@pytest.mark.parametrize(
    "raw_output",
    [
        '"intent":"search","slots":{"food":"italian"}',
        'JSON: {"intent":"search","slots":{"food":"italian"}}',
        '{"intent":"search","slots":{"food":"italian"}} trailing',
        'prefix {"intent":"search","slots":{"food":"italian"}}',
        '{"intent":"search"}',
    ],
)
def test_strict_parser_rejects_malformed_json(raw_output):
    with pytest.raises(ValueError):
        strict_parse_llm_json_output(raw_output)


def test_generated_training_labels_are_valid_json_and_do_not_copy_eval_text():
    rows = builder.build_rows(total_examples=1500, seed=builder.SEED, eval_files=builder.DEFAULT_EVAL_FILES)
    summary = builder.validate_rows(rows, eval_files=builder.DEFAULT_EVAL_FILES)
    eval_texts = builder.load_eval_texts(builder.DEFAULT_EVAL_FILES)

    assert summary["count"] == 1500
    assert {"search", "list", "book", "update_booking", "cancel_booking", "unsupported"} <= set(
        summary["intent_counts"]
    )
    for item in rows:
        target = json.loads(item["target"])
        assert item["target"] == builder.compact_target(target)
        assert builder.normalize_text(item["input"]) not in eval_texts


def test_optional_extractor_maps_strict_booking_intents_to_internal_labels():
    extractor = OptionalLLMSlotExtractor("fake-slot-model")

    class FakePipeline:
        def __call__(self, prompt, **generation_kwargs):
            return [
                {
                    "generated_text": json.dumps(
                        {
                            "intent": "update_booking",
                            "slots": {"booking_reference": "BK-X9Y8Z7", "day": "friday"},
                        }
                    )
                }
            ]

    extractor._pipeline = FakePipeline()

    result = extractor.extract("move booking BK-X9Y8Z7 to friday")

    assert result.intent == "reschedule"
    assert result.slots["booking_reference"] == "BK-X9Y8Z7"
    assert result.slots["day"] == "friday"


def test_raw_metrics_and_fallback_metrics_are_reported_separately(tmp_path, monkeypatch):
    fixture = tmp_path / "slot_cases.jsonl"
    fixture.write_text(
        json.dumps({"text": "find italian", "intent": "search", "slots": {"food": "italian"}}) + "\n",
        encoding="utf-8",
    )

    class FakeExtractor:
        def __init__(self, model_name, num_beams=1):
            self.model_name = model_name
            self.num_beams = num_beams

        def extract(self, message):
            return SlotExtractionResult(
                intent="search",
                slots={"food": "italian"},
                used_llm=False,
                llm_attempted=True,
                llm_parse_success=False,
                llm_repair_success=False,
                llm_intent_trusted=False,
                llm_meaningful_slot_contribution=False,
                llm_raw_output='"intent":"search","slots":',
                errors=["LLM output did not contain a valid intent/slots JSON object."],
            )

    monkeypatch.setattr(evaluate, "OptionalLLMSlotExtractor", FakeExtractor)

    metrics = evaluate.evaluate_slots(
        enable_llm=True,
        slot_model_name="models/slot-extractor-lora-strict",
        slot_fixture=fixture,
        slot_num_beams=1,
    )

    assert metrics["strict_json_parse_success_rate"] == 0.0
    assert metrics["raw_parse_error_count"] == 1
    assert metrics["strict_slot_f1"] == 0.0
    assert metrics["final_slot_f1"] == 1.0
    assert metrics["fallback_used_cases"] == 1
    assert metrics["raw_llm_metrics"]["raw_parse_error_count"] == 1
    assert metrics["after_repair_fallback_metrics"]["final_slot_f1"] == 1.0


def test_evaluate_accepts_strict_lora_model_path_and_report_path(tmp_path):
    report_path = tmp_path / "reports" / "strict.json"
    args = evaluate.build_parser().parse_args(
        [
            "--sample-data",
            "--enable-llm",
            "--slot-model-name",
            "models/slot-extractor-lora-strict",
            "--slot-num-beams",
            "4",
            "--report-path",
            str(report_path),
        ]
    )

    assert args.slot_model_name == "models/slot-extractor-lora-strict"
    assert args.slot_num_beams == 4
    assert args.report_path == report_path
