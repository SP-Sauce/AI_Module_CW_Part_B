import json

import pytest

from scripts.train_qlora_slot_extractor import load_examples


def test_training_targets_are_compact_json_objects(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            {
                "text": "Find Thai food",
                "intent": "search",
                "slots": {"food": "thai"},
            }
        ),
        encoding="utf-8",
    )

    examples = load_examples(path)

    assert examples[0]["target"] == '{"intent":"search","slots":{"food":"thai"}}'


def test_training_target_string_is_parsed_without_losing_braces(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            {
                "text": "Hello",
                "output": '{"intent": "greeting", "slots": {}}',
            }
        ),
        encoding="utf-8",
    )

    examples = load_examples(path)

    assert examples[0]["target"] == '{"intent":"greeting","slots":{}}'


def test_training_target_requires_intent_and_slots(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps({"text": "Hello", "output": {"intent": "greeting"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="intent.*slots"):
        load_examples(path)
