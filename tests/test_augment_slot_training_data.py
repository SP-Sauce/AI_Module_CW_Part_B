import json
from collections import Counter
from pathlib import Path

import pytest

from scripts.augment_slot_training_data import (
    ALLOWED_INTENTS,
    ALLOWED_SLOT_KEYS,
    DEFAULT_EVAL,
    DEFAULT_INPUT,
    REQUIRED_INTENTS,
    compact_output,
    main,
)
from scripts.check_data_leakage import load_records, normalize_text


@pytest.fixture(scope="module")
def generated_data(tmp_path_factory):
    output_file = tmp_path_factory.mktemp("augmented_slots") / "augmented.jsonl"
    result = main(
        [
            "--input-file",
            str(DEFAULT_INPUT),
            "--output-file",
            str(output_file),
            "--eval-file",
            str(DEFAULT_EVAL),
        ]
    )
    assert result == 0
    return output_file, load_records(output_file)


def test_augmentation_creates_larger_valid_jsonl(generated_data):
    output_file, records = generated_data
    original_records = load_records(DEFAULT_INPUT)

    assert output_file.exists()
    assert len(records) > len(original_records)
    assert 800 <= len(records) <= 1500
    assert output_file.read_text(encoding="utf-8").endswith("\n")


def test_augmented_text_does_not_overlap_holdout(generated_data):
    _, records = generated_data
    training_texts = {normalize_text(record["text"]) for record in records}
    evaluation_texts = {
        normalize_text(record["text"])
        for record in load_records(DEFAULT_EVAL)
    }

    assert training_texts.isdisjoint(evaluation_texts)


def test_augmented_outputs_are_compact_allowed_json(generated_data):
    _, records = generated_data

    for record in records:
        assert set(record) == {"text", "output"}
        output = record["output"]
        assert output.startswith("{") and output.endswith("}")
        target = json.loads(output)
        assert set(target) == {"intent", "slots"}
        assert target["intent"] in ALLOWED_INTENTS
        assert set(target["slots"]).issubset(ALLOWED_SLOT_KEYS)
        assert output == compact_output(target["intent"], target["slots"])


def test_augmented_distribution_contains_every_required_intent(generated_data):
    _, records = generated_data
    distribution = Counter(json.loads(record["output"])["intent"] for record in records)

    assert set(REQUIRED_INTENTS).issubset(distribution)
    assert all(distribution[intent] >= 55 for intent in REQUIRED_INTENTS)


def test_augmented_data_reinforces_all_slot_object_sizes(generated_data):
    _, records = generated_data
    slot_sizes = [len(json.loads(record["output"])["slots"]) for record in records]

    assert 0 in slot_sizes
    assert 1 in slot_sizes
    assert 2 in slot_sizes
    assert any(size >= 3 for size in slot_sizes)


def test_augmented_rows_are_deduplicated_by_normalized_text_and_output(generated_data):
    _, records = generated_data
    keys = [(normalize_text(record["text"]), record["output"]) for record in records]

    assert len(keys) == len(set(keys))
