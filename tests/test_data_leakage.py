import json

from scripts.check_data_leakage import find_leaked_texts, normalize_text


def test_normalize_text_collapses_case_punctuation_and_spacing():
    assert normalize_text("  Book,  It for FRIDAY!! ") == "book it for friday"


def test_find_leaked_texts_detects_normalized_duplicates(tmp_path):
    train_file = tmp_path / "train.jsonl"
    eval_file = tmp_path / "eval.json"
    train_file.write_text('{"text":"Book it for Friday!"}\n', encoding="utf-8")
    eval_file.write_text(json.dumps([{"text": "book it for friday"}]), encoding="utf-8")

    leaked = find_leaked_texts(train_file, eval_file)

    assert leaked == [
        {
            "normalized_text": "book it for friday",
            "train_text": "Book it for Friday!",
            "eval_text": "book it for friday",
        }
    ]
