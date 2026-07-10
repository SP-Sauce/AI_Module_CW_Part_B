import json
import re

from restaurant_assistant.dialogue_state import DialogueState
from restaurant_assistant.llm_generator import validate_generated_response
from restaurant_assistant.llm_generator import GroundedResponseGenerator
from restaurant_assistant.ranking import RankedRestaurant
from restaurant_assistant.response_prompt import build_response_prompt, parse_response_input, prompt_from_row
from scripts import build_response_training_data as builder
from scripts import check_response_data_leakage as leakage
from scripts import evaluate_response_generation as response_eval
from scripts import train_lora_response_generator as train_response


def _restaurants(count=12):
    foods = ["italian", "indian", "chinese", "british", "thai", "turkish"]
    areas = ["centre", "north", "south", "east", "west"]
    prices = ["cheap", "moderate", "expensive"]
    return [
        {
            "source_id": f"r{index:03d}",
            "name": f"Restaurant {index}",
            "food": foods[index % len(foods)],
            "area": areas[index % len(areas)],
            "pricerange": prices[index % len(prices)],
            "address": f"{index} Test Street",
            "postcode": f"CB{index % 9 + 1} {index % 9}AA",
            "phone": f"01223 55{index:04d}",
            "type": "restaurant",
        }
        for index in range(1, count + 1)
    ]


def _users(rows):
    return {builder.extract_user_norm(row["input"]) for row in rows}


def _row_key(row):
    return json.dumps(row, sort_keys=True, separators=(",", ":"))


def test_response_dataset_is_deterministic_for_same_seed_and_changes_for_different_seed():
    restaurants = _restaurants(15)

    first = builder.build_dataset(restaurants, train_count=30, eval_count=12, challenge_count=10, seed=123)
    second = builder.build_dataset(restaurants, train_count=30, eval_count=12, challenge_count=10, seed=123)
    third = builder.build_dataset(restaurants, train_count=30, eval_count=12, challenge_count=10, seed=456)

    assert first.train_rows == second.train_rows
    assert first.eval_rows == second.eval_rows
    assert first.challenge_rows == second.challenge_rows
    assert first.train_rows != third.train_rows


def test_response_dataset_sizes_schema_uniqueness_and_split_separation():
    result = builder.build_dataset(_restaurants(18), train_count=66, eval_count=22, challenge_count=22, seed=6062026)

    assert len(result.train_rows) == 66
    assert len(result.eval_rows) == 22
    assert len(result.challenge_rows) == 22
    for rows in [result.train_rows, result.eval_rows, result.challenge_rows]:
        assert all(set(row) == {"instruction", "input", "output"} for row in rows)
        assert all(row["instruction"] and row["input"] and row["output"] for row in rows)
        assert len({_row_key(row) for row in rows}) == len(rows)

    assert not (_users(result.train_rows) & _users(result.eval_rows))
    assert not (_users(result.train_rows) & _users(result.challenge_rows))
    assert not (_users(result.eval_rows) & _users(result.challenge_rows))
    assert result.report["restaurant_level_disjoint_split_achieved"] is True
    assert result.report["overlap_counts"]["train_eval_restaurant_overlap"] == 0
    assert result.report["overlap_counts"]["train_challenge_restaurant_overlap"] == 0
    assert result.report["overlap_counts"]["eval_challenge_restaurant_overlap"] == 0


def test_response_targets_are_safe_and_booking_references_are_grounded():
    result = builder.build_dataset(_restaurants(14), train_count=80, eval_count=30, challenge_count=30, seed=6062026)
    rows = result.train_rows + result.eval_rows + result.challenge_rows

    for row in rows:
        evidence = builder.evidence_records_from_input(row["input"])
        validation = validate_generated_response(row["output"], evidence_records=evidence, known_restaurant_records=_restaurants(14))
        assert validation.ok, (validation.reason, row)
        assert "{" not in row["output"]
        assert "debug" not in row["output"].casefold()
        output_refs = set(re.findall(r"\bBK-[A-Z0-9]{6}\b", row["output"]))
        input_refs = set(re.findall(r"\bBK-[A-Z0-9]{6}\b", row["input"]))
        assert output_refs <= input_refs

    assert result.report["safety_validation_failure_count"] == 0
    assert result.report["booking_reference_grounding_failure_count"] == 0


def test_shared_response_prompt_is_aligned_and_does_not_include_target():
    result = builder.build_dataset(_restaurants(24), train_count=80, eval_count=24, challenge_count=24, seed=6062026)
    row = next(
        item
        for item in result.train_rows
        if (fields := parse_response_input(item["input"])).intent == "search"
        and {"food", "area", "pricerange"} <= set(fields.state)
        and fields.evidence_records
    )
    fields = parse_response_input(row["input"])
    shared_prompt = build_response_prompt(
        intent=fields.intent,
        user=fields.user,
        state=fields.state,
        evidence_records=fields.evidence_records,
        missing_slots=fields.missing_slots,
        instruction=row["instruction"],
    )
    runtime_state = DialogueState(
        food=fields.state.get("food"),
        area=fields.state.get("area"),
        pricerange=fields.state.get("pricerange"),
        day=fields.state.get("day"),
        time=fields.state.get("time"),
        people=int(fields.state["people"]) if fields.state.get("people", "").isdigit() else None,
        booking_status=fields.state.get("booking_status") or fields.state.get("status") or "none",
        booking_reference=fields.state.get("booking_reference"),
    )
    runtime_prompt = GroundedResponseGenerator(enable_llm=False)._build_prompt(
        fields.user,
        runtime_state,
        fields.evidence_records,
        intent=fields.intent,
        missing_slots=fields.missing_slots,
        baseline_text=row["output"],
    )

    assert prompt_from_row(row) == shared_prompt
    assert train_response.format_prompt(row) == shared_prompt
    assert runtime_prompt == shared_prompt
    assert "Baseline response:" not in runtime_prompt
    assert row["output"] not in runtime_prompt


def test_matching_labelled_restaurants_satisfy_prompt_constraints():
    result = builder.build_dataset(_restaurants(45), train_count=140, eval_count=50, challenge_count=50, seed=6062026)
    matching_rows = 0
    available_rows = 0
    for row in result.train_rows + result.eval_rows + result.challenge_rows:
        output = row["output"]
        if not output.startswith(("Matching restaurants:", "Other matching options:", "Other available records:")):
            continue
        fields = parse_response_input(row["input"])
        constraints = {
            key: value.casefold()
            for key, value in fields.state.items()
            if key in {"food", "area", "pricerange"} and value
        }
        if output.startswith("Other available records:"):
            available_rows += 1
            assert "may not match every preference" in output
            continue
        matching_rows += 1
        assert constraints, row
        for record in fields.evidence_records:
            for key, value in constraints.items():
                assert str(record.get(key, "")).casefold() == value, (key, value, record, row)

    assert matching_rows > 0
    assert available_rows >= 0


def test_response_evaluator_separates_baseline_raw_and_guarded_rows(tmp_path):
    result = builder.build_dataset(_restaurants(18), train_count=30, eval_count=10, challenge_count=10, seed=6062026)
    eval_file = tmp_path / "eval.jsonl"
    builder.write_jsonl(eval_file, result.eval_rows[:2])
    args = response_eval.build_parser().parse_args(
        [
            "--sample-data",
            "--eval-file",
            str(eval_file),
            "--json-report",
            str(tmp_path / "comparison.json"),
            "--markdown-report",
            str(tmp_path / "comparison.md"),
        ]
    )

    payload = response_eval.evaluate(args)
    case = payload["cases"][0]

    assert set(payload["metrics"]) == {
        "deterministic_baseline_response",
        "raw_pretrained_model_response",
        "raw_trained_lora_response",
        "final_guarded_response",
    }
    assert case["deterministic_baseline_response"]["text"] == case["reference_output"]
    assert case["raw_pretrained_model_response"]["skipped"] is True
    assert case["raw_trained_lora_response"]["skipped"] is True
    assert case["final_guarded_response"]["text"] == case["reference_output"]
    assert "model_attempt_count" in payload["metrics"]["final_guarded_response"]
    assert "rejection_reason_counts" in payload["metrics"]["final_guarded_response"]


def test_synthetic_language_uses_natural_articles_and_people_wording():
    result = builder.build_dataset(_restaurants(45), train_count=140, eval_count=50, challenge_count=50, seed=6062026)
    text = "\n".join(row["input"] + "\n" + row["output"] for row in result.train_rows + result.eval_rows + result.challenge_rows).casefold()

    assert "1 people" not in text
    assert "a expensive" not in text
    assert "a inexpensive" not in text
    assert "a upmarket" not in text
    assert "for expensive" not in text
    assert "1 person" in text
    assert "2 people" in text
    assert "an expensive" in text
    assert "a cheap" in text


def test_sample_sized_dataset_is_handled_gracefully_and_challenge_wording_differs():
    result = builder.build_dataset(_restaurants(4), train_count=20, eval_count=8, challenge_count=8, seed=6062026, sample_data=True)

    assert len(result.train_rows) == 20
    assert len(result.eval_rows) == 8
    assert len(result.challenge_rows) == 8
    assert result.report["sample_data_used"] is True
    assert not (_users(result.eval_rows) & _users(result.challenge_rows))
    challenge_users = " ".join(builder.extract_user(row["input"]).casefold() for row in result.challenge_rows)
    assert any(term in challenge_users for term in ["ta", "mate", "pls", "quick one", "resturant"])


def test_report_files_and_leakage_checker_success(tmp_path):
    result = builder.build_dataset(_restaurants(12), train_count=30, eval_count=10, challenge_count=10, seed=6062026)
    train = tmp_path / "train.jsonl"
    eval_file = tmp_path / "eval.jsonl"
    challenge = tmp_path / "challenge.jsonl"
    report_json = tmp_path / "report.json"
    report_md = tmp_path / "report.md"

    builder.write_jsonl(train, result.train_rows)
    builder.write_jsonl(eval_file, result.eval_rows)
    builder.write_jsonl(challenge, result.challenge_rows)
    builder.write_report_json(report_json, result.report)
    builder.write_report_md(report_md, result.report)

    assert report_json.exists()
    assert report_md.exists()
    metrics = leakage.check_splits(
        train_file=train,
        eval_file=eval_file,
        challenge_file=challenge,
        dataset_report=report_json,
    )
    assert metrics["error_count"] == 0


def test_leakage_checker_detects_deliberately_duplicated_data(tmp_path):
    result = builder.build_dataset(_restaurants(9), train_count=6, eval_count=3, challenge_count=3, seed=6062026)
    duplicated = result.train_rows[:1] + result.train_rows[:1]
    train = tmp_path / "train.jsonl"
    eval_file = tmp_path / "eval.jsonl"
    challenge = tmp_path / "challenge.jsonl"
    report_json = tmp_path / "report.json"

    builder.write_jsonl(train, duplicated)
    builder.write_jsonl(eval_file, duplicated)
    builder.write_jsonl(challenge, result.challenge_rows)
    builder.write_report_json(report_json, {**result.report, "restaurant_level_disjoint_split_achieved": True})

    metrics = leakage.check_splits(
        train_file=train,
        eval_file=eval_file,
        challenge_file=challenge,
        dataset_report=report_json,
    )

    assert metrics["error_count"] > 0
    assert any("duplicate rows" in error or "input_overlap" in error for error in metrics["errors"])


def test_cautious_unsupported_disclaimers_are_allowed_but_positive_claims_are_rejected():
    safe_texts = [
        "I cannot verify halal status from the loaded restaurant records.",
        "I cannot confirm allergy safety. Please contact the restaurant directly.",
        "I cannot check live table availability, but I can create a local booking record for the demonstration.",
    ]
    for text in safe_texts:
        assert validate_generated_response(text).ok

    unsafe_texts = [
        "This restaurant is halal.",
        "This restaurant is safe for allergies.",
        "The table is available tonight.",
        "You can pay online by card.",
    ]
    for text in unsafe_texts:
        result = validate_generated_response(text)
        assert not result.ok
        assert result.reason == "unsupported_claim"
