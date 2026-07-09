from argparse import Namespace

from scripts import run_evaluation_matrix as matrix


def _args(tmp_path, *, base_adapter_exists: bool) -> Namespace:
    small_adapter = tmp_path / "slot-extractor-qlora"
    small_adapter.mkdir()
    base_adapter = tmp_path / "slot-extractor-qlora-base"
    if base_adapter_exists:
        base_adapter.mkdir()
    return Namespace(
        sample_data=True,
        slot_fixture=tmp_path / "slot_cases.jsonl",
        model_name=None,
        adapter_path=small_adapter,
        base_adapter_path=base_adapter,
        output_dir=tmp_path / "outputs",
    )


def test_matrix_includes_separate_base_adapter_when_present(tmp_path, monkeypatch):
    calls = []

    def fake_run_evaluation(**kwargs):
        calls.append(kwargs)
        return {"slot_extraction": {}}

    monkeypatch.setattr(matrix, "run_evaluation", fake_run_evaluation)
    args = _args(tmp_path, base_adapter_exists=True)

    rows = matrix.run_matrix(args)

    assert [row["name"] for row in rows] == [
        "baseline_rule_based",
        "base_llm",
        "qlora_adapter",
        "qlora_adapter_base",
    ]
    assert all(row["status"] == "completed" for row in rows)
    assert calls[-1]["slot_model_name"] == str(args.base_adapter_path)
    assert f"--slot-model-name {args.base_adapter_path}" in rows[-1]["command"]


def test_matrix_skips_base_adapter_evaluation_when_folder_is_missing(tmp_path, monkeypatch):
    calls = []

    def fake_run_evaluation(**kwargs):
        calls.append(kwargs)
        return {"slot_extraction": {}}

    monkeypatch.setattr(matrix, "run_evaluation", fake_run_evaluation)

    rows = matrix.run_matrix(_args(tmp_path, base_adapter_exists=False))

    base_row = next(row for row in rows if row["name"] == "qlora_adapter_base")
    assert base_row["status"] == "skipped"
    assert "slot-extractor-qlora-base" in base_row["reason"]
    assert len(calls) == 3


def test_main_writes_separate_challenge_matrix_outputs(tmp_path, monkeypatch):
    main_fixture = tmp_path / "main.jsonl"
    challenge_fixture = tmp_path / "challenge.jsonl"
    output_dir = tmp_path / "reports"
    main_fixture.write_text("", encoding="utf-8")
    challenge_fixture.write_text("", encoding="utf-8")

    def fake_run_matrix(args):
        return [
            {
                "name": f"fixture:{args.slot_fixture.name}",
                "status": "completed",
                "command": "test",
                "metrics": {},
            }
        ]

    monkeypatch.setattr(matrix, "run_matrix", fake_run_matrix)

    matrix.main(
        [
            "--slot-fixture",
            str(main_fixture),
            "--challenge-fixture",
            str(challenge_fixture),
            "--output-dir",
            str(output_dir),
        ]
    )

    main_payload = (output_dir / "evaluation_matrix.json").read_text(encoding="utf-8")
    challenge_payload = (output_dir / "challenge_evaluation_matrix.json").read_text(encoding="utf-8")
    assert "fixture:main.jsonl" in main_payload
    assert "fixture:challenge.jsonl" in challenge_payload
    assert (output_dir / "evaluation_matrix.md").exists()
    assert (output_dir / "challenge_evaluation_matrix.md").exists()
