"""
test_cli.py

Purpose:
    Unit tests for EvalHub CLI commands and exit code contracts.
"""

from pathlib import Path
import pytest
from evalhub.cli import main


def test_cli_validate_success(tmp_path):
    dataset_file = tmp_path / "valid.jsonl"
    dataset_file.write_text(
        '{"case_id":"ALT-1","input":"prompt","actual_output":"response"}\n',
        encoding="utf-8"
    )
    exit_code = main(["validate", "--dataset", str(dataset_file)])
    assert exit_code == 0


def test_cli_validate_invalid_syntax(tmp_path):
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text('{not valid json}\n', encoding="utf-8")
    exit_code = main(["validate", "--dataset", str(bad_file)])
    assert exit_code == 2


def test_cli_run_and_compare(tmp_path):
    db_file = str(tmp_path / "cli_test.db")
    dataset_file = tmp_path / "cases.jsonl"
    dataset_file.write_text(
        '{"case_id":"ALT-1","input":"prompt","actual_output":"response",'
        '"expected_output":"response"}\n',
        encoding="utf-8"
    )
    config_file = tmp_path / "config.json"
    config_file.write_text(
        '{"adapter":"generic","metrics":['
        '{"name":"exact_match","threshold":1.0,"required":true}]}\n',
        encoding="utf-8"
    )

    # Run baseline
    exit_code = main([
        "run",
        "--dataset", str(dataset_file),
        "--config", str(config_file),
        "--label", "baseline",
        "--db", db_file
    ])
    assert exit_code == 0

    # List runs
    assert main(["list-runs", "--db", db_file]) == 0


def test_cli_validates_input_only_dataset(tmp_path):
    dataset_file = tmp_path / "inputs.jsonl"
    dataset_file.write_text('{"case_id":"C-1","input":"question"}\n', encoding="utf-8")
    assert main(["validate", "--dataset", str(dataset_file), "--input-only"]) == 0


def test_cli_runs_bundled_python_agent(tmp_path):
    root = Path(__file__).resolve().parents[1]
    assert main([
        "run",
        "--dataset", str(root / "examples" / "generic" / "agent_cases.jsonl"),
        "--config", str(root / "examples" / "generic" / "python_agent_config.json"),
        "--db", str(tmp_path / "example.db"),
    ]) == 0
