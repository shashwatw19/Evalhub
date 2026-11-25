"""
test_offline.py

Purpose:
    Verification test that strictly blocks network access at the socket level
    and verifies that the deterministic evaluation suite executes completely offline.
"""

import socket
from pathlib import Path
import pytest
from pydantic import SecretStr
from deepeval.config.settings import get_settings
from evalhub.runner import EvaluationRunner
from evalhub.storage import StorageEngine


def test_evaluation_runs_completely_offline(monkeypatch, tmp_path):
    """
    Ensure evaluation does not perform any network calls.
    Monkeypatches socket creation to raise an error if any network call is attempted.
    """
    def blocked_socket(*args, **kwargs):
        raise RuntimeError("Blocked network access: EvalHub deterministic metrics must be 100% offline!")

    # Block socket operations
    monkeypatch.setattr(socket, "socket", blocked_socket)
    monkeypatch.setattr(socket, "create_connection", blocked_socket)
    settings = get_settings()
    original_file_system = settings.DEEPEVAL_FILE_SYSTEM
    monkeypatch.setattr(
        settings, "CONFIDENT_API_KEY", SecretStr("offline-test-key")
    )

    db_path = str(tmp_path / "offline_test.db")
    storage = StorageEngine(db_path=db_path)
    runner = EvaluationRunner(storage=storage)

    project_root = Path(__file__).resolve().parent.parent
    dataset_path = str(project_root / "examples" / "generic" / "cases.jsonl")
    config_path = str(project_root / "examples" / "generic" / "config.json")

    # Must complete successfully without triggering blocked_socket
    summary, case_results = runner.run_evaluation(
        dataset_path=dataset_path,
        config_path=config_path,
        label="offline_verification_test"
    )

    assert summary.total_cases > 0
    assert summary.run_status == "completed"
    assert len(case_results) == summary.total_cases
    assert settings.CONFIDENT_API_KEY.get_secret_value() == "offline-test-key"
    assert settings.DEEPEVAL_FILE_SYSTEM == original_file_system
