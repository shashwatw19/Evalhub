"""
test_storage.py

Purpose:
    Unit tests for SQLite StorageEngine persistence, schema versioning, and comparison logic.
"""

import pytest
from evalhub.models import (
    CaseEvaluationResult,
    EvaluationCase,
    MetricResult,
    MetricStatus,
    RunSummary,
)
from evalhub.storage import StorageEngine


@pytest.fixture
def temp_storage(tmp_path):
    db_file = tmp_path / "test_evalhub.db"
    return StorageEngine(db_path=str(db_file))


def test_schema_version_and_tables(temp_storage):
    assert temp_storage.get_schema_version() == 1


def test_save_and_retrieve_run(temp_storage):
    summary = RunSummary(
        run_id="run_test_1",
        label="baseline_test",
        created_at="2026-11-23T10:00:00Z",
        run_status="completed",
        dataset_path="test.jsonl",
        dataset_hash="hash123",
        comparison_fingerprint="fp123",
        config_hash="conf123",
        config_json="{}",
        deepeval_version="4.3.4",
        metric_versions_json="{}",
        total_cases=1,
        passed_cases=1,
        failed_cases=0,
        error_cases=0,
        not_applicable_cases=0,
        safety_failures=0
    )

    case = EvaluationCase(case_id="C-1", input="in", actual_output="out")
    case_res = CaseEvaluationResult(
        case=case,
        metric_results=[
            MetricResult(name="exact_match", status=MetricStatus.PASSED, score=1.0, explanation="ok")
        ],
        overall_status=MetricStatus.PASSED
    )

    temp_storage.save_run(summary, [case_res])

    fetched_summary = temp_storage.get_run("run_test_1")
    assert fetched_summary is not None
    assert fetched_summary.run_id == "run_test_1"
    assert fetched_summary.label == "baseline_test"

    results = temp_storage.get_case_results("run_test_1")
    assert len(results) == 1
    assert results[0]["case_id"] == "C-1"
    assert results[0]["overall_status"] == "passed"


def test_compare_runs_incompatible_fingerprint(temp_storage):
    summary_b = RunSummary(
        run_id="run_b",
        created_at="2026-11-23T10:00:00Z",
        dataset_path="b.jsonl",
        dataset_hash="h1",
        comparison_fingerprint="fp_A",
        config_hash="c1",
        config_json="{}",
        deepeval_version="4.3.4",
        metric_versions_json="{}",
        total_cases=1,
        passed_cases=1,
        failed_cases=0,
        error_cases=0,
        not_applicable_cases=0
    )
    summary_c = RunSummary(
        run_id="run_c",
        created_at="2026-11-23T11:00:00Z",
        dataset_path="c.jsonl",
        dataset_hash="h2",
        comparison_fingerprint="fp_B",  # Mismatch
        config_hash="c2",
        config_json="{}",
        deepeval_version="4.3.4",
        metric_versions_json="{}",
        total_cases=1,
        passed_cases=1,
        failed_cases=0,
        error_cases=0,
        not_applicable_cases=0
    )
    temp_storage.save_run(summary_b, [])
    temp_storage.save_run(summary_c, [])

    diff = temp_storage.compare_runs("run_b", "run_c")
    assert diff.fingerprint_match is False
    assert diff.status == "incompatible"
