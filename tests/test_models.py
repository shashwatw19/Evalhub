"""
test_models.py

Purpose:
    Unit tests for canonical domain models in evalhub.models.
"""

import pytest
from evalhub.models import (
    EvaluationCase,
    EvidenceRecord,
    ExecutionStatus,
    MetricConfig,
    MetricResult,
    MetricStatus,
    ToolCallRecord,
)


def test_tool_call_record():
    tc = ToolCallRecord(name="lookup_transaction", arguments={"id": "TXN-1"})
    assert tc.name == "lookup_transaction"
    assert tc.arguments == {"id": "TXN-1"}
    assert tc.status == "success"
    assert tc.result is None


def test_evidence_record():
    ev = EvidenceRecord(source="source-a", record_id="record-1", fields={"value": 100})
    assert ev.source == "source-a"
    assert ev.record_id == "record-1"
    assert ev.fields["value"] == 100


def test_evaluation_case_completed_requires_output():
    with pytest.raises(ValueError, match="actual_output is None"):
        EvaluationCase(
            case_id="C-1",
            input="test input",
            actual_output=None,
            execution_status=ExecutionStatus.COMPLETED
        )


def test_evaluation_case_failed_allows_none_output():
    case = EvaluationCase(
        case_id="C-2",
        input="test input",
        actual_output=None,
        execution_status=ExecutionStatus.FAILED,
        execution_error="Agent crashed"
    )
    assert case.case_id == "C-2"
    assert case.execution_status == ExecutionStatus.FAILED
    assert case.actual_output is None
    assert case.execution_error == "Agent crashed"


def test_tool_calls_none_vs_empty():
    case_none = EvaluationCase(
        case_id="C-3",
        input="test",
        actual_output="output",
        tool_calls=None
    )
    case_empty = EvaluationCase(
        case_id="C-4",
        input="test",
        actual_output="output",
        tool_calls=[]
    )
    assert case_none.tool_calls is None
    assert case_empty.tool_calls == []


def test_metric_result_safety_flag():
    res = MetricResult(
        name="custom_policy_check",
        status=MetricStatus.FAILED,
        score=0.0,
        explanation="Safety breached",
        is_safety_failure=True
    )
    assert res.is_safety_failure is True
    assert res.score == 0.0
