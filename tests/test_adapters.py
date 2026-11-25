"""Tests for canonical agent-record normalization and adapter registration."""

import pytest

from evalhub.adapters import (
    GenericAdapter,
    normalize_records,
    register_adapter,
)
from evalhub.models import EvaluationCase


def test_generic_adapter_parses_canonical_record_and_preserves_metadata():
    record = {
        "case_id": "GEN-1",
        "input": '{"prompt": "hello"}',
        "actual_output": '{"response": "world"}',
        "expected_output": '{"response": "world"}',
        "tool_calls": '[{"name": "search", "arguments": {"query": "hello"}}]',
        "agent_trace_id": "trace-1",
    }

    case = GenericAdapter.adapt(record)

    assert case.case_id == "GEN-1"
    assert case.input == {"prompt": "hello"}
    assert case.actual_output == {"response": "world"}
    assert case.expected_output == {"response": "world"}
    assert case.tool_calls is not None
    assert case.tool_calls[0].name == "search"
    assert case.metadata["agent_trace_id"] == "trace-1"


def test_generic_adapter_requires_canonical_case_id():
    with pytest.raises(ValueError):
        GenericAdapter.adapt({"input": "hello", "actual_output": "world"})


def test_normalize_records_rejects_duplicate_case_ids():
    records = [
        {"case_id": "DUP-1", "input": "a", "actual_output": "a"},
        {"case_id": "DUP-1", "input": "b", "actual_output": "b"},
    ]

    with pytest.raises(ValueError, match="Duplicate case_id 'DUP-1'"):
        normalize_records(records)


def test_custom_python_adapter_can_be_registered():
    def adapt(record):
        return EvaluationCase(
            case_id=record["id"],
            input=record["prompt"],
            actual_output=record["response"],
        )

    register_adapter("test-agent", adapt)
    cases = normalize_records(
        [{"id": "T-1", "prompt": "hello", "response": "world"}],
        adapter_name="test-agent",
    )

    assert cases[0].case_id == "T-1"
    assert cases[0].actual_output == "world"


def test_adapter_registration_rejects_duplicate_names():
    with pytest.raises(ValueError, match="already registered"):
        register_adapter("generic", GenericAdapter.adapt)
