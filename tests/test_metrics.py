"""Unit tests for EvalHub's generic deterministic metrics."""

from evalhub.metrics import (
    EvidenceReferenceValidityMetric,
    ExactMatchMetric,
    ToolCoverageMetric,
)
from evalhub.models import (
    EvaluationCase,
    EvidenceRecord,
    MetricStatus,
    ToolCallRecord,
)
from evalhub.models import MetricConfig
from evalhub.metrics import _make_gemini_metric, get_metric_instance
import pytest


def test_exact_match_pass_and_fail():
    matching_case = EvaluationCase(
        case_id="T-1",
        input="question",
        actual_output={"answer": 42},
        expected_output={"answer": 42},
    )
    mismatching_case = EvaluationCase(
        case_id="T-2",
        input="question",
        actual_output={"answer": 41},
        expected_output={"answer": 42},
    )
    metric = ExactMatchMetric()

    assert metric.evaluate_case(matching_case).status == MetricStatus.PASSED
    result = metric.evaluate_case(mismatching_case)
    assert result.status == MetricStatus.FAILED
    assert result.score == 0.0


def test_exact_match_missing_expected_output():
    case = EvaluationCase(
        case_id="T-3",
        input="question",
        actual_output="answer",
    )

    assert ExactMatchMetric(required=True).evaluate_case(case).status == MetricStatus.ERROR
    assert (
        ExactMatchMetric(required=False).evaluate_case(case).status
        == MetricStatus.NOT_APPLICABLE
    )


def test_evidence_reference_validity_with_source_and_record_id():
    case = EvaluationCase(
        case_id="T-4",
        input="question",
        actual_output={
            "evidence": [
                {"source": "documents", "record_id": "DOC-1"},
                {"record_id": "DOC-2"},
            ]
        },
        evidence=[
            EvidenceRecord(source="documents", record_id="DOC-1"),
            EvidenceRecord(source="archive", record_id="DOC-2"),
        ],
    )

    result = EvidenceReferenceValidityMetric().evaluate_case(case)

    assert result.status == MetricStatus.PASSED
    assert result.score == 1.0


def test_evidence_reference_validity_rejects_missing_and_ambiguous_references():
    case = EvaluationCase(
        case_id="T-5",
        input="question",
        actual_output={
            "evidence": [
                {"record_id": "MISSING"},
                {"record_id": "DUPLICATE"},
            ]
        },
        evidence=[
            EvidenceRecord(source="one", record_id="DUPLICATE"),
            EvidenceRecord(source="two", record_id="DUPLICATE"),
        ],
    )

    result = EvidenceReferenceValidityMetric().evaluate_case(case)

    assert result.status == MetricStatus.FAILED
    assert result.score == 0.0
    assert result.metadata["citations"][1]["reason"] == (
        "Ambiguous record_id; include its source."
    )


def test_evidence_reference_check_can_be_not_applicable():
    case = EvaluationCase(
        case_id="T-6",
        input="question",
        actual_output="plain-text response",
    )

    result = EvidenceReferenceValidityMetric(required=False).evaluate_case(case)

    assert result.status == MetricStatus.NOT_APPLICABLE


def test_tool_coverage_pass_and_fail():
    case = EvaluationCase(
        case_id="T-7",
        input="question",
        actual_output="answer",
        expected_required_tools=["search", "fetch"],
        tool_calls=[ToolCallRecord(name="search"), ToolCallRecord(name="fetch")],
    )
    metric = ToolCoverageMetric()
    assert metric.evaluate_case(case).status == MetricStatus.PASSED

    case.tool_calls = [ToolCallRecord(name="search")]
    result = metric.evaluate_case(case)
    assert result.status == MetricStatus.FAILED
    assert result.score == 0.5


def test_tool_coverage_missing_trace_is_error_or_not_applicable():
    case = EvaluationCase(
        case_id="T-8",
        input="question",
        actual_output="answer",
        expected_required_tools=["search"],
        tool_calls=None,
    )

    assert ToolCoverageMetric(required=True).evaluate_case(case).status == MetricStatus.ERROR
    assert (
        ToolCoverageMetric(required=False).evaluate_case(case).status
        == MetricStatus.NOT_APPLICABLE
    )


def test_gemini_metric_requires_opt_in_and_maps_judge_result(monkeypatch):
    class FakeJudge:
        reason = "The answer is relevant."

        def measure(self, test_case):
            assert test_case.input == "question"
            assert test_case.actual_output == "answer"
            return 0.8

    monkeypatch.setattr("evalhub.metrics._make_gemini_metric", lambda *args: FakeJudge())
    config = MetricConfig(name="answer_relevancy", threshold=0.7)
    with pytest.raises(ValueError, match="judge_enabled"):
        get_metric_instance(config)
    metric = get_metric_instance(config, judge_enabled=True, judge_model="gemini-test")
    case = EvaluationCase(case_id="judge", input="question", actual_output="answer")
    result = metric.evaluate_case(case)
    assert result.status == MetricStatus.PASSED
    assert result.score == 0.8
    assert result.explanation == "The answer is relevant."


@pytest.mark.parametrize("required", [True, False])
def test_judge_does_not_use_lookup_as_original_input(monkeypatch, required):
    class UnexpectedJudge:
        def measure(self, _case):
            pytest.fail("A judge must not be called without the original task.")

    monkeypatch.setattr("evalhub.metrics._make_gemini_metric", lambda *args: UnexpectedJudge())
    metric = get_metric_instance(
        MetricConfig(name="answer_relevancy", required=required),
        judge_enabled=True, judge_model="gemini-test",
    )
    case = EvaluationCase(
        case_id="lookup-only", lookup={"run_id": "run-1"}, actual_output="answer",
    )
    result = metric.evaluate_case(case)
    assert result.status == (MetricStatus.ERROR if required else MetricStatus.NOT_APPLICABLE)
    assert "original agent task" in result.explanation


@pytest.mark.parametrize("vertex", [False, True])
def test_gemini_judge_selects_configured_backend(monkeypatch, vertex):
    captured = {}

    def fake_model(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("deepeval.models.GeminiModel", fake_model)
    monkeypatch.setattr("deepeval.metrics.GEval", lambda **kwargs: kwargs)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", str(vertex).lower())
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")

    _make_gemini_metric("answer_correctness", "gemini-test", 0.7)

    if vertex:
        assert captured == {
            "model": "gemini-test",
            "project": "test-project",
            "location": "global",
            "use_vertexai": True,
        }
    else:
        assert captured == {
            "model": "gemini-test",
            "api_key": "test-key",
            "use_vertexai": False,
        }


@pytest.mark.parametrize("missing", ["GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"])
def test_vertex_judge_requires_project_and_location(monkeypatch, missing):
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")
    monkeypatch.delenv(missing)
    with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION"):
        _make_gemini_metric("answer_correctness", "gemini-test", 0.7)


def test_gemini_judge_rejects_invalid_vertex_setting(monkeypatch):
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "yes")
    with pytest.raises(ValueError, match="GOOGLE_GENAI_USE_VERTEXAI"):
        _make_gemini_metric("answer_correctness", "gemini-test", 0.7)


def test_gemini_faithfulness_requires_retrieval_context(monkeypatch):
    monkeypatch.setattr("evalhub.metrics._make_gemini_metric", lambda *args: object())
    metric = get_metric_instance(
        MetricConfig(name="faithfulness", required=False),
        judge_enabled=True,
        judge_model="gemini-test",
    )
    result = metric.evaluate_case(EvaluationCase(case_id="c", input="q", actual_output="a"))
    assert result.status == MetricStatus.NOT_APPLICABLE
    assert "retrieval_context" in result.explanation


def test_gemini_faithfulness_uses_successful_tool_results_as_context(monkeypatch):
    class FakeJudge:
        reason = "The suggestion is grounded in tool results."

        def measure(self, test_case):
            assert test_case.retrieval_context == [
                "[Tool: alert_lookup]\nSeverity: high",
                '[Tool: policy_lookup]\n{"action": "escalate"}',
            ]
            return 0.9

    monkeypatch.setattr("evalhub.metrics._make_gemini_metric", lambda *args: FakeJudge())
    metric = get_metric_instance(
        MetricConfig(name="faithfulness", threshold=0.7),
        judge_enabled=True,
        judge_model="gemini-test",
    )
    case = EvaluationCase(
        case_id="c",
        input="Investigate alert A-1",
        actual_output="Escalate this high-severity alert.",
        tool_calls=[
            ToolCallRecord(name="alert_lookup", result="Severity: high"),
            ToolCallRecord(
                name="policy_lookup",
                result={"action": "escalate"},
                status="success",
            ),
            ToolCallRecord(
                name="failed_lookup",
                result="Ignore this result",
                status="failed",
            ),
            ToolCallRecord(name="empty_lookup", result=None),
        ],
    )

    result = metric.evaluate_case(case)

    assert result.status == MetricStatus.PASSED
    assert result.score == 0.9


def test_gemini_faithfulness_prefers_explicit_retrieval_context(monkeypatch):
    class FakeJudge:
        reason = "Used the supplied reference context."

        def measure(self, test_case):
            assert test_case.retrieval_context == ["Reviewed reference"]
            return 0.9

    monkeypatch.setattr("evalhub.metrics._make_gemini_metric", lambda *args: FakeJudge())
    metric = get_metric_instance(
        MetricConfig(name="faithfulness", threshold=0.7),
        judge_enabled=True,
        judge_model="gemini-test",
    )
    case = EvaluationCase(
        case_id="c",
        input="question",
        actual_output="answer",
        retrieval_context=["Reviewed reference"],
        tool_calls=[ToolCallRecord(name="lookup", result="Unreviewed tool result")],
    )

    result = metric.evaluate_case(case)

    assert result.status == MetricStatus.PASSED
