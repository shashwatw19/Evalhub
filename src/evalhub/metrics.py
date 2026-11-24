"""
EvalHub's generic deterministic metrics executed through DeepEval.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from evalhub.models import (
    EvaluationCase,
    EvidenceRecord,
    MetricConfig,
    MetricResult,
    MetricStatus,
)


class EvalHubDeepEvalMetric(BaseMetric):
    """Run an EvalHub metric through DeepEval and retain its typed diagnostics."""

    metric_name: str

    def __init__(self, threshold: float = 1.0, required: bool = True):
        super().__init__()
        self.threshold = threshold
        self.required = required
        self._case_results: Dict[str, MetricResult] = {}

    @property
    def __name__(self) -> str:
        return self.metric_name

    def measure(
        self, test_case: LLMTestCase, *_args: Any, **_kwargs: Any
    ) -> float:
        if _args or _kwargs:
            raise TypeError(
                f"{self.metric_name} does not accept additional measurement arguments."
            )

        case_data = (test_case.metadata or {}).get("evalhub_case")
        if not isinstance(case_data, dict):
            raise ValueError(
                f"DeepEval test case '{test_case.name}' is missing EvalHub case metadata."
            )

        case = EvaluationCase.model_validate(case_data)
        result = self.evaluate_case(case)
        self._case_results[case.case_id] = result

        self.reason = result.explanation
        self.error = None
        self.success = result.status in (
            MetricStatus.PASSED,
            MetricStatus.NOT_APPLICABLE,
        )
        self.score = (
            result.score
            if result.score is not None
            else 1.0 if result.status == MetricStatus.NOT_APPLICABLE else 0.0
        )
        return self.score

    async def a_measure(
        self, test_case: LLMTestCase, *args: Any, **kwargs: Any
    ) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return bool(self.success)

    def result_for_case(self, case_id: str) -> MetricResult:
        try:
            return self._case_results[case_id]
        except KeyError as exc:
            raise RuntimeError(
                f"DeepEval did not produce '{self.metric_name}' for case '{case_id}'."
            ) from exc

    def evaluate_case(self, case: EvaluationCase) -> MetricResult:
        raise NotImplementedError(
            f"{self.metric_name} does not implement evaluation for case '{case.case_id}'."
        )


class ExactMatchMetric(EvalHubDeepEvalMetric):
    """Compare an agent's actual output with a supplied expected output."""

    metric_name = "exact_match"

    def evaluate_case(self, case: EvaluationCase) -> MetricResult:
        if case.actual_output is None:
            return MetricResult(
                name=self.metric_name,
                status=MetricStatus.ERROR,
                score=None,
                explanation=(
                    f"Execution did not produce output "
                    f"(status: {case.execution_status.value})."
                ),
                metadata={"case_id": case.case_id},
            )

        if case.expected_output is None:
            status = (
                MetricStatus.ERROR if self.required else MetricStatus.NOT_APPLICABLE
            )
            explanation = (
                "Prerequisite missing: expected_output is required."
                if self.required
                else "No expected_output provided; exact match is not applicable."
            )
            return MetricResult(
                name=self.metric_name,
                status=status,
                score=None,
                explanation=explanation,
                metadata={"case_id": case.case_id},
            )

        matched = case.actual_output == case.expected_output
        score = 1.0 if matched else 0.0
        status = MetricStatus.PASSED if score >= self.threshold else MetricStatus.FAILED
        return MetricResult(
            name=self.metric_name,
            status=status,
            score=score,
            explanation=(
                "Actual output matched expected output."
                if matched
                else "Actual output did not match expected output."
            ),
            metadata={"case_id": case.case_id},
        )


def _output_object(output: Any) -> Optional[Dict[str, Any]]:
    if isinstance(output, dict):
        return output
    if isinstance(output, str):
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


class EvidenceReferenceValidityMetric(EvalHubDeepEvalMetric):
    """Check that output evidence references exist in the case's supplied evidence."""

    metric_name = "evidence_reference_validity"

    def evaluate_case(self, case: EvaluationCase) -> MetricResult:
        if case.actual_output is None:
            return MetricResult(
                name=self.metric_name,
                status=MetricStatus.ERROR,
                score=None,
                explanation="No actual output to inspect for evidence references.",
                metadata={"case_id": case.case_id},
            )

        output = _output_object(case.actual_output)
        claimed = output.get("evidence") if output is not None else None
        if not isinstance(claimed, list):
            status = MetricStatus.FAILED if self.required else MetricStatus.NOT_APPLICABLE
            return MetricResult(
                name=self.metric_name,
                status=status,
                score=0.0 if self.required else None,
                explanation=(
                    "Output must contain an 'evidence' list."
                    if self.required
                    else "Output contains no evidence list; check is not applicable."
                ),
                metadata={"case_id": case.case_id},
            )

        if not claimed:
            status = MetricStatus.FAILED if self.required else MetricStatus.PASSED
            return MetricResult(
                name=self.metric_name,
                status=status,
                score=0.0 if self.required else 1.0,
                explanation=(
                    "The required evidence list is empty."
                    if self.required
                    else "No evidence references were supplied."
                ),
                metadata={"case_id": case.case_id, "citations": []},
            )

        authoritative: Dict[Tuple[str, str], EvidenceRecord] = {}
        records_by_id: Dict[str, List[EvidenceRecord]] = {}
        for record in case.evidence or []:
            source = record.source.strip().casefold()
            record_id = record.record_id.strip().casefold()
            authoritative[(source, record_id)] = record
            records_by_id.setdefault(record_id, []).append(record)

        diagnostics: List[Dict[str, Any]] = []
        valid_count = 0
        for citation in claimed:
            if not isinstance(citation, dict):
                diagnostics.append(
                    {"citation": citation, "valid": False, "reason": "Not an object."}
                )
                continue

            record_id = str(citation.get("record_id") or citation.get("id") or "")
            record_id = record_id.strip().casefold()
            source = str(citation.get("source") or "").strip().casefold()
            if not record_id:
                reason = "Missing record_id."
                diagnostics.append(
                    {"citation": citation, "valid": False, "reason": reason}
                )
                continue

            if source:
                matches = (
                    [authoritative[(source, record_id)]]
                    if (source, record_id) in authoritative
                    else []
                )
            else:
                matches = records_by_id.get(record_id, [])

            if len(matches) == 1:
                valid_count += 1
                diagnostics.append({"citation": citation, "valid": True})
                continue
            elif len(matches) > 1:
                reason = "Ambiguous record_id; include its source."
            else:
                reason = "Evidence reference not found in the supplied evidence."
            diagnostics.append(
                {"citation": citation, "valid": False, "reason": reason}
            )

        score = valid_count / len(claimed)
        status = MetricStatus.PASSED if score >= self.threshold else MetricStatus.FAILED
        return MetricResult(
            name=self.metric_name,
            status=status,
            score=score,
            explanation=(
                f"Verified {valid_count} of {len(claimed)} evidence references."
            ),
            metadata={
                "case_id": case.case_id,
                "citations": diagnostics,
                "valid_count": valid_count,
                "total_count": len(claimed),
            },
        )


class ToolCoverageMetric(EvalHubDeepEvalMetric):
    """Check that all configured tools were observed in the agent trace."""

    metric_name = "tool_coverage"

    def evaluate_case(self, case: EvaluationCase) -> MetricResult:
        required_tools = case.expected_required_tools
        if required_tools is None:
            status = MetricStatus.ERROR if self.required else MetricStatus.NOT_APPLICABLE
            return MetricResult(
                name=self.metric_name,
                status=status,
                score=None,
                explanation=(
                    "Prerequisite missing: expected_required_tools is required."
                    if self.required
                    else "No expected tool list provided; check is not applicable."
                ),
                metadata={"case_id": case.case_id},
            )

        if not required_tools:
            return MetricResult(
                name=self.metric_name,
                status=MetricStatus.NOT_APPLICABLE,
                score=None,
                explanation="No tools were required for this case.",
                metadata={"case_id": case.case_id},
            )

        if case.tool_calls is None:
            status = MetricStatus.ERROR if self.required else MetricStatus.NOT_APPLICABLE
            return MetricResult(
                name=self.metric_name,
                status=status,
                score=None,
                explanation=(
                    "Observed tool trace is missing."
                    if self.required
                    else "No observed tool trace; check is not applicable."
                ),
                metadata={"case_id": case.case_id},
            )

        required: Set[str] = set(required_tools)
        called = {tool_call.name for tool_call in case.tool_calls}
        matched = required.intersection(called)
        missing = sorted(required - called)
        score = len(matched) / len(required)
        status = MetricStatus.PASSED if score >= self.threshold else MetricStatus.FAILED
        return MetricResult(
            name=self.metric_name,
            status=status,
            score=score,
            explanation=(
                "All required tools were called."
                if not missing
                else f"Required tools not called: {missing}."
            ),
            metadata={
                "case_id": case.case_id,
                "required": sorted(required),
                "called": sorted(called),
                "missing": missing,
            },
        )


class ExplanationQualityMetric(EvalHubDeepEvalMetric):
    """Optional offline heuristic that scores free-text output length."""

    metric_name = "explanation_quality"

    def __init__(
        self,
        threshold: float = 0.7,
        required: bool = False,
        judge_enabled: bool = False,
    ):
        super().__init__(threshold=threshold, required=required)
        self.judge_enabled = judge_enabled

    def evaluate_case(self, case: EvaluationCase) -> MetricResult:
        if not self.judge_enabled:
            return MetricResult(
                name=self.metric_name,
                status=MetricStatus.NOT_APPLICABLE,
                score=None,
                explanation="Explanation quality check is disabled in configuration.",
                metadata={"case_id": case.case_id},
            )

        output = case.actual_output
        text = output if isinstance(output, str) else ""
        if not text and isinstance(output, dict):
            text = " ".join(
                str(value)
                for value in output.values()
                if isinstance(value, str)
            )
        if not text.strip():
            return MetricResult(
                name=self.metric_name,
                status=MetricStatus.FAILED,
                score=0.0,
                explanation="Output contains no text to evaluate.",
                metadata={"case_id": case.case_id},
            )

        word_count = len(text.split())
        score = min(1.0, word_count / 30.0)
        status = MetricStatus.PASSED if score >= self.threshold else MetricStatus.FAILED
        return MetricResult(
            name=self.metric_name,
            status=status,
            score=score,
            explanation=f"Offline length heuristic: {word_count} words, score {score:.2f}.",
            metadata={"case_id": case.case_id, "word_count": word_count},
        )


class GeminiJudgeMetric(EvalHubDeepEvalMetric):
    """Run an optional native DeepEval LLM metric using a Gemini judge."""

    def __init__(self, name: str, metric: BaseMetric, threshold: float, required: bool):
        super().__init__(threshold=threshold, required=required)
        self.metric_name = name
        self.metric = metric

    def evaluate_case(self, case: EvaluationCase) -> MetricResult:
        missing = []
        if case.actual_output is None:
            missing.append("actual_output")
        if self.metric_name == "answer_correctness" and case.expected_output is None:
            missing.append("expected_output")
        if self.metric_name == "faithfulness" and not case.retrieval_context:
            missing.append("retrieval_context")
        if missing:
            status = (
                MetricStatus.ERROR
                if self.required or case.actual_output is None
                else MetricStatus.NOT_APPLICABLE
            )
            return MetricResult(
                name=self.metric_name,
                status=status,
                explanation=f"Missing required fields: {', '.join(missing)}.",
            )

        def as_text(value: Any) -> str:
            return value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)

        test_case = LLMTestCase(
            input=as_text(case.input),
            actual_output=as_text(case.actual_output),
            expected_output=as_text(case.expected_output) if case.expected_output is not None else None,
            retrieval_context=case.retrieval_context,
        )
        try:
            score = self.metric.measure(test_case)
        except Exception as exc:
            return MetricResult(
                name=self.metric_name,
                status=MetricStatus.ERROR,
                explanation=f"Gemini judge failed ({type(exc).__name__}).",
                metadata={"case_id": case.case_id},
            )
        if score is None or not 0 <= score <= 1:
            raise ValueError(f"Judge metric '{self.metric_name}' returned an invalid score.")
        return MetricResult(
            name=self.metric_name,
            status=MetricStatus.PASSED if score >= self.threshold else MetricStatus.FAILED,
            score=score,
            explanation=self.metric.reason or "Gemini judge returned no explanation.",
            metadata={"case_id": case.case_id},
        )


def _make_gemini_metric(name: str, model_name: str, threshold: float) -> BaseMetric:
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
    from deepeval.models import GeminiModel
    from deepeval.test_case import SingleTurnParams

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is required for Gemini judge metrics.")
    model = GeminiModel(model=model_name, api_key=api_key, use_vertexai=False)
    if name == "answer_relevancy":
        return AnswerRelevancyMetric(model=model, threshold=threshold, async_mode=False)
    if name == "faithfulness":
        return FaithfulnessMetric(model=model, threshold=threshold, async_mode=False)
    return GEval(
        name="Answer correctness",
        criteria="Assess whether the actual output is factually consistent with the expected output.",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        model=model,
        threshold=threshold,
        async_mode=False,
    )


METRIC_REGISTRY: Dict[str, type[EvalHubDeepEvalMetric]] = {
    "exact_match": ExactMatchMetric,
    "evidence_reference_validity": EvidenceReferenceValidityMetric,
    "tool_coverage": ToolCoverageMetric,
    "explanation_quality": ExplanationQualityMetric,
}


def get_metric_instance(
    config: MetricConfig, judge_enabled: bool = False, judge_model: Optional[str] = None
) -> EvalHubDeepEvalMetric:
    """Instantiate a configured metric from the registry."""
    if config.name in {"answer_relevancy", "answer_correctness", "faithfulness"}:
        if not judge_enabled:
            raise ValueError(f"Metric '{config.name}' requires judge_enabled=true.")
        if not judge_model:
            raise ValueError("A Gemini judge_model is required for LLM metrics.")
        if config.options:
            raise ValueError(f"Metric '{config.name}' does not support options.")
        return GeminiJudgeMetric(
            name=config.name,
            metric=_make_gemini_metric(config.name, judge_model, config.threshold),
            threshold=config.threshold,
            required=config.required,
        )
    metric_cls = METRIC_REGISTRY.get(config.name)
    if metric_cls is None:
        raise ValueError(
            f"Unknown metric '{config.name}'. Available metrics: "
            f"{sorted(METRIC_REGISTRY)}"
        )
    if config.name == "explanation_quality":
        return metric_cls(
            threshold=config.threshold,
            required=config.required,
            judge_enabled=judge_enabled,
            **config.options,
        )
    return metric_cls(
        threshold=config.threshold,
        required=config.required,
        **config.options,
    )
