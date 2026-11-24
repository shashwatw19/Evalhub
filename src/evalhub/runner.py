"""
runner.py

Purpose:
    Core evaluation orchestrator for EvalHub.

Responsibility:
    - Load and validate JSONL datasets and JSON configurations.
    - Compute full dataset hashes (provenance) and comparison fingerprints (compatibility).
    - Execute configured metrics sequentially over normalized test cases.
    - Track safety failures, metric statuses, and aggregate run counts.
    - Persist execution results to SQLite storage.

Dependencies:
    - os
    - json
    - hashlib
    - datetime
    - uuid
    - pathlib (Path)
    - typing (Any, Dict, List, Optional, Tuple)
    - deepeval
    - evalhub.models (CaseEvaluationResult, EvaluationCase, EvaluationConfig, MetricResult, MetricStatus, RunSummary)
    - evalhub.adapters (get_adapter, normalize_records)
    - evalhub.metrics (get_metric_instance)
    - evalhub.storage (StorageEngine)

Key Design Decisions:
    - Offline deterministic evaluation by default; optional live connectors and Gemini judges.
    - Separation of dataset hash (provenance) vs comparison fingerprint (compatibility).
    - Sequential execution avoiding unnecessary async complexity for deterministic evaluation.
    - Interrupted runs are recorded as 'incomplete' rather than successful.
"""

from __future__ import annotations
import os
import json
import hashlib
import uuid
import io
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"
import deepeval
from deepeval import evaluate as deepeval_evaluate
from deepeval.config.settings import get_settings
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig, ErrorConfig
from deepeval.test_case import LLMTestCase

from evalhub.models import (
    CaseEvaluationResult,
    EvaluationCase,
    EvaluationConfig,
    MetricConfig,
    MetricResult,
    MetricStatus,
    RunSummary,
)
from evalhub.adapters import normalize_records
from evalhub.connectors import AgentExecutionError, AgentTimeoutError, create_agent
from evalhub.metrics import EvalHubDeepEvalMetric, get_metric_instance
from evalhub.storage import StorageEngine


_DEEPEVAL_EVALUATION_LOCK = RLock()


def compute_file_sha256(file_path: str) -> str:
    """Compute SHA-256 hash of a file's raw content."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_comparison_fingerprint(
    cases: List[EvaluationCase],
    config: EvaluationConfig,
    deepeval_ver: str,
    metric_versions: Dict[str, str],
) -> str:
    """
    Computes a deterministic comparison fingerprint over evaluation requirements.
    Excludes actual outputs, observed tool traces, execution statuses, and timestamps.
    """
    sorted_cases = sorted(cases, key=lambda c: c.case_id)
    canonical_cases_payload = []

    for c in sorted_cases:
        # Canonical representation of requirements and ground truth
        case_dict = {
            "case_id": c.case_id,
            "input": c.input,
            "expected_output": c.expected_output,
            "expected_labels": c.expected_labels,
            "expected_required_tools": sorted(c.expected_required_tools) if c.expected_required_tools else None,
            "evidence": [
                {"source": e.source, "record_id": e.record_id, "fields": e.fields}
                for e in sorted(c.evidence or [], key=lambda x: (x.source, x.record_id))
            ],
            "tags": sorted(c.tags),
        }
        if c.retrieval_context:
            case_dict["retrieval_context"] = c.retrieval_context
        canonical_cases_payload.append(case_dict)

    canonical_config_payload = {
        "metrics": [
            {
                "name": m.name,
                "threshold": m.threshold,
                "required": m.required,
                "options": m.options,
            }
            for m in sorted(config.metrics, key=lambda x: x.name)
        ],
        "judge_enabled": config.judge_enabled,
        "judge_model": config.judge_model if config.judge_enabled else None,
        "deepeval_version": deepeval_ver,
        "metric_versions": metric_versions,
    }

    full_payload = {
        "cases": canonical_cases_payload,
        "config": canonical_config_payload,
    }

    payload_json = json.dumps(full_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def _format_test_case_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _run_deepeval(
    cases: List[EvaluationCase],
    metrics: List[Tuple[MetricConfig, EvalHubDeepEvalMetric]],
) -> Dict[str, List[MetricResult]]:
    if not metrics:
        return {case.case_id: [] for case in cases}

    test_cases = [
        LLMTestCase(
            name=case.case_id,
            input=_format_test_case_value(case.input) or "",
            actual_output=(
                _format_test_case_value(case.actual_output)
                if case.actual_output is not None
                else "<missing actual output>"
            ),
            expected_output=_format_test_case_value(case.expected_output),
            metadata={"evalhub_case": case.model_dump(mode="json")},
        )
        for case in cases
    ]

    settings = get_settings()
    previous_api_key = settings.CONFIDENT_API_KEY
    previous_file_system = settings.DEEPEVAL_FILE_SYSTEM

    # DeepEval's high-level evaluator may upload runs when a Confident AI key is
    # configured. EvalHub evaluates saved runs locally and keeps persistence in SQLite.
    with _DEEPEVAL_EVALUATION_LOCK, settings.edit(persist=False):
        settings.CONFIDENT_API_KEY = None
        settings.DEEPEVAL_FILE_SYSTEM = "READ_ONLY"
        try:
            # DeepEval also writes terminal output; EvalHub renders its own report.
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                evaluation = deepeval_evaluate(
                    test_cases=test_cases,
                    metrics=[metric for _, metric in metrics],
                    async_config=AsyncConfig(run_async=False),
                    cache_config=CacheConfig(write_cache=False, use_cache=False),
                    display_config=DisplayConfig(
                        show_indicator=False,
                        print_results=False,
                        inspect_after_run=False,
                    ),
                    error_config=ErrorConfig(ignore_errors=False),
                )
        finally:
            settings.CONFIDENT_API_KEY = previous_api_key
            settings.DEEPEVAL_FILE_SYSTEM = previous_file_system

    test_results = {result.name: result for result in evaluation.test_results}
    if len(test_results) != len(cases):
        raise RuntimeError(
            f"DeepEval returned {len(test_results)} case results for {len(cases)} cases."
        )

    results_by_case: Dict[str, List[MetricResult]] = {}
    for case in cases:
        test_result = test_results.get(case.case_id)
        if test_result is None:
            raise RuntimeError(
                f"DeepEval did not return a result for case '{case.case_id}'."
            )

        metric_data = test_result.metrics_data or []
        if len(metric_data) != len(metrics):
            raise RuntimeError(
                f"DeepEval returned {len(metric_data)} metric results for case "
                f"'{case.case_id}', expected {len(metrics)}."
            )

        case_metric_results: List[MetricResult] = []
        for (metric_config, metric), deepeval_result in zip(metrics, metric_data):
            if deepeval_result.error:
                result = MetricResult(
                    name=metric_config.name,
                    status=MetricStatus.ERROR,
                    score=None,
                    explanation=f"DeepEval metric execution error: {deepeval_result.error}",
                    metadata={"case_id": case.case_id},
                )
            else:
                result = metric.result_for_case(case.case_id)
                if bool(deepeval_result.success) != (
                    result.status
                    in (MetricStatus.PASSED, MetricStatus.NOT_APPLICABLE)
                ):
                    raise RuntimeError(
                        f"DeepEval and EvalHub disagree on '{metric_config.name}' "
                        f"for case '{case.case_id}'."
                    )
            case_metric_results.append(result)
        results_by_case[case.case_id] = case_metric_results

    return results_by_case


class EvaluationRunner:
    """Executes evaluation runs against datasets and configurations."""

    def __init__(self, storage: Optional[StorageEngine] = None):
        self.storage = storage or StorageEngine()

    @staticmethod
    def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
        """Load JSONL dataset file into raw dicts."""
        path = Path(dataset_path)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    records.append(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at {dataset_path}:{line_no}: {exc}"
                    ) from exc

        if not records:
            raise ValueError(f"Dataset file is empty: {dataset_path}")
        return records

    @staticmethod
    def load_config(config_path: str) -> EvaluationConfig:
        """Load and parse JSON evaluation configuration."""
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(path, "r", encoding="utf-8") as f:
            try:
                raw = json.load(f)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in configuration file {config_path}: {exc}"
                ) from exc

        return EvaluationConfig.model_validate(raw)

    def validate_dataset(
        self, dataset_path: str, adapter_name: str = "generic", input_only: bool = False
    ) -> List[EvaluationCase]:
        """Validate raw dataset file and return normalized cases."""
        raw_records = self.load_dataset(dataset_path)
        if input_only:
            if adapter_name != "generic":
                raise ValueError("Input-only datasets require the generic adapter.")
            raw_records = self._prepare_inputs(raw_records)
        normalized = normalize_records(raw_records, adapter_name)
        return normalized

    @staticmethod
    def _prepare_inputs(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prepared = []
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("Each dataset record must be a JSON object.")
            prepared.append({
                **record,
                "actual_output": None,
                "execution_status": "failed",
                "execution_error": None,
                "tool_calls": None,
            })
        return prepared

    def run_evaluation(
        self,
        dataset_path: str,
        config_path: str,
        label: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Tuple[RunSummary, List[CaseEvaluationResult]]:
        """
        Execute an evaluation run, optionally invoking a configured agent.
        Normalizes cases, evaluates metrics, computes fingerprints, and saves to storage.
        """
        run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
        created_at = datetime.now(timezone.utc).isoformat()

        # Load inputs
        config = self.load_config(config_path)
        raw_records = self.load_dataset(dataset_path)
        dataset_hash = compute_file_sha256(dataset_path)
        config_hash = compute_file_sha256(config_path)
        if config.agent is not None:
            if config.adapter != "generic":
                raise ValueError("Live agent evaluation requires the generic adapter.")
            raw_records = self._prepare_inputs(raw_records)
        normalized_cases = normalize_records(raw_records, config.adapter)
        fingerprint_cases = list(normalized_cases)

        # Validate all metric and connector configuration before invoking an agent.
        instantiated_metrics: List[Tuple[MetricConfig, EvalHubDeepEvalMetric]] = [
            (
                m_cfg,
                get_metric_instance(m_cfg, judge_enabled=config.judge_enabled, judge_model=config.judge_model),
            )
            for m_cfg in config.metrics
        ]
        if config.agent is not None:
            agent = create_agent(config.agent)
            for index, case in enumerate(normalized_cases):
                data = case.model_dump(mode="python")
                try:
                    response = agent(case.input)
                    data.update({
                        "actual_output": response.actual_output,
                        "tool_calls": response.tool_calls,
                        "execution_status": "completed",
                        "execution_error": None,
                    })
                    if response.retrieval_context is not None:
                        data["retrieval_context"] = response.retrieval_context
                except AgentExecutionError as exc:
                    data["execution_error"] = str(exc)
                    if isinstance(exc, AgentTimeoutError):
                        data["execution_status"] = "timed_out"
                normalized_cases[index] = EvaluationCase.model_validate(data)

        # Hashes & Provenance
        deepeval_ver = getattr(deepeval, "__version__", "4.3.4")

        metric_versions = {m.name: "3.0.0" for m in config.metrics}
        comparison_fingerprint = compute_comparison_fingerprint(
            cases=fingerprint_cases,
            config=config,
            deepeval_ver=deepeval_ver,
            metric_versions=metric_versions,
        )

        case_results: List[CaseEvaluationResult] = []
        total_cases = len(normalized_cases)
        passed_cases = 0
        failed_cases = 0
        error_cases = 0
        not_applicable_cases = 0
        total_safety_failures = 0

        try:
            metric_results_by_case = _run_deepeval(
                normalized_cases, instantiated_metrics
            )
            for case in normalized_cases:
                metric_results = metric_results_by_case[case.case_id]
                case_has_failure = False
                case_has_error = False
                case_has_safety_failure = False

                for result in metric_results:
                    if result.is_safety_failure:
                        case_has_safety_failure = True
                        total_safety_failures += 1

                    if result.status == MetricStatus.FAILED:
                        case_has_failure = True
                    elif result.status == MetricStatus.ERROR:
                        case_has_error = True

                # Determine overall case status
                if case.actual_output is None:
                    overall_status = MetricStatus.ERROR
                    error_cases += 1
                elif case_has_safety_failure or case_has_failure:
                    overall_status = MetricStatus.FAILED
                    failed_cases += 1
                elif case_has_error:
                    overall_status = MetricStatus.ERROR
                    error_cases += 1
                elif all(m.status == MetricStatus.NOT_APPLICABLE for m in metric_results):
                    overall_status = MetricStatus.NOT_APPLICABLE
                    not_applicable_cases += 1
                else:
                    overall_status = MetricStatus.PASSED
                    passed_cases += 1

                case_results.append(
                    CaseEvaluationResult(
                        case=case,
                        metric_results=metric_results,
                        overall_status=overall_status,
                    )
                )

            summary = RunSummary(
                run_id=run_id,
                label=label,
                created_at=created_at,
                run_status="completed",
                dataset_path=dataset_path,
                dataset_hash=dataset_hash,
                comparison_fingerprint=comparison_fingerprint,
                config_hash=config_hash,
                config_json=config.model_dump_json(),
                deepeval_version=deepeval_ver,
                metric_versions_json=json.dumps(metric_versions),
                total_cases=total_cases,
                passed_cases=passed_cases,
                failed_cases=failed_cases,
                error_cases=error_cases,
                not_applicable_cases=not_applicable_cases,
                safety_failures=total_safety_failures,
            )

            # Transactional save
            self.storage.save_run(summary, case_results)
            return summary, case_results

        except Exception as exc:
            # Mark incomplete on failure
            self.storage.mark_run_incomplete(run_id, str(exc))
            raise
