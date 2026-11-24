"""
models.py

Purpose:
    Canonical data contracts and domain models for EvalHub.

Responsibility:
    - Define validation schemas for normalized evaluation cases, tool calls, and evidence.
    - Define evaluation configurations, metric results, run summaries, and comparison contracts.

Dependencies:
    - pydantic (BaseModel, Field, model_validator)
    - typing (Any, Dict, List, Literal, Optional, Union)

Key Design Decisions:
    - Flexible representation of actual_output and input (str or dict).
    - Explicit preservation of failed agent runs (actual_output is None only when execution_status != "completed").
    - Preservation of missing tool traces (None) as distinct from empty tool traces ([]).
    - Status enums covering passed, failed, error, and not_applicable.
    - Safety failure isolation flag so critical safety failures are not masked by aggregate scores.
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExecutionStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class MetricStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    NOT_APPLICABLE = "not_applicable"


class ToolCallRecord(BaseModel):
    """Represents a recorded tool execution from an agent trace."""
    name: str = Field(..., description="Tool or function name invoked")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments passed to the tool")
    result: Optional[Any] = Field(default=None, description="Result returned by tool invocation")
    status: Optional[str] = Field(default="success", description="Execution status of the tool call")
    call_id: Optional[str] = Field(default=None, description="Optional unique identifier for call")


class EvidenceRecord(BaseModel):
    """Represents an authoritative or claimed evidence item."""
    source: str = Field(..., min_length=1, description="Evidence source identifier")
    record_id: str = Field(..., min_length=1, description="Evidence record identifier")
    fields: Dict[str, Any] = Field(default_factory=dict, description="Key-value fields belonging to the record")


class EvaluationCase(BaseModel):
    """
    Canonical normalized evaluation case.
    All external datasets are mapped to this shape via adapters.
    """
    case_id: str = Field(..., min_length=1, description="Stable, unique case identifier within dataset")
    input: Any = Field(..., description="Input supplied to the agent")
    actual_output: Optional[Any] = Field(
        default=None,
        description="Observed agent response. Required if execution_status == completed."
    )
    expected_output: Optional[Any] = Field(
        default=None,
        description="Optional expected response or target value"
    )
    expected_labels: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional domain-specific ground-truth labels"
    )
    tool_calls: Optional[List[ToolCallRecord]] = Field(
        default=None,
        description="Observed tool execution trace. None indicates missing/unobserved trace; [] indicates no tools called."
    )
    expected_required_tools: Optional[List[str]] = Field(
        default=None,
        description="List of tool names required to have been invoked for this case"
    )
    evidence: Optional[List[EvidenceRecord]] = Field(
        default=None,
        description="Optional authoritative records used to verify output references"
    )
    retrieval_context: Optional[List[str]] = Field(
        default=None,
        description="Reference passages supplied to grounding metrics",
    )
    tags: List[str] = Field(default_factory=list, description="Categorization tags")
    source_run_id: Optional[str] = Field(default=None, description="Original execution run ID from agent system")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional source-specific data preserved by adapters",
    )
    execution_status: ExecutionStatus = Field(
        default=ExecutionStatus.COMPLETED,
        description="Status of the original agent run"
    )
    execution_error: Optional[str] = Field(
        default=None,
        description="Error message if the original agent run failed"
    )

    @model_validator(mode="after")
    def validate_completed_execution_has_output(self) -> EvaluationCase:
        if self.input is None:
            raise ValueError(f"Case '{self.case_id}' must include a non-null input.")
        if self.execution_status == ExecutionStatus.COMPLETED and self.actual_output is None:
            raise ValueError(f"Case '{self.case_id}' has execution_status='completed' but actual_output is None.")
        return self


class MetricResult(BaseModel):
    """Evaluation result for a single metric on a single test case."""
    name: str = Field(..., description="Name of the evaluated metric")
    status: MetricStatus = Field(..., description="Outcome status: passed, failed, error, not_applicable")
    score: Optional[float] = Field(default=None, description="Normalized score [0.0, 1.0], or None for error/N/A")
    explanation: str = Field(..., description="Detailed diagnostic rationale for pass, failure, or error")
    is_safety_failure: bool = Field(default=False, description="True if this failure represents a critical safety breach")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metric-specific diagnostic data")


class CaseEvaluationResult(BaseModel):
    """Combined metric outcomes for a specific evaluation case."""
    case: EvaluationCase
    metric_results: List[MetricResult]
    overall_status: MetricStatus


class MetricConfig(BaseModel):
    """Configuration for an individual metric."""
    name: str = Field(..., description="Registered metric name")
    threshold: float = Field(default=1.0, ge=0.0, le=1.0, description="Minimum score required to pass")
    required: bool = Field(default=True, description="Whether missing prerequisites cause an error vs not_applicable")
    options: Dict[str, Any] = Field(default_factory=dict, description="Metric-specific configuration options")


class HTTPAgentConfig(BaseModel):
    """Configuration for an HTTP JSON agent endpoint."""
    model_config = ConfigDict(extra="forbid")
    type: Literal["http"] = "http"
    url: str
    input_field: str = Field(default="input", min_length=1)
    response_path: Optional[str] = None
    tool_calls_path: Optional[str] = None
    retrieval_context_path: Optional[str] = None
    auth_env: Optional[str] = None
    auth_header: str = "Authorization"
    auth_prefix: str = "Bearer "
    timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    retries: int = Field(default=0, ge=0, le=3)


class PythonAgentConfig(BaseModel):
    """Configuration for a trusted importable Python agent callable."""
    model_config = ConfigDict(extra="forbid")
    type: Literal["python"] = "python"
    callable: str = Field(..., min_length=1)
    response_path: Optional[str] = None
    tool_calls_path: Optional[str] = None
    retrieval_context_path: Optional[str] = None


class EvaluationConfig(BaseModel):
    """Complete evaluation configuration loaded from JSON."""
    adapter: str = Field(default="generic", description="Input adapter name to normalize records")
    judge_enabled: bool = Field(
        default=False,
        description="Enable the explanation heuristic and optional Gemini judge metrics",
    )
    judge_model: Optional[str] = Field(
        default=None,
        description="Gemini model name for optional LLM-based metrics",
    )
    agent: Optional[Union[HTTPAgentConfig, PythonAgentConfig]] = Field(
        default=None,
        discriminator="type",
        description="Optional live agent connector; omitted for saved-output evaluation",
    )
    metrics: List[MetricConfig] = Field(default_factory=list, description="List of configured metrics")


class RunSummary(BaseModel):
    """Metadata and aggregate counts for an evaluation run."""
    run_id: str
    label: Optional[str] = None
    created_at: str
    run_status: str = "completed"
    dataset_path: str
    dataset_hash: str
    comparison_fingerprint: str
    config_hash: str
    config_json: str
    deepeval_version: str
    metric_versions_json: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    error_cases: int
    not_applicable_cases: int
    safety_failures: int = 0


class ComparisonDiff(BaseModel):
    """Comparison result between a baseline run and a candidate run."""
    baseline_run_id: str
    candidate_run_id: str
    fingerprint_match: bool
    status: str
    message: Optional[str] = None
    new_failures: List[Dict[str, Any]] = Field(default_factory=list)
    fixed_cases: List[Dict[str, Any]] = Field(default_factory=list)
    new_errors: List[Dict[str, Any]] = Field(default_factory=list)
    missing_cases: List[str] = Field(default_factory=list)
    extra_cases: List[str] = Field(default_factory=list)
    score_changes: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    safety_failures_baseline: int = 0
    safety_failures_candidate: int = 0
