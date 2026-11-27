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
from string import Formatter
from urllib.parse import parse_qsl, urlsplit
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
    input: Any = Field(default=None, description="Original task supplied to the agent")
    lookup: Optional[Dict[str, Any]] = None
    retrieval: Optional[Dict[str, Any]] = None
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
        if self.input is None and not self.lookup:
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
    method: Literal["POST", "GET"] = "POST"
    query_params: Dict[str, str] = Field(default_factory=dict)
    path_params: Dict[str, str] = Field(default_factory=dict)
    input_path: Optional[str] = None
    run_id_path: Optional[str] = None
    status_path: Optional[str] = None
    explanation_path: Optional[str] = None
    evidence_path: Optional[str] = None
    completed_statuses: List[str] = Field(default_factory=lambda: ["completed", "success"])

    @model_validator(mode="after")
    def validate_request_mapping(self) -> HTTPAgentConfig:
        url = urlsplit(self.url)
        if (
            url.scheme not in ("http", "https") or not url.hostname
            or url.username or url.password or url.fragment
        ):
            raise ValueError("Agent URL must be HTTP(S), without credentials or a fragment.")
        if self.method == "POST" and (self.query_params or self.path_params):
            raise ValueError("Request parameter mappings require GET.")
        fields = []
        for _, name, spec, conversion in Formatter().parse(self.url):
            if name is not None:
                if spec or conversion or not name.isidentifier():
                    raise ValueError("URL placeholders must be simple parameter names.")
                fields.append(name)
        if set(fields) != set(self.path_params):
            raise ValueError("URL placeholders must match path_params.")
        if any("{" + name + "}" not in url.path for name in fields):
            raise ValueError("URL placeholders are only allowed in the path.")
        if any(not key or not value for key, value in
               [*self.query_params.items(), *self.path_params.items()]):
            raise ValueError("Parameter mappings must have non-empty names and lookup fields.")
        existing = {key for key, _ in parse_qsl(url.query, keep_blank_values=True)}
        if existing.intersection(self.query_params):
            raise ValueError("Mapped query parameters must not already occur in the URL.")
        if self.status_path and not self.completed_statuses:
            raise ValueError("completed_statuses is required when status_path is set.")
        return self


class AgentRegistration(BaseModel):
    """A versioned, read-only source of existing agent results."""
    model_config = ConfigDict(extra="forbid")
    agent_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    name: str = Field(min_length=1)
    enabled: bool = True
    connector: HTTPAgentConfig
    lookup_fields: Dict[str, Literal["string", "integer", "number", "boolean"]] = Field(
        default_factory=lambda: {"run_id": "string"}
    )

    @model_validator(mode="after")
    def validate_results_source(self) -> AgentRegistration:
        if not self.name.strip():
            raise ValueError("Agent name must not be blank.")
        if self.connector.method != "GET":
            raise ValueError("Registered result sources must use GET.")
        mapped = set(self.connector.query_params.values()) | set(self.connector.path_params.values())
        if not mapped or mapped != set(self.lookup_fields):
            raise ValueError("lookup_fields must match the fields used in request mappings.")
        return self

    def validate_lookup(self, lookup: Dict[str, Any]) -> None:
        if set(lookup) != set(self.lookup_fields):
            raise ValueError(f"Lookup must contain exactly: {', '.join(self.lookup_fields)}.")
        for name, kind in self.lookup_fields.items():
            value = lookup[name]
            valid = {
                "string": isinstance(value, str) and bool(value.strip()),
                "integer": type(value) is int,
                "number": type(value) in (int, float),
                "boolean": type(value) is bool,
            }[kind]
            if not valid:
                raise ValueError(f"Lookup field '{name}' must be a non-empty {kind}.")


class RegisteredAgent(AgentRegistration):
    revision: int = Field(ge=1)
    created_at: str


class AgentReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent_id: str = Field(min_length=1)
    revision: Optional[int] = Field(default=None, ge=1)


class PythonAgentConfig(BaseModel):
    """Configuration for a trusted importable Python agent callable."""
    model_config = ConfigDict(extra="forbid")
    type: Literal["python"] = "python"
    callable: str = Field(..., min_length=1)
    response_path: Optional[str] = None
    tool_calls_path: Optional[str] = None
    retrieval_context_path: Optional[str] = None


class AgentTestCase(BaseModel):
    """Input and optional deterministic expectations for a registered live-agent test."""
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(..., min_length=1)
    input: Any
    expected_output: Optional[Any] = None
    expected_labels: Optional[Dict[str, Any]] = None
    expected_required_tools: Optional[List[str]] = None
    evidence: Optional[List[EvidenceRecord]] = None
    retrieval_context: Optional[List[str]] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_agent_input(self) -> AgentTestCase:
        if self.input is None:
            raise ValueError(f"Test case '{self.case_id}' must include a non-null input.")
        return self


class InvokableAgentRegistration(BaseModel):
    """Versioned agent invocation profile and its evaluation suite."""
    model_config = ConfigDict(extra="forbid")
    agent_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    name: str = Field(min_length=1)
    enabled: bool = True
    connector: Union[HTTPAgentConfig, PythonAgentConfig] = Field(discriminator="type")
    test_cases: List[AgentTestCase] = Field(min_length=1)
    reference_dossiers: List[Any] = Field(
        min_length=1,
        description="Reference investigations for judge calibration, not case-specific ground truth.",
    )
    metrics: List[MetricConfig] = Field(min_length=1)
    judge_model: Optional[str] = None

    @model_validator(mode="after")
    def validate_profile(self) -> InvokableAgentRegistration:
        if not self.name.strip():
            raise ValueError("Agent name must not be blank.")
        if isinstance(self.connector, HTTPAgentConfig) and self.connector.method != "POST":
            raise ValueError("Invokable HTTP agents must use POST.")
        case_ids = [case.case_id for case in self.test_cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("Agent test case IDs must be unique.")
        judge_metrics = {
            "answer_relevancy", "answer_correctness", "faithfulness",
            "investigation_quality", "investigation_correctness", "investigation_tone",
        }
        if any(metric.name in judge_metrics for metric in self.metrics):
            if not self.judge_model or not self.judge_model.strip():
                raise ValueError("A judge_model is required for LLM judge metrics.")
        return self


class RegisteredInvokableAgent(InvokableAgentRegistration):
    revision: int = Field(ge=1)
    created_at: str


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
    agent_ref: Optional[AgentReference] = None
    agent_profile_ref: Optional[AgentReference] = None
    judge_reference_dossiers: List[Any] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_agent_selection(self) -> EvaluationConfig:
        if self.agent is not None and self.agent_ref is not None:
            raise ValueError("Choose either agent or agent_ref, not both.")
        if self.agent_profile_ref is not None and (
            self.agent is not None or self.agent_ref is not None
        ):
            raise ValueError("Choose only one of agent, agent_ref, or agent_profile_ref.")
        return self


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
