"""
storage.py

Purpose:
    SQLite persistence layer for EvalHub runs, normalized test cases, and metric results.

Responsibility:
    - Initialize database schema and migrate the registry to schema version 2.
    - Save complete evaluation runs and per-case results transactionally.
    - Retrieve runs, case-level diagnostics, and run listings.
    - Provide comparison diff logic across baseline and candidate runs.

Dependencies:
    - sqlite3
    - json
    - typing (Any, Dict, List, Optional, Tuple)
    - evalhub.models (CaseEvaluationResult, ComparisonDiff, MetricStatus, RunSummary)

Key Design Decisions:
    - Enforces PRAGMA foreign_keys = ON.
    - Uses PRAGMA user_version for lightweight schema versioning.
    - Stores runs and case_results with composite uniqueness on (run_id, case_id).
    - Interrupted runs are marked 'incomplete' rather than successful.
    - Parameterized SQL for security and reliability.
"""

from __future__ import annotations
import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from evalhub.models import (
    AgentRegistration, InvokableAgentRegistration, RegisteredAgent,
    RegisteredInvokableAgent, CaseEvaluationResult, ComparisonDiff, MetricStatus,
    RunSummary,
)

SCHEMA_VERSION = 3

CREATE_AGENTS_TABLE = """
CREATE TABLE IF NOT EXISTS agent_revisions (
    agent_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    registration_json TEXT NOT NULL,
    PRIMARY KEY (agent_id, revision)
);
"""

CREATE_INVOKABLE_AGENTS_TABLE = """
CREATE TABLE IF NOT EXISTS invokable_agent_revisions (
    agent_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    registration_json TEXT NOT NULL,
    PRIMARY KEY (agent_id, revision)
);
"""

CREATE_RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    label TEXT,
    created_at TEXT NOT NULL,
    run_status TEXT NOT NULL,
    dataset_path TEXT NOT NULL,
    dataset_hash TEXT NOT NULL,
    comparison_fingerprint TEXT NOT NULL,
    config_json TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    deepeval_version TEXT NOT NULL,
    metric_versions_json TEXT NOT NULL,
    total_cases INTEGER NOT NULL,
    passed_cases INTEGER NOT NULL,
    failed_cases INTEGER NOT NULL,
    error_cases INTEGER NOT NULL,
    not_applicable_cases INTEGER NOT NULL,
    safety_failures INTEGER NOT NULL DEFAULT 0
);
"""

CREATE_CASE_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS case_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    normalized_case_json TEXT NOT NULL,
    execution_status TEXT NOT NULL,
    overall_status TEXT NOT NULL,
    metric_results_json TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
    UNIQUE(run_id, case_id)
);
"""

CREATE_INDICES = """
CREATE INDEX IF NOT EXISTS idx_case_results_run_id ON case_results(run_id);
CREATE INDEX IF NOT EXISTS idx_case_results_case_id ON case_results(case_id);
"""


class StorageEngine:
    """SQLite storage engine for evaluation runs and results."""

    def __init__(self, db_path: str = "evalhub.db"):
        self.db_path = str(Path(db_path).resolve())
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Initialize tables and user_version schema pragma."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA user_version;")
            current_version = cursor.fetchone()[0]
            if current_version > SCHEMA_VERSION:
                raise ValueError(f"Database schema {current_version} is newer than supported.")

            cursor.execute(CREATE_RUNS_TABLE)
            cursor.execute(CREATE_CASE_RESULTS_TABLE)
            cursor.executescript(CREATE_INDICES)
            cursor.execute(CREATE_AGENTS_TABLE)
            cursor.execute(CREATE_INVOKABLE_AGENTS_TABLE)

            if current_version < SCHEMA_VERSION:
                cursor.execute(f"PRAGMA user_version = {SCHEMA_VERSION};")
            conn.commit()

    def save_agent(self, registration: AgentRegistration, *, update: bool = False) -> RegisteredAgent:
        """Append a registry revision without changing historical definitions."""
        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT MAX(revision) AS revision FROM agent_revisions WHERE agent_id = ?",
                (registration.agent_id,),
            ).fetchone()
            revision = row["revision"]
            if revision is not None and not update:
                raise ValueError(f"Agent '{registration.agent_id}' already exists; use update.")
            if revision is None and update:
                raise ValueError(f"Agent '{registration.agent_id}' not found.")
            agent = RegisteredAgent(
                **registration.model_dump(),
                revision=(revision or 0) + 1,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            conn.execute(
                "INSERT INTO agent_revisions VALUES (?, ?, ?)",
                (agent.agent_id, agent.revision, agent.model_dump_json()),
            )
        return agent

    def get_agent(self, agent_id: str, revision: Optional[int] = None) -> RegisteredAgent:
        with self._get_connection() as conn:
            row = conn.execute(
                """SELECT registration_json FROM agent_revisions
                   WHERE agent_id = ? AND (? IS NULL OR revision = ?)
                   ORDER BY revision DESC LIMIT 1""",
                (agent_id, revision, revision),
            ).fetchone()
        if row is None:
            raise ValueError(f"Agent '{agent_id}' or requested revision not found.")
        return RegisteredAgent.model_validate_json(row["registration_json"])

    def list_agents(self) -> List[RegisteredAgent]:
        with self._get_connection() as conn:
            rows = conn.execute(
                """SELECT a.registration_json FROM agent_revisions a
                   JOIN (SELECT agent_id, MAX(revision) AS revision FROM agent_revisions
                         GROUP BY agent_id) latest
                   ON a.agent_id = latest.agent_id AND a.revision = latest.revision
                   ORDER BY a.agent_id"""
            ).fetchall()
        return [RegisteredAgent.model_validate_json(row["registration_json"]) for row in rows]

    def get_active_agent(self, agent_id: str, revision: Optional[int] = None) -> RegisteredAgent:
        latest = self.get_agent(agent_id)
        agent = self.get_agent(agent_id, revision) if revision is not None else latest
        if not latest.enabled or not agent.enabled:
            raise ValueError(f"Agent '{agent_id}' is disabled.")
        return agent

    def disable_agent(self, agent_id: str) -> RegisteredAgent:
        current = self.get_agent(agent_id)
        registration = AgentRegistration.model_validate({
            **current.model_dump(exclude={"revision", "created_at"}), "enabled": False,
        })
        return self.save_agent(registration, update=True)

    def save_invokable_agent(
        self, registration: InvokableAgentRegistration, *, update: bool = False
    ) -> RegisteredInvokableAgent:
        """Append a runnable-agent revision with its immutable evaluation suite."""
        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT MAX(revision) AS revision FROM invokable_agent_revisions WHERE agent_id = ?",
                (registration.agent_id,),
            ).fetchone()
            revision = row["revision"]
            if revision is not None and not update:
                raise ValueError(
                    f"Invokable agent '{registration.agent_id}' already exists; use update."
                )
            if revision is None and update:
                raise ValueError(f"Invokable agent '{registration.agent_id}' not found.")
            agent = RegisteredInvokableAgent(
                **registration.model_dump(),
                revision=(revision or 0) + 1,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            conn.execute(
                "INSERT INTO invokable_agent_revisions VALUES (?, ?, ?)",
                (agent.agent_id, agent.revision, agent.model_dump_json()),
            )
        return agent

    def get_invokable_agent(
        self, agent_id: str, revision: Optional[int] = None
    ) -> RegisteredInvokableAgent:
        with self._get_connection() as conn:
            row = conn.execute(
                """SELECT registration_json FROM invokable_agent_revisions
                   WHERE agent_id = ? AND (? IS NULL OR revision = ?)
                   ORDER BY revision DESC LIMIT 1""",
                (agent_id, revision, revision),
            ).fetchone()
        if row is None:
            raise ValueError(
                f"Invokable agent '{agent_id}' or requested revision not found."
            )
        return RegisteredInvokableAgent.model_validate_json(row["registration_json"])

    def list_invokable_agents(self) -> List[RegisteredInvokableAgent]:
        with self._get_connection() as conn:
            rows = conn.execute(
                """SELECT a.registration_json FROM invokable_agent_revisions a
                   JOIN (
                       SELECT agent_id, MAX(revision) AS revision
                       FROM invokable_agent_revisions GROUP BY agent_id
                   ) latest
                   ON a.agent_id = latest.agent_id AND a.revision = latest.revision
                   ORDER BY a.agent_id"""
            ).fetchall()
        return [
            RegisteredInvokableAgent.model_validate_json(row["registration_json"])
            for row in rows
        ]

    def get_active_invokable_agent(
        self, agent_id: str, revision: Optional[int] = None
    ) -> RegisteredInvokableAgent:
        latest = self.get_invokable_agent(agent_id)
        agent = self.get_invokable_agent(agent_id, revision) if revision is not None else latest
        if not latest.enabled or not agent.enabled:
            raise ValueError(f"Invokable agent '{agent_id}' is disabled.")
        return agent

    def disable_invokable_agent(self, agent_id: str) -> RegisteredInvokableAgent:
        current = self.get_invokable_agent(agent_id)
        registration = InvokableAgentRegistration.model_validate(
            {**current.model_dump(exclude={"revision", "created_at"}), "enabled": False}
        )
        return self.save_invokable_agent(registration, update=True)

    def get_schema_version(self) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA user_version;")
            return int(cursor.fetchone()[0])

    def save_run(self, summary: RunSummary, case_results: List[CaseEvaluationResult]) -> None:
        """Save a complete run summary and all individual case results inside a single transaction."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id, label, created_at, run_status, dataset_path,
                    dataset_hash, comparison_fingerprint, config_json, config_hash,
                    deepeval_version, metric_versions_json, total_cases, passed_cases,
                    failed_cases, error_cases, not_applicable_cases, safety_failures
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    summary.run_id,
                    summary.label,
                    summary.created_at,
                    summary.run_status,
                    summary.dataset_path,
                    summary.dataset_hash,
                    summary.comparison_fingerprint,
                    summary.config_json,
                    summary.config_hash,
                    summary.deepeval_version,
                    summary.metric_versions_json,
                    summary.total_cases,
                    summary.passed_cases,
                    summary.failed_cases,
                    summary.error_cases,
                    summary.not_applicable_cases,
                    summary.safety_failures,
                )
            )

            # Insert case results
            for cr in case_results:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO case_results (
                        run_id, case_id, normalized_case_json, execution_status,
                        overall_status, metric_results_json
                    ) VALUES (?, ?, ?, ?, ?, ?);
                    """,
                    (
                        summary.run_id,
                        cr.case.case_id,
                        cr.case.model_dump_json(),
                        cr.case.execution_status.value,
                        cr.overall_status.value,
                        json.dumps([m.model_dump() for m in cr.metric_results]),
                    )
                )
            conn.commit()

    def mark_run_incomplete(self, run_id: str, error_message: str) -> None:
        """Mark an in-progress or interrupted run as incomplete."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE runs SET run_status = 'incomplete' WHERE run_id = ?;",
                (run_id,)
            )
            conn.commit()

    def get_run(self, run_id: str) -> Optional[RunSummary]:
        """Fetch a run summary by run_id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM runs WHERE run_id = ?;", (run_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return RunSummary(**dict(row))

    def list_runs(self) -> List[RunSummary]:
        """List all runs ordered by creation timestamp descending."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM runs ORDER BY created_at DESC;")
            return [RunSummary(**dict(row)) for row in cursor.fetchall()]

    def get_case_results(self, run_id: str) -> List[Dict[str, Any]]:
        """Retrieve all case results and metric details for a specific run."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM case_results WHERE run_id = ? ORDER BY case_id ASC;",
                (run_id,)
            )
            rows = cursor.fetchall()
            results = []
            for row in rows:
                results.append({
                    "id": row["id"],
                    "run_id": row["run_id"],
                    "case_id": row["case_id"],
                    "case": json.loads(row["normalized_case_json"]),
                    "execution_status": row["execution_status"],
                    "overall_status": row["overall_status"],
                    "metric_results": json.loads(row["metric_results_json"])
                })
            return results

    def compare_runs(self, baseline_run_id: str, candidate_run_id: str) -> ComparisonDiff:
        """
        Compare baseline and candidate runs.
        Validates comparison fingerprints for evaluation compatibility.
        """
        baseline = self.get_run(baseline_run_id)
        candidate = self.get_run(candidate_run_id)

        if not baseline:
            raise ValueError(f"Baseline run '{baseline_run_id}' not found in storage.")
        if not candidate:
            raise ValueError(f"Candidate run '{candidate_run_id}' not found in storage.")

        # Check comparison fingerprint compatibility
        fingerprint_match = (baseline.comparison_fingerprint == candidate.comparison_fingerprint)
        if not fingerprint_match:
            return ComparisonDiff(
                baseline_run_id=baseline_run_id,
                candidate_run_id=candidate_run_id,
                fingerprint_match=False,
                status="incompatible",
                message=(
                    "Incompatible comparison: baseline and candidate have different evaluation "
                    "fingerprints (e.g. modified labels, inputs, required tools, evidence, or metric settings)."
                )
            )

        baseline_cases = {cr["case_id"]: cr for cr in self.get_case_results(baseline_run_id)}
        candidate_cases = {cr["case_id"]: cr for cr in self.get_case_results(candidate_run_id)}

        b_ids = set(baseline_cases.keys())
        c_ids = set(candidate_cases.keys())

        missing_cases = sorted(list(b_ids - c_ids))
        extra_cases = sorted(list(c_ids - b_ids))
        common_ids = sorted(list(b_ids.intersection(c_ids)))

        new_failures: List[Dict[str, Any]] = []
        fixed_cases: List[Dict[str, Any]] = []
        new_errors: List[Dict[str, Any]] = []
        score_changes: Dict[str, Dict[str, Any]] = {}

        for cid in common_ids:
            b_res = baseline_cases[cid]
            c_res = candidate_cases[cid]

            b_status = b_res["overall_status"]
            c_status = c_res["overall_status"]

            # Regressions / Fixes
            if b_status == MetricStatus.PASSED.value and c_status == MetricStatus.FAILED.value:
                new_failures.append({
                    "case_id": cid,
                    "baseline_status": b_status,
                    "candidate_status": c_status,
                    "metrics": c_res["metric_results"]
                })
            elif b_status != MetricStatus.PASSED.value and c_status == MetricStatus.PASSED.value:
                fixed_cases.append({
                    "case_id": cid,
                    "baseline_status": b_status,
                    "candidate_status": c_status
                })
            elif b_status != MetricStatus.ERROR.value and c_status == MetricStatus.ERROR.value:
                new_errors.append({
                    "case_id": cid,
                    "baseline_status": b_status,
                    "candidate_status": c_status,
                    "metrics": c_res["metric_results"]
                })

            # Score breakdown
            b_metric_map = {m["name"]: m.get("score") for m in b_res["metric_results"]}
            c_metric_map = {m["name"]: m.get("score") for m in c_res["metric_results"]}

            for m_name, c_score in c_metric_map.items():
                b_score = b_metric_map.get(m_name)
                if b_score is not None and c_score is not None and b_score != c_score:
                    score_changes.setdefault(m_name, {})[cid] = {
                        "baseline_score": b_score,
                        "candidate_score": c_score,
                        "diff": c_score - b_score
                    }

        diff_status = "regressed" if (new_failures or new_errors or missing_cases) else "passed"

        return ComparisonDiff(
            baseline_run_id=baseline_run_id,
            candidate_run_id=candidate_run_id,
            fingerprint_match=True,
            status=diff_status,
            message=f"Comparison completed: {len(new_failures)} new failures, {len(new_errors)} new errors, {len(fixed_cases)} fixes.",
            new_failures=new_failures,
            fixed_cases=fixed_cases,
            new_errors=new_errors,
            missing_cases=missing_cases,
            extra_cases=extra_cases,
            score_changes=score_changes,
            safety_failures_baseline=baseline.safety_failures,
            safety_failures_candidate=candidate.safety_failures
        )
