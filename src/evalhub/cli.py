"""
cli.py

Purpose:
    Command-line interface for EvalHub evaluation and comparison workflows.

Responsibility:
    - Dispatch commands: validate, run, compare, list-runs.
    - Format human-readable terminal output and optional JSON reports.
    - Enforce contractual exit codes:
        0: Passed quality checks / validation success / no regression.
        1: Failed quality checks / candidate regressions.
        2: Invalid input / schema error / incompatible comparison.

Dependencies:
    - argparse
    - sys
    - json
    - evalhub.models (ComparisonDiff, MetricStatus, RunSummary)
    - evalhub.runner (EvaluationRunner)
    - evalhub.storage (StorageEngine)

Key Design Decisions:
    - Built using standard library argparse without heavy CLI frameworks.
    - Safety failures are surfaced with prominent visual alerts.
    - Contractual exit codes are strictly preserved.
"""

from __future__ import annotations
import argparse
import sys
import json
from pathlib import Path
from typing import List, Optional
from evalhub.runner import EvaluationRunner
from evalhub.storage import StorageEngine
from evalhub.models import MetricStatus


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate dataset records using specified adapter."""
    runner = EvaluationRunner()
    print(f"\n[EvalHub] Validating dataset: {args.dataset} (adapter: {args.adapter})")
    try:
        cases = runner.validate_dataset(
            args.dataset, adapter_name=args.adapter, input_only=args.input_only
        )
        print(f"[OK] Successfully validated {len(cases)} records. All case IDs unique.")
        for c in cases[:3]:
            status = "input-only" if args.input_only else c.execution_status.value
            print(f"  - Case ID: {c.case_id} | Status: {status} | Tags: {c.tags}")
        if len(cases) > 3:
            print(f"  ... and {len(cases) - 3} more records.")
        return 0
    except Exception as exc:
        print(f"\n[ERROR] Dataset validation failed: {exc}", file=sys.stderr)
        return 2


def cmd_run(args: argparse.Namespace) -> int:
    """Execute evaluation run and persist results."""
    storage = StorageEngine(db_path=args.db)
    runner = EvaluationRunner(storage=storage)

    print(f"\n[EvalHub] Starting evaluation...")
    print(f"  Dataset: {args.dataset}")
    print(f"  Config:  {args.config}")
    if args.label:
        print(f"  Label:   {args.label}")

    try:
        summary, case_results = runner.run_evaluation(
            dataset_path=args.dataset,
            config_path=args.config,
            label=args.label,
        )
    except Exception as exc:
        print(f"\n[ERROR] Evaluation execution failed: {exc}", file=sys.stderr)
        return 2

    print("\n" + "=" * 60)
    print(f" EVALUATION SUMMARY: Run ID {summary.run_id}")
    print("=" * 60)
    print(f" Total Cases:        {summary.total_cases}")
    print(f" Passed:             {summary.passed_cases}")
    print(f" Failed:             {summary.failed_cases}")
    print(f" Errors:             {summary.error_cases}")
    print(f" Not Applicable:     {summary.not_applicable_cases}")
    print(f" Safety Failures:    {summary.safety_failures}")
    print(f" Comparison Hash:    {summary.comparison_fingerprint[:16]}...")
    print("=" * 60)

    # Highlight safety failures prominently
    if summary.safety_failures > 0:
        print("\n [!] CRITICAL SAFETY FAILURES DETECTED:")
        for cr in case_results:
            for m in cr.metric_results:
                if m.is_safety_failure:
                    print(f"   * Case '{cr.case.case_id}': {m.explanation}")

    # Display failing diagnostics
    failing_cases = [cr for cr in case_results if cr.overall_status != MetricStatus.PASSED]
    if failing_cases:
        print("\n [!] Case Failures & Diagnostics:")
        for cr in failing_cases:
            print(f"\n   - Case: {cr.case.case_id} [Status: {cr.overall_status.value}]")
            if cr.case.execution_error:
                print(f"       [agent]: {cr.case.execution_error}")
            for m in cr.metric_results:
                if m.status in (MetricStatus.FAILED, MetricStatus.ERROR):
                    score_str = f" (Score: {m.score})" if m.score is not None else ""
                    print(f"       [{m.name}]{score_str}: {m.explanation}")

    # Contractual exit code
    if summary.error_cases > 0:
        return 2
    if summary.failed_cases > 0 or summary.safety_failures > 0:
        return 1
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare baseline run against candidate run."""
    storage = StorageEngine(db_path=args.db)

    print(f"\n[EvalHub] Comparing Runs:")
    print(f"  Baseline:  {args.baseline}")
    print(f"  Candidate: {args.candidate}")

    try:
        diff = storage.compare_runs(args.baseline, args.candidate)
    except Exception as exc:
        print(f"\n[ERROR] Comparison failed: {exc}", file=sys.stderr)
        return 2

    if not diff.fingerprint_match:
        print(f"\n[ERROR] {diff.message}", file=sys.stderr)
        return 2

    print("\n" + "=" * 60)
    print(" COMPARISON REPORT")
    print("=" * 60)
    print(f" Status:                  {diff.status.upper()}")
    print(f" New Failures:            {len(diff.new_failures)}")
    print(f" New Errors:              {len(diff.new_errors)}")
    print(f" Fixed Cases:             {len(diff.fixed_cases)}")
    print(f" Missing Cases:           {len(diff.missing_cases)}")
    print(f" Extra Cases:             {len(diff.extra_cases)}")
    print(f" Baseline Safety Breaches: {diff.safety_failures_baseline}")
    print(f" Candidate Safety Breaches:{diff.safety_failures_candidate}")
    print("=" * 60)

    if diff.new_failures:
        print("\n [!] New Regressions in Candidate:")
        for nf in diff.new_failures:
            print(f"   * Case '{nf['case_id']}': regressed from {nf['baseline_status']} to {nf['candidate_status']}")

    if diff.new_errors:
        print("\n [!] New Errors in Candidate:")
        for ne in diff.new_errors:
            print(f"   * Case '{ne['case_id']}': error occurred")

    if diff.fixed_cases:
        print("\n [✓] Improvements in Candidate:")
        for fc in diff.fixed_cases:
            print(f"   * Case '{fc['case_id']}': improved from {fc['baseline_status']} to {fc['candidate_status']}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.write_text(diff.model_dump_json(indent=2), encoding="utf-8")
        print(f"\n[OK] Comparison report exported to: {args.json_out}")

    # Return 1 if regressions detected, 0 if clean
    if diff.status == "regressed":
        return 1
    return 0


def cmd_list_runs(args: argparse.Namespace) -> int:
    """List historical evaluation runs."""
    storage = StorageEngine(db_path=args.db)
    runs = storage.list_runs()

    if not runs:
        print("\n[EvalHub] No evaluation runs found in database.")
        return 0

    print(f"\n[EvalHub] Found {len(runs)} evaluation run(s):")
    print("-" * 80)
    print(f"{'Run ID':<18} {'Label':<15} {'Created At':<22} {'Total':<6} {'Pass':<6} {'Fail':<6} {'Safety':<6}")
    print("-" * 80)
    for r in runs:
        label = r.label or "-"
        created = r.created_at[:19].replace("T", " ")
        print(
            f"{r.run_id:<18} {label:<15} {created:<22} {r.total_cases:<6} {r.passed_cases:<6} {r.failed_cases:<6} {r.safety_failures:<6}"
        )
    print("-" * 80)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evalhub",
        description="EvalHub: Standalone evaluation harness for saved agent runs using DeepEval."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Validate
    p_val = subparsers.add_parser("validate", help="Validate a JSONL dataset file")
    p_val.add_argument("--dataset", required=True, help="Path to JSONL dataset file")
    p_val.add_argument("--adapter", default="generic", help="Input adapter (default: generic)")
    p_val.add_argument("--input-only", action="store_true", help="Validate inputs for a live agent")

    # Run
    p_run = subparsers.add_parser("run", help="Run an evaluation against a dataset")
    p_run.add_argument("--dataset", required=True, help="Path to JSONL dataset file")
    p_run.add_argument("--config", required=True, help="Path to JSON configuration file")
    p_run.add_argument("--label", default=None, help="Optional human-readable run label")
    p_run.add_argument("--db", default="evalhub.db", help="Path to SQLite database file (default: evalhub.db)")

    # Compare
    p_cmp = subparsers.add_parser("compare", help="Compare a baseline run with a candidate run")
    p_cmp.add_argument("--baseline", required=True, help="Baseline run ID")
    p_cmp.add_argument("--candidate", required=True, help="Candidate run ID")
    p_cmp.add_argument("--db", default="evalhub.db", help="Path to SQLite database file (default: evalhub.db)")
    p_cmp.add_argument("--json-out", default=None, help="Optional path to export JSON comparison diff")

    # List
    p_list = subparsers.add_parser("list-runs", help="List all evaluation runs")
    p_list.add_argument("--db", default="evalhub.db", help="Path to SQLite database file (default: evalhub.db)")

    args = parser.parse_args(argv)

    if args.command == "validate":
        return cmd_validate(args)
    elif args.command == "run":
        return cmd_run(args)
    elif args.command == "compare":
        return cmd_compare(args)
    elif args.command == "list-runs":
        return cmd_list_runs(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
