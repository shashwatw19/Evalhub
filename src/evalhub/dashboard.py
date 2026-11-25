"""
dashboard.py

Purpose:
    Standalone Streamlit inspection dashboard for EvalHub evaluation results.

Responsibility:
    - View 1 (Run List): Display historical runs, dates, labels, and pass/fail/error/safety metrics.
    - View 2 (Run Detail): Inspect individual cases, inputs, actual outputs, expected labels,
      tool calls, and metric diagnostic explanations.

Dependencies:
    - streamlit
    - json
    - evalhub.storage (StorageEngine)
    - evalhub.models (MetricStatus)

Key Design Decisions:
    - Local dashboard for running evaluations and inspecting persisted results.
    - Prominently isolates critical safety failures.
"""

from __future__ import annotations
import os
import json
import hashlib
from pathlib import Path
import streamlit as st
from pydantic import ValidationError
from evalhub.models import AgentRegistration, EvaluationConfig, MetricStatus
from evalhub.runner import EvaluationRunner
from evalhub.storage import StorageEngine
from evalhub.connectors import AgentExecutionError, retrieve_registered_run

st.set_page_config(
    page_title="EvalHub Dashboard",
    page_icon="🔍",
    layout="wide"
)

# Database path resolution
db_path = os.environ.get("EVALHUB_DB", "evalhub.db")
storage = StorageEngine(db_path=db_path)


def render_sidebar():
    st.sidebar.title("🔍 EvalHub")
    st.sidebar.caption("Standalone Agent Evaluation Harness")
    st.sidebar.info(f"Connected DB: `{db_path}`")
    st.sidebar.markdown("---")


def render_run_list(runs):
    st.header("Evaluation Runs")
    if not runs:
        st.warning("No evaluation runs found in the database. Run an evaluation via the CLI first.")
        st.code("evalhub run --dataset examples/generic/cases.jsonl --config examples/generic/config.json --label baseline")
        return None

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    total_runs = len(runs)
    total_evals = sum(r.total_cases for r in runs)
    total_safety_breaches = sum(r.safety_failures for r in runs)
    col1.metric("Total Runs", total_runs)
    col2.metric("Total Cases Evaluated", total_evals)
    col3.metric("Safety Breaches", total_safety_breaches, delta_color="inverse")

    # Table formatting
    table_data = []
    for r in runs:
        table_data.append({
            "Run ID": r.run_id,
            "Label": r.label or "-",
            "Created At": r.created_at[:19].replace("T", " "),
            "Total": r.total_cases,
            "Passed": r.passed_cases,
            "Failed": r.failed_cases,
            "Errors": r.error_cases,
            "Safety Failures": r.safety_failures,
            "Status": r.run_status
        })

    st.dataframe(table_data, use_container_width=True)

    st.markdown("---")
    st.subheader("Select Run to Inspect Details")
    run_options = {f"{r.run_id} ({r.label or 'unlabeled'} - {r.created_at[:10]})": r.run_id for r in runs}
    selected_label = st.selectbox("Choose Run", list(run_options.keys()))
    return run_options[selected_label]


def render_run_detail(run_id: str):
    run_summary = storage.get_run(run_id)
    if not run_summary:
        st.error(f"Run '{run_id}' not found.")
        return

    st.subheader(f"Run Details: {run_id}")
    if run_summary.label:
        st.caption(f"Label: **{run_summary.label}** | Evaluated: `{run_summary.created_at}`")

    # Metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Cases", run_summary.total_cases)
    m2.metric("Passed", run_summary.passed_cases)
    m3.metric("Failed", run_summary.failed_cases)
    m4.metric("Errors", run_summary.error_cases)
    m5.metric("Safety Failures", run_summary.safety_failures, delta_color="inverse")

    if run_summary.safety_failures > 0:
        st.error(f"⚠️ Critical Safety Alert: This run contains {run_summary.safety_failures} safety failure(s)!")

    # Provenance expander
    with st.expander("Provenance & Hashes"):
        st.json({
            "dataset_path": run_summary.dataset_path,
            "dataset_hash": run_summary.dataset_hash,
            "comparison_fingerprint": run_summary.comparison_fingerprint,
            "deepeval_version": run_summary.deepeval_version,
            "config_hash": run_summary.config_hash
        })

    # Cases breakdown
    st.markdown("### Case Results")
    case_results = storage.get_case_results(run_id)

    # Filter
    status_filter = st.radio(
        "Filter by status:",
        ["All", "Failed / Error", "Passed"],
        horizontal=True
    )

    filtered_cases = []
    for cr in case_results:
        st_val = cr["overall_status"]
        if status_filter == "Failed / Error" and st_val not in (MetricStatus.FAILED.value, MetricStatus.ERROR.value):
            continue
        if status_filter == "Passed" and st_val != MetricStatus.PASSED.value:
            continue
        filtered_cases.append(cr)

    st.caption(f"Showing {len(filtered_cases)} of {len(case_results)} cases")

    for item in filtered_cases:
        case = item["case"]
        case_id = item["case_id"]
        status = item["overall_status"]
        metrics = item["metric_results"]

        status_icon = "✅" if status == "passed" else ("🚨" if any(m.get("is_safety_failure") for m in metrics) else "❌")

        with st.expander(f"{status_icon} Case: {case_id} [{status.upper()}]"):
            c1, c2 = st.columns([1, 1])

            with c1:
                st.markdown("**Agent Input:**")
                st.json(case.get("input"))
                if case.get("lookup") is not None:
                    st.markdown("**Lookup parameters:**")
                    st.json(case["lookup"])
                if case.get("source_run_id"):
                    st.write("Source run ID:", case["source_run_id"])

                st.markdown("**Expected Labels:**")
                st.json(case.get("expected_labels") or "None")
                if case.get("expected_output") is not None:
                    st.markdown("**Expected Output:**")
                    st.json(case["expected_output"])

                if case.get("expected_required_tools"):
                    st.markdown("**Expected Required Tools:**")
                    st.write(case.get("expected_required_tools"))

            with c2:
                st.markdown("**Actual Output:**")
                output = case.get("actual_output")
                st.json(output if output is not None else "None (Failed Execution)")
                if case.get("execution_error"):
                    st.error(case["execution_error"])

                if case.get("tool_calls"):
                    st.markdown("**Observed Tool Calls:**")
                    st.json([{"name": tc["name"], "args": tc.get("arguments")} for tc in case["tool_calls"]])
                if case.get("retrieval_context"):
                    st.markdown("**Retrieval Context:**")
                    st.json(case["retrieval_context"])
                if case.get("retrieval"):
                    st.markdown("**Retrieval snapshot:**")
                    st.json(case["retrieval"])

            st.markdown("---")
            st.markdown("**Metric Diagnoses:**")
            for m in metrics:
                badge = "🟢 PASS" if m["status"] == "passed" else ("🔴 FAIL" if m["status"] == "failed" else "🟡 " + m["status"].upper())
                safety_alert = " **[SAFETY FAILURE]**" if m.get("is_safety_failure") else ""
                score_str = f" | Score: `{m.get('score')}`" if m.get("score") is not None else ""
                st.write(f"- {badge} **{m['name']}**{score_str}{safety_alert}: {m['explanation']}")


def render_new_run():
    st.header("Run an evaluation")
    st.warning(
        "Use this dashboard only on a trusted local machine. Python connectors execute "
        "local code; HTTP connectors and Gemini metrics send case data to their configured services."
    )
    default_config = {
        "agent": {
            "type": "http",
            "url": "http://127.0.0.1:8000/invoke",
            "response_path": "output",
        },
        "metrics": [{"name": "exact_match", "threshold": 1.0, "required": True}],
    }
    config_mode = st.radio("Configuration", ["Guided", "Advanced JSON"], horizontal=True)
    if config_mode == "Guided":
        agent_mode = st.selectbox(
            "Agent connection",
            ["Saved outputs", "Registered agent results", "HTTP agent", "Python callable"],
        )
    else:
        agent_mode = None
    with st.form("new_evaluation"):
        dataset = st.file_uploader("JSONL dataset", type=["jsonl"])
        if config_mode == "Advanced JSON":
            config_text = st.text_area(
                "Evaluation configuration (JSON)",
                value=json.dumps(default_config, indent=2),
                height=300,
                help="Use environment variables for credentials; never put secrets in this configuration.",
            )
        else:
            config_text = ""
            selected_registration = None
            if agent_mode == "HTTP agent":
                url = st.text_input("Agent endpoint URL", value="http://127.0.0.1:8000/invoke")
                input_field = st.text_input("Input field", value="input")
                auth_env = st.text_input("Authentication token environment variable (optional)")
                timeout = st.number_input("HTTP timeout (seconds)", min_value=1, max_value=300, value=30)
            elif agent_mode == "Python callable":
                callable_path = st.text_input("Python callable (module:function)")
            elif agent_mode == "Registered agent results":
                agents = {agent.agent_id: agent for agent in storage.list_agents() if agent.enabled}
                if agents:
                    agent_id = st.selectbox("Registered agent", list(agents))
                    selected_registration = agents[agent_id]
                    st.caption(f"Registry revision {selected_registration.revision}. "
                               "Uploads must contain lookup objects and expected results.")
                else:
                    st.info("Register an agent in the Agents tab first.")
            if agent_mode in ("HTTP agent", "Python callable"):
                response_path = st.text_input("Response field path (optional)")
                tool_path = st.text_input("Tool calls field path (optional)")
                context_path = st.text_input("Retrieval context field path (optional)")
            metric_names = st.multiselect(
                "Metrics",
                [
                    "exact_match", "tool_coverage", "evidence_reference_validity",
                    "explanation_quality", "answer_relevancy", "answer_correctness", "faithfulness",
                ],
                default=["exact_match"],
            )
            judge_model = st.text_input("Gemini model (required for LLM metrics)")
        label = st.text_input("Run label")
        consent = st.checkbox("I trust the connector and approve configured external data transfers.")
        preview = (
            st.form_submit_button("Preview first case")
            if agent_mode == "Registered agent results" else False
        )
        submitted = st.form_submit_button("Run evaluation")
    if not submitted and not preview:
        return
    if dataset is None:
        st.error("Upload a JSONL dataset.")
        return
    if not consent:
        st.error("Confirm that you trust the connector and approve configured data transfers.")
        return
    try:
        if config_mode == "Guided":
            if not metric_names:
                raise ValueError("Select at least one metric.")
            if agent_mode == "HTTP agent":
                agent = {
                    "type": "http",
                    "url": url,
                    "input_field": input_field,
                    "auth_env": auth_env or None,
                    "timeout_seconds": timeout,
                }
            elif agent_mode == "Python callable":
                agent = {"type": "python", "callable": callable_path}
            else:
                agent = None
            if agent is not None:
                agent.update({
                    "response_path": response_path or None,
                    "tool_calls_path": tool_path or None,
                    "retrieval_context_path": context_path or None,
                })
            llm_metrics = {"answer_relevancy", "answer_correctness", "faithfulness"}
            if llm_metrics.intersection(metric_names) and not judge_model.strip():
                raise ValueError("Enter a Gemini model for the selected LLM metrics.")
            config_data = {
                "agent": agent,
                "judge_enabled": bool(llm_metrics.intersection(metric_names) or "explanation_quality" in metric_names),
                "judge_model": judge_model.strip() or None,
                "metrics": [
                    {
                        "name": name,
                        "threshold": 0.7 if name in llm_metrics or name == "explanation_quality" else 1.0,
                    }
                    for name in metric_names
                ],
            }
            if agent_mode == "Registered agent results":
                if selected_registration is None:
                    raise ValueError("Register and select an enabled agent first.")
                config_data["agent_ref"] = {
                    "agent_id": selected_registration.agent_id,
                    "revision": selected_registration.revision,
                }
            config_text = json.dumps(config_data)
        config = EvaluationConfig.model_validate_json(config_text)
        dataset_bytes = dataset.getvalue()
        if preview:
            records = [
                json.loads(line) for line in dataset_bytes.decode("utf-8").splitlines()
                if line.strip()
            ]
            if not records or not isinstance(records[0], dict) or not isinstance(records[0].get("lookup"), dict):
                raise ValueError("The first dataset record must contain a lookup object.")
            if config.agent_ref is None:
                raise ValueError("Preview requires a registered agent.")
            registration = storage.get_active_agent(
                config.agent_ref.agent_id, config.agent_ref.revision
            )
            response = retrieve_registered_run(registration, records[0]["lookup"])
            st.json(response.model_dump(mode="json"))
            st.info("Preview only; no evaluation performed. Running an evaluation retrieves data again.")
            return
        data_dir = Path(".evalhub")
        data_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = data_dir / (hashlib.sha256(dataset_bytes).hexdigest() + ".jsonl")
        config_bytes = config.model_dump_json(indent=2).encode("utf-8")
        config_path = data_dir / (hashlib.sha256(config_bytes).hexdigest() + ".json")
        dataset_path.write_bytes(dataset_bytes)
        config_path.write_bytes(config_bytes)
        with st.spinner("Running evaluation..."):
            summary, _ = EvaluationRunner(storage).run_evaluation(
                str(dataset_path), str(config_path), label=label or None
            )
        st.success(f"Run {summary.run_id}: {summary.passed_cases} passed, "
                   f"{summary.failed_cases} failed, {summary.error_cases} errors.")
    except (ValidationError, ValueError, OSError, RuntimeError, UnicodeError) as exc:
        st.error(f"Evaluation failed: {exc}")


def render_agents():
    st.header("Agents")
    st.caption("Register read-only GET APIs returning existing agent runs. Registration makes no API calls.")
    agents = storage.list_agents()
    if agents:
        st.dataframe([
            {"Agent ID": agent.agent_id, "Name": agent.name, "Revision": agent.revision,
             "Enabled": agent.enabled, "URL": agent.connector.url}
            for agent in agents
        ], use_container_width=True)
    selected_id = st.selectbox("Registration to edit", ["New agent"] + [a.agent_id for a in agents])
    current = storage.get_agent(selected_id) if selected_id != "New agent" else None
    template = (
        current.model_dump(mode="json", exclude={"revision", "created_at"})
        if current else {
            "agent_id": "card-risk",
            "name": "Card risk",
            "enabled": True,
            "lookup_fields": {"run_id": "string"},
            "connector": {
                "type": "http", "method": "GET",
                "url": "http://127.0.0.1:8000/api/evalhub/investigate",
                "query_params": {"run_id": "run_id"},
                "response_path": "recommendation",
                "input_path": None, "run_id_path": None, "status_path": None,
                "explanation_path": None, "evidence_path": None,
                "tool_calls_path": None, "retrieval_context_path": None,
                "auth_env": None, "timeout_seconds": 300, "retries": 0,
            },
        }
    )
    with st.form("agent_registration"):
        registration_text = st.text_area(
            "Agent registration (JSON)", value=json.dumps(template, indent=2),
            height=420, key=f"registration_{selected_id}_{current.revision if current else 0}",
            help="Response mappings are optional. Set only paths the API actually returns. "
                 "Use auth_env for credentials; never store tokens in URLs or JSON.",
        )
        save = st.form_submit_button("Save registration")
        disable = st.form_submit_button("Disable agent") if current else False
    try:
        if save:
            registration = AgentRegistration.model_validate_json(registration_text)
            if current and registration.agent_id != current.agent_id:
                raise ValueError("Agent ID cannot change when updating a registration.")
            saved = storage.save_agent(registration, update=current is not None)
            st.success(f"Saved {saved.agent_id}, revision {saved.revision}.")
            st.rerun()
        if disable and current:
            storage.disable_agent(current.agent_id)
            st.rerun()
    except (ValueError, OSError) as exc:
        st.error(f"Registration failed: {exc}")
    st.subheader("Test retrieval")
    active = [agent.agent_id for agent in agents if agent.enabled]
    if not active:
        st.info("Save an enabled registration to test retrieval.")
        return
    with st.form("test_retrieval"):
        test_id = st.selectbox("Agent to test", active)
        lookup_text = st.text_area("Lookup (JSON)", value='{"run_id": "replace-with-run-id"}')
        consent = st.checkbox("I approve fetching this record from the configured API.")
        test = st.form_submit_button("Test retrieval")
    if test:
        try:
            if not consent:
                raise ValueError("Approve fetching the record first.")
            lookup = json.loads(lookup_text)
            if not isinstance(lookup, dict):
                raise ValueError("Lookup must be a JSON object.")
            response = retrieve_registered_run(storage.get_active_agent(test_id), lookup)
            st.json(response.model_dump(mode="json"))
            if not response.ready:
                st.warning("Source run is not completed. No evaluation performed.")
        except (ValueError, OSError, AgentExecutionError) as exc:
            st.error(f"Retrieval failed: {exc}")


def render_comparison(runs):
    st.header("Compare runs")
    if len(runs) < 2:
        st.info("Run at least two evaluations to compare them.")
        return
    ids = [run.run_id for run in runs]
    baseline = st.selectbox("Baseline", ids, index=1)
    candidate = st.selectbox("Candidate", ids, index=0)
    if st.button("Compare"):
        try:
            diff = storage.compare_runs(baseline, candidate)
            if not diff.fingerprint_match:
                st.error(diff.message)
            else:
                st.json(diff.model_dump(mode="json"))
        except ValueError as exc:
            st.error(str(exc))


def main():
    render_sidebar()
    new_tab, agents_tab, runs_tab, compare_tab = st.tabs(["New evaluation", "Agents", "Runs", "Compare"])
    with new_tab:
        render_new_run()
    with agents_tab:
        render_agents()
    runs = storage.list_runs()
    with runs_tab:
        selected_run_id = render_run_list(runs)
        if selected_run_id:
            render_run_detail(selected_run_id)
    with compare_tab:
        render_comparison(runs)


if __name__ == "__main__":
    main()
