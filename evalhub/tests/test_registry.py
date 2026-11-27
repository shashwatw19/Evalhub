"""Registered result retrieval without executing an agent."""

import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest

from evalhub.cli import main
from evalhub.connectors import build_get_url, create_agent
from evalhub.models import AgentRegistration, EvaluationConfig, HTTPAgentConfig
from evalhub.runner import EvaluationRunner
from evalhub.storage import CREATE_CASE_RESULTS_TABLE, CREATE_RUNS_TABLE, StorageEngine


@pytest.fixture
def results_api():
    state = {
        "payload": {
            "run_id": "run-1", "status": "completed", "input": "Investigate the alert",
            "recommendation": "ESCALATE", "explanation": "High risk",
            "tools": [{"name": "lookup", "result": "Risk: high"}],
            "context": ["Risk: high"], "evidence": [{"record_id": "claimed"}],
        },
        "status": 200,
        "requests": [],
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append({
                "path": self.path, "length": self.headers.get("Content-Length"),
                "authorization": self.headers.get("Authorization"),
            })
            self.send_response(state["status"])
            if state["status"] == 302:
                self.send_header("Location", "/results")
            self.end_headers()
            self.wfile.write(json.dumps(state["payload"]).encode())

        def do_POST(self):
            state["requests"].append({"method": "POST"})
            self.send_response(405)
            self.end_headers()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["url"] = f"http://127.0.0.1:{server.server_port}/results"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def retrieval_setup(tmp_path, results_api):
    registration = AgentRegistration(
        agent_id="risk", name="Risk", connector=HTTPAgentConfig(
            method="GET", url=results_api["url"], query_params={"run_id": "run_id"},
            response_path="recommendation", input_path="input", run_id_path="run_id",
            status_path="status", explanation_path="explanation", evidence_path="evidence",
            tool_calls_path="tools", retrieval_context_path="context",
        ),
    )
    storage = StorageEngine(str(tmp_path / "registry.db"))
    storage.save_agent(registration)
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(json.dumps({
        "case_id": "case-1", "lookup": {"run_id": "run-1"}, "expected_output": "ESCALATE",
        "evidence": [{"source": "trusted", "record_id": "authoritative"}],
    }) + "\n", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "agent_ref": {"agent_id": "risk"}, "metrics": [{"name": "exact_match"}],
    }), encoding="utf-8")
    return storage, registration, dataset, config


def test_get_encodes_query_without_body_and_reads_auth(results_api, monkeypatch):
    monkeypatch.setenv("RESULTS_TEST_TOKEN", "test-token")
    connector = create_agent(HTTPAgentConfig(
        method="GET", url=results_api["url"] + "?tenant=example",
        query_params={"run_id": "id"}, response_path="recommendation",
        auth_env="RESULTS_TEST_TOKEN",
    ))
    assert connector({"id": "run & +/?#"}).actual_output == "ESCALATE"
    request = results_api["requests"][0]
    assert parse_qs(urlsplit(request["path"]).query) == {
        "tenant": ["example"], "run_id": ["run & +/?#"],
    }
    assert request["length"] is None
    assert request["authorization"] == "Bearer test-token"


def test_get_path_mapping_cannot_change_url_structure(results_api):
    config = HTTPAgentConfig(
        method="GET", url=results_api["url"] + "/{id}", path_params={"id": "run_id"},
    )
    url = build_get_url(config, {"run_id": "run/1?next=evil#fragment"})
    assert url.endswith("/run%2F1%3Fnext%3Devil%23fragment")
    with pytest.raises(ValueError, match="dot segments"):
        build_get_url(config, {"run_id": ".."})
    with pytest.raises(ValueError, match="missing"):
        build_get_url(config, {})


def test_registry_retrieves_normalizes_and_preserves_ground_truth(retrieval_setup, results_api):
    storage, registration, dataset, config = retrieval_setup
    runner = EvaluationRunner(storage)
    assert runner.validate_dataset(str(dataset), lookup_only=True)[0].input is None
    summary, cases = runner.run_evaluation(str(dataset), str(config))
    assert summary.passed_cases == 1
    case = cases[0].case
    assert case.input == "Investigate the alert"
    assert case.lookup == {"run_id": "run-1"}
    assert case.source_run_id == "run-1"
    assert case.tool_calls[0].name == "lookup"
    assert case.retrieval_context == ["Risk: high"]
    assert case.evidence[0].record_id == "authoritative"
    assert case.retrieval["raw_response"] == results_api["payload"]
    assert case.retrieval["explanation"] == "High risk"
    assert storage.get_case_results(summary.run_id)[0]["case"]["retrieval"] == case.retrieval

    storage.save_agent(registration.model_copy(update={"name": "Updated risk"}), update=True)
    stored = storage.get_run(summary.run_id)
    snapshot = json.loads(stored.config_json)["registry_snapshot"]
    assert snapshot["revision"] == 1
    assert snapshot["name"] == "Risk"
    assert storage.get_agent("risk", 1).name == "Risk"


def test_retrieval_supports_recommendation_only(retrieval_setup, results_api):
    storage, registration, dataset, config = retrieval_setup
    results_api["payload"] = {"recommendation": "ESCALATE"}
    minimal = HTTPAgentConfig(
        method="GET", url=results_api["url"], query_params={"run_id": "run_id"},
        response_path="recommendation",
    )
    storage.save_agent(registration.model_copy(update={"connector": minimal}), update=True)
    summary, cases = EvaluationRunner(storage).run_evaluation(str(dataset), str(config))
    assert summary.passed_cases == 1
    assert cases[0].case.input is None
    assert cases[0].case.source_run_id == "run-1"


@pytest.mark.parametrize("source_status", ["pending", "failed", "cancelled", "unknown"])
def test_uncompleted_source_is_not_scored(retrieval_setup, results_api, monkeypatch, source_status):
    storage, _, dataset, config = retrieval_setup
    results_api["payload"] = {"status": source_status}

    def unexpected_evaluation(*_args):
        pytest.fail("Incomplete source runs must not reach DeepEval.")

    monkeypatch.setattr("evalhub.runner._run_deepeval", unexpected_evaluation)
    summary, cases = EvaluationRunner(storage).run_evaluation(str(dataset), str(config))
    assert summary.error_cases == 1
    assert summary.passed_cases == 0
    assert cases[0].case.retrieval["status"] == "source_not_completed"
    assert cases[0].case.retrieval["source_status"] == source_status
    assert cases[0].case.actual_output is None


@pytest.mark.parametrize("failure", ["missing", "ambiguous", "mismatched-id", "404", "redirect"])
def test_retrieval_errors_are_persisted(retrieval_setup, results_api, failure):
    storage, _, dataset, config = retrieval_setup
    if failure == "missing":
        del results_api["payload"]["recommendation"]
    elif failure == "ambiguous":
        results_api["payload"] = [results_api["payload"], results_api["payload"]]
    elif failure == "mismatched-id":
        results_api["payload"]["run_id"] = "different-run"
    elif failure == "404":
        results_api["status"] = 404
    else:
        results_api["status"] = 302
    summary, cases = EvaluationRunner(storage).run_evaluation(str(dataset), str(config))
    assert summary.error_cases == 1
    assert cases[0].case.retrieval["status"] == "error"
    assert cases[0].case.execution_error
    assert len(results_api["requests"]) == 1
    assert storage.get_case_results(summary.run_id)[0]["case"]["retrieval"]["error"]


def test_invalid_lookup_and_disabled_agent_do_not_make_requests(retrieval_setup, results_api):
    storage, _, dataset, config = retrieval_setup
    dataset.write_text('{"case_id":"one","lookup":{"run_id":true}}\n', encoding="utf-8")
    runner = EvaluationRunner(storage)
    with pytest.raises(ValueError, match="run_id"):
        runner.run_evaluation(str(dataset), str(config))
    storage.disable_agent("risk")
    config.write_text('{"agent_ref":{"agent_id":"risk","revision":1}}', encoding="utf-8")
    with pytest.raises(ValueError, match="disabled"):
        runner.run_evaluation(str(dataset), str(config))
    assert not results_api["requests"]


def test_comparison_uses_original_task_not_source_lookup(retrieval_setup, results_api):
    storage, _, dataset, config = retrieval_setup
    runner = EvaluationRunner(storage)
    baseline, _ = runner.run_evaluation(str(dataset), str(config))
    record = json.loads(dataset.read_text())
    record["lookup"]["run_id"] = "run-2"
    dataset.write_text(json.dumps(record), encoding="utf-8")
    results_api["payload"]["run_id"] = "run-2"
    results_api["payload"]["recommendation"] = "CLOSE"
    candidate, _ = runner.run_evaluation(str(dataset), str(config))
    assert storage.compare_runs(baseline.run_id, candidate.run_id).status == "regressed"
    results_api["payload"]["input"] = "A different task"
    different, _ = runner.run_evaluation(str(dataset), str(config))
    assert storage.compare_runs(baseline.run_id, different.run_id).status == "incompatible"


def test_registry_cli_lifecycle_and_explicit_retrieval(retrieval_setup, results_api, tmp_path, capsys):
    storage, registration, dataset, config = retrieval_setup
    file = tmp_path / "agent.json"
    file.write_text(registration.model_dump_json(), encoding="utf-8")
    db = str(tmp_path / "cli.db")
    assert main(["agents", "register", "--file", str(file), "--db", db]) == 0
    assert main(["agents", "register", "--file", str(file), "--db", db]) == 2
    assert not results_api["requests"]
    assert main(["agents", "update", "--file", str(file), "--db", db]) == 0
    assert main(["agents", "list", "--db", db]) == 0
    assert main(["agents", "show", "--agent-id", "risk", "--revision", "1", "--db", db]) == 0
    assert main(["agents", "test", "--agent-id", "risk", "--lookup", '{"run_id":"run-1"}', "--db", db]) == 0
    assert main(["validate", "--dataset", str(dataset), "--lookup-only"]) == 0
    assert main(["run", "--dataset", str(dataset), "--config", str(config), "--db", db]) == 0
    assert "ESCALATE" in capsys.readouterr().out
    assert main(["agents", "disable", "--agent-id", "risk", "--db", db]) == 0
    before = len(results_api["requests"])
    assert main(["agents", "test", "--agent-id", "risk", "--lookup", '{"run_id":"run-1"}', "--db", db]) == 2
    assert len(results_api["requests"]) == before


def test_registry_migrates_existing_database_without_losing_runs(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(CREATE_RUNS_TABLE + CREATE_CASE_RESULTS_TABLE)
        conn.execute("PRAGMA user_version = 1")
        conn.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
            "legacy", "original", "2026-01-01", "completed", "cases.jsonl", "hash", "fingerprint",
            "{}", "config-hash", "4.3.4", "{}", 1, 1, 0, 0, 0, 0,
        ))
        conn.execute("""INSERT INTO case_results
            (run_id, case_id, normalized_case_json, execution_status, overall_status, metric_results_json)
            VALUES (?, ?, ?, ?, ?, ?)""", (
                "legacy", "case-1", '{"input":"q","actual_output":"a"}', "completed", "passed", "[]",
        ))
    migrated = StorageEngine(str(path))
    assert migrated.get_schema_version() == 3
    assert migrated.get_run("legacy").label == "original"
    assert migrated.get_case_results("legacy")[0]["case"]["actual_output"] == "a"
    assert migrated.list_agents() == []


def test_config_rejects_ambiguous_source_and_invalid_mapping():
    with pytest.raises(ValueError, match="either agent or agent_ref"):
        EvaluationConfig(
            agent_ref={"agent_id": "risk"},
            agent=HTTPAgentConfig(url="http://localhost/invoke"),
        )
    with pytest.raises(ValueError, match="must use GET"):
        AgentRegistration(
            agent_id="risk", name="Risk", connector=HTTPAgentConfig(url="http://localhost"),
        )
    with pytest.raises(ValueError, match="lookup_fields"):
        AgentRegistration(
            agent_id="risk", name="Risk",
            connector=HTTPAgentConfig(method="GET", url="http://localhost", query_params={"id": "wrong"}),
        )
    with pytest.raises(ValueError, match="placeholders"):
        HTTPAgentConfig(method="GET", url="http://localhost/runs/{id}")
