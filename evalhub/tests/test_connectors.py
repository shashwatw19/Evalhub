"""Integration tests for live agent connectors."""

import json
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from evalhub.connectors import AgentExecutionError, create_agent
from evalhub.models import HTTPAgentConfig
from evalhub.runner import EvaluationRunner
from evalhub.storage import StorageEngine


def test_python_agent_runs_input_dataset_and_comparisons(tmp_path, monkeypatch):
    module = types.ModuleType("test_evalhub_agent")
    module.respond = lambda value: {
        "answer": value.upper(),
        "calls": [{"name": "search"}],
    }
    monkeypatch.setitem(sys.modules, module.__name__, module)
    dataset = tmp_path / "inputs.jsonl"
    dataset.write_text(
        '{"case_id":"one","input":"hello","expected_output":"HELLO",'
        '"expected_required_tools":["search"]}\n',
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "agent": {
            "type": "python",
            "callable": "test_evalhub_agent:respond",
            "response_path": "answer",
            "tool_calls_path": "calls",
        },
        "metrics": [{"name": "exact_match"}, {"name": "tool_coverage"}],
    }), encoding="utf-8")
    storage = StorageEngine(str(tmp_path / "test.db"))
    runner = EvaluationRunner(storage)
    assert len(runner.validate_dataset(str(dataset), input_only=True)) == 1
    baseline, cases = runner.run_evaluation(str(dataset), str(config))
    assert baseline.passed_cases == 1
    assert cases[0].case.actual_output == "HELLO"
    assert cases[0].case.tool_calls[0].name == "search"

    module.respond = lambda value: {"answer": "wrong", "calls": []}
    candidate, _ = runner.run_evaluation(str(dataset), str(config))
    assert candidate.failed_cases == 1
    assert storage.compare_runs(baseline.run_id, candidate.run_id).status == "regressed"


def test_python_agent_failure_is_persisted_without_exception_details(tmp_path, monkeypatch):
    module = types.ModuleType("failing_evalhub_agent")

    def fail(value):
        raise RuntimeError("sensitive exception details")

    module.respond = fail
    monkeypatch.setitem(sys.modules, module.__name__, module)
    dataset = tmp_path / "inputs.jsonl"
    dataset.write_text('{"case_id":"one","input":"hello"}\n', encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "agent": {"type": "python", "callable": "failing_evalhub_agent:respond"},
        "metrics": [{"name": "exact_match"}],
    }), encoding="utf-8")
    storage = StorageEngine(str(tmp_path / "test.db"))
    summary, cases = EvaluationRunner(storage).run_evaluation(str(dataset), str(config))
    assert summary.error_cases == 1
    assert cases[0].case.execution_error == "Python agent failed (RuntimeError)."
    assert storage.get_case_results(summary.run_id)[0]["execution_status"] == "failed"


def test_http_agent_posts_input_maps_response_and_reads_env_auth(monkeypatch, tmp_path):
    received = []
    retry_count = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((body, self.headers.get("Authorization")))
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/invoke")
                self.end_headers()
                return
            if self.path == "/retry" and not retry_count:
                retry_count.append(1)
                self.send_response(503)
                self.end_headers()
                return
            payload = json.dumps({
                "result": {"answer": "world", "tools": [{"name": "search"}]},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("TEST_AGENT_TOKEN", "test-token")
    try:
        agent = create_agent(HTTPAgentConfig(
            url=f"http://127.0.0.1:{server.server_port}/invoke",
            input_field="prompt",
            response_path="result.answer",
            tool_calls_path="result.tools",
            auth_env="TEST_AGENT_TOKEN",
        ))
        result = agent("hello")
        assert result.actual_output == "world"
        assert result.tool_calls[0].name == "search"
        assert received == [({"prompt": "hello"}, "Bearer test-token")]
        redirect_agent = create_agent(HTTPAgentConfig(
            url=f"http://127.0.0.1:{server.server_port}/redirect",
            auth_env="TEST_AGENT_TOKEN",
        ))
        with pytest.raises(AgentExecutionError, match="status 302"):
            redirect_agent("hello")
        assert len(received) == 2
        monkeypatch.setattr("evalhub.connectors.time.sleep", lambda _: None)
        retry_agent = create_agent(HTTPAgentConfig(
            url=f"http://127.0.0.1:{server.server_port}/retry",
            response_path="result.answer",
            retries=1,
        ))
        assert retry_agent("hello").actual_output == "world"
        assert len(received) == 4
        dataset = tmp_path / "cases.jsonl"
        dataset.write_text(
            '{"case_id":"one","input":"hello","expected_output":"world"}\n',
            encoding="utf-8",
        )
        config = tmp_path / "config.json"
        config.write_text(json.dumps({
            "agent": {
                "type": "http",
                "url": f"http://127.0.0.1:{server.server_port}/invoke",
                "response_path": "result.answer",
                "auth_env": "TEST_AGENT_TOKEN",
            },
            "metrics": [{"name": "exact_match"}],
        }), encoding="utf-8")
        summary, _ = EvaluationRunner(StorageEngine(str(tmp_path / "test.db"))).run_evaluation(
            str(dataset), str(config)
        )
        assert summary.passed_cases == 1
        assert "test-token" not in summary.config_json
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_connector_rejects_missing_auth_and_response_fields(monkeypatch):
    monkeypatch.delenv("MISSING_AGENT_TOKEN", raising=False)
    with pytest.raises(ValueError, match="MISSING_AGENT_TOKEN"):
        create_agent(HTTPAgentConfig(url="http://localhost", auth_env="MISSING_AGENT_TOKEN"))
    module = types.ModuleType("invalid_evalhub_agent")
    module.respond = lambda value: {"wrong": value}
    monkeypatch.setitem(sys.modules, module.__name__, module)
    from evalhub.models import PythonAgentConfig
    agent = create_agent(PythonAgentConfig(
        callable="invalid_evalhub_agent:respond", response_path="answer"
    ))
    with pytest.raises(AgentExecutionError, match="missing field path 'answer'"):
        agent("hello")


def test_agent_response_must_be_json_serializable(monkeypatch):
    module = types.ModuleType("non_json_evalhub_agent")
    module.respond = lambda value: object()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    from evalhub.models import PythonAgentConfig
    agent = create_agent(PythonAgentConfig(callable="non_json_evalhub_agent:respond"))
    with pytest.raises(AgentExecutionError, match="non-JSON output"):
        agent("hello")


def test_http_timeout_is_recorded_as_a_case_error(tmp_path, monkeypatch):
    class TimeoutOpener:
        def open(self, request, timeout):
            raise TimeoutError("untrusted endpoint details")

    monkeypatch.setattr("evalhub.connectors.build_opener", lambda *args: TimeoutOpener())
    dataset = tmp_path / "inputs.jsonl"
    dataset.write_text('{"case_id":"one","input":"hello"}\n', encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "agent": {"type": "http", "url": "https://agent.example.test/invoke"},
        "metrics": [{"name": "exact_match"}],
    }), encoding="utf-8")
    summary, cases = EvaluationRunner(StorageEngine(str(tmp_path / "test.db"))).run_evaluation(
        str(dataset), str(config)
    )
    assert summary.error_cases == 1
    assert cases[0].case.execution_status.value == "timed_out"
    assert cases[0].case.execution_error == "Agent HTTP request timed out."
