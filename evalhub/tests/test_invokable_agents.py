import json

from evalhub.connectors import AgentResponse
from evalhub.models import (
    AgentTestCase,
    InvokableAgentRegistration,
    MetricConfig,
    MetricResult,
    MetricStatus,
    PythonAgentConfig,
    ToolCallRecord,
)
from evalhub.runner import EvaluationRunner
from evalhub.storage import StorageEngine


def test_registered_profile_runs_its_saved_suite_and_snapshots_context(
    tmp_path, monkeypatch
):
    storage = StorageEngine(str(tmp_path / "profiles.db"))
    profile = InvokableAgentRegistration(
        agent_id="alert-investigator",
        name="Alert investigator",
        connector=PythonAgentConfig(callable="trusted_agent:investigate"),
        test_cases=[
            AgentTestCase(
                case_id="alert-1",
                input={"alert_id": "A-1"},
            )
        ],
        reference_dossiers=["reference investigation"],
        metrics=[MetricConfig(name="investigation_correctness")],
        judge_model="gemini-test",
    )
    saved_profile = storage.save_invokable_agent(profile)
    config_path = tmp_path / "profile-run.json"
    config_path.write_text(
        json.dumps({
            "agent_profile_ref": {
                "agent_id": saved_profile.agent_id,
                "revision": saved_profile.revision,
            }
        }),
        encoding="utf-8",
    )

    def fake_agent_factory(connector):
        assert connector.callable == "trusted_agent:investigate"

        def invoke(input_value):
            assert input_value == {"alert_id": "A-1"}
            return AgentResponse(
                actual_output="Escalate based on high severity.",
                tool_calls=[
                    ToolCallRecord(
                        name="alert_lookup",
                        arguments={"alert_id": "A-1"},
                        result={"severity": "high"},
                    )
                ],
            )

        return invoke

    def fake_metric_factory(*_args, **_kwargs):
        return object()

    def fake_evaluation(cases, metrics):
        assert len(metrics) == 1
        case = cases[0]
        assert case.metadata["judge_reference_dossiers"] == ["reference investigation"]
        assert case.tool_calls[0].result == {"severity": "high"}
        return {
            case.case_id: [
                MetricResult(
                    name="investigation_correctness",
                    status=MetricStatus.PASSED,
                    score=0.9,
                    explanation="The conclusion matches the tool result.",
                )
            ]
        }

    monkeypatch.setattr("evalhub.runner.create_agent", fake_agent_factory)
    monkeypatch.setattr("evalhub.runner.get_metric_instance", fake_metric_factory)
    monkeypatch.setattr("evalhub.runner._run_deepeval", fake_evaluation)

    summary, results = EvaluationRunner(storage).run_evaluation(
        None, str(config_path), label="registered-profile"
    )

    assert summary.passed_cases == 1
    assert summary.dataset_path == "registered-agent:alert-investigator@1"
    assert results[0].case.actual_output == "Escalate based on high severity."
    assert results[0].case.execution_status.value == "completed"
    saved_config = json.loads(storage.get_run(summary.run_id).config_json)
    assert saved_config["agent_profile_snapshot"]["revision"] == 1
    assert saved_config["agent_profile_snapshot"]["reference_dossiers"] == [
        "reference investigation"
    ]
