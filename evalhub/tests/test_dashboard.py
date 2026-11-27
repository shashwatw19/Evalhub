from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest
from evalhub.models import (
    AgentTestCase,
    InvokableAgentRegistration,
    MetricConfig,
    PythonAgentConfig,
)
from evalhub.storage import StorageEngine


def test_dashboard_renders_without_running_an_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALHUB_DB", str(tmp_path / "dashboard.db"))
    dashboard = Path(__file__).resolve().parents[1] / "src" / "evalhub" / "dashboard.py"
    app = AppTest.from_file(str(dashboard), default_timeout=15).run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["New evaluation", "Agents", "Runs", "Compare"]
    app.selectbox[0].select("Python callable").run()
    assert any("Python callable" in field.label for field in app.text_input)
    app.button[0].click().run()
    assert any("Upload a JSONL dataset" in error.value for error in app.error)


def test_dashboard_registers_and_disables_without_api_calls(tmp_path, monkeypatch):
    db = str(tmp_path / "dashboard-registry.db")
    monkeypatch.setenv("EVALHUB_DB", db)

    def unexpected_request(*_args):
        raise AssertionError("Registration must not fetch data.")

    monkeypatch.setattr("evalhub.connectors.create_agent", unexpected_request)
    dashboard = Path(__file__).resolve().parents[1] / "src" / "evalhub" / "dashboard.py"
    app = AppTest.from_file(str(dashboard), default_timeout=15).run()
    next(button for button in app.button if button.label == "Save registration").click().run()
    assert not app.exception
    assert StorageEngine(db).get_agent("card-risk").revision == 1
    app.selectbox[0].select("Registered agent results").run()
    assert any(box.label == "Registered agent" for box in app.selectbox)
    assert any(button.label == "Preview first case" for button in app.button)
    next(box for box in app.selectbox if box.label == "Registration to edit").select("card-risk").run()
    next(button for button in app.button if button.label == "Disable agent").click().run()
    assert not app.exception
    assert not StorageEngine(db).get_agent("card-risk").enabled


def test_new_evaluation_runs_saved_profile_without_dataset(tmp_path, monkeypatch):
    db = str(tmp_path / "dashboard-profile.db")
    monkeypatch.setenv("EVALHUB_DB", db)
    monkeypatch.chdir(tmp_path)
    StorageEngine(db).save_invokable_agent(
        InvokableAgentRegistration(
            agent_id="alert-investigator",
            name="Alert investigator",
            connector=PythonAgentConfig(callable="trusted_agent:investigate"),
            test_cases=[
                AgentTestCase(case_id="alert-1", input={"alert_id": "A-1"})
            ],
            reference_dossiers=["reference investigation"],
            metrics=[MetricConfig(name="investigation_quality")],
            judge_model="gemini-test",
        )
    )
    observed = {}

    def fake_run(self, dataset_path, config_path, label=None):
        observed.update(dataset_path=dataset_path, config_path=config_path, label=label)
        return SimpleNamespace(
            run_id="profile-run",
            passed_cases=1,
            failed_cases=0,
            error_cases=0,
        ), []

    monkeypatch.setattr("evalhub.dashboard.EvaluationRunner.run_evaluation", fake_run)
    dashboard = Path(__file__).resolve().parents[1] / "src" / "evalhub" / "dashboard.py"
    app = AppTest.from_file(str(dashboard), default_timeout=15).run()
    next(box for box in app.selectbox if box.label == "Agent connection").select(
        "Registered live-agent profile"
    ).run()
    next(box for box in app.selectbox if box.label == "Runnable agent profile").select(
        "alert-investigator"
    ).run()
    next(
        checkbox for checkbox in app.checkbox
        if checkbox.label.startswith("I trust the connector")
    ).set_value(True)
    next(button for button in app.button if button.label == "Run evaluation").click().run()

    assert not app.exception
    assert observed["dataset_path"] is None
    assert observed["label"] is None
