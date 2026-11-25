from pathlib import Path

from streamlit.testing.v1 import AppTest
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
