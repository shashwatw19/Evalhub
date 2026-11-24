from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_renders_without_running_an_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALHUB_DB", str(tmp_path / "dashboard.db"))
    dashboard = Path(__file__).resolve().parents[1] / "src" / "evalhub" / "dashboard.py"
    app = AppTest.from_file(str(dashboard)).run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["New evaluation", "Runs", "Compare"]
    app.selectbox[0].select("Python callable").run()
    assert any("Python callable" in field.label for field in app.text_input)
    app.button[0].click().run()
    assert any("Upload a JSONL dataset" in error.value for error in app.error)
