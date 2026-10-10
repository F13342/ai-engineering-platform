import uvicorn

from ai_platform.core.application import Application


def test_application_can_be_created() -> None:
    application = Application()
    assert application is not None


def test_application_run_starts_api_server(monkeypatch) -> None:
    captured = {}

    def fake_run(app, **kwargs):
        captured["app"] = app
        captured.update(kwargs)

    monkeypatch.setattr(uvicorn, "run", fake_run)

    application = Application(host="127.0.0.1", port=8765)
    application.run()

    assert captured["app"] == "ai_platform.api.app:app"
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8765
    assert captured["reload"] is False


def test_application_rejects_invalid_port() -> None:
    import pytest

    with pytest.raises(ValueError, match="port"):
        Application(port=70000)
