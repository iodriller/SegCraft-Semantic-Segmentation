import pytest


fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")


def test_webapp_homepage_loads():
    from fastapi.testclient import TestClient
    from segcraft.webapp import create_app

    client = TestClient(create_app())
    response = client.get("/")

    assert response.status_code == 200
    assert "SegCraft" in response.text
    assert "/jobs" in response.text
    assert "Upload video" in response.text
    assert "YouTube URL" in response.text
    assert "pascal_video" in response.text
    assert "/runtime" in response.text
    assert "Stop" in response.text
    assert "/cancel" in response.text


def test_webapp_runtime_endpoint_loads():
    from fastapi.testclient import TestClient
    from segcraft.webapp import create_app

    client = TestClient(create_app())
    response = client.get("/runtime")

    assert response.status_code == 200
    payload = response.json()
    assert "python_executable" in payload
    assert "packages" in payload
    assert "torch" in payload


def test_webapp_cancel_unknown_job_returns_404():
    from fastapi.testclient import TestClient
    from segcraft.webapp import create_app

    client = TestClient(create_app())
    response = client.post("/jobs/missing/cancel")

    assert response.status_code == 404


def test_webapp_cancel_marks_running_job():
    from fastapi.testclient import TestClient
    from segcraft.webapp import JOBS, JOBS_LOCK, create_app

    with JOBS_LOCK:
        JOBS.clear()
        JOBS["abc123"] = {"id": "abc123", "status": "running", "output_dir": "outputs/test"}

    client = TestClient(create_app())
    response = client.post("/jobs/abc123/cancel")

    assert response.status_code == 200
    assert response.json()["status"] == "canceling"
    with JOBS_LOCK:
        assert JOBS["abc123"]["cancel_requested"] is True


def test_webapp_cancel_leaves_completed_job_completed():
    from fastapi.testclient import TestClient
    from segcraft.webapp import JOBS, JOBS_LOCK, create_app

    with JOBS_LOCK:
        JOBS.clear()
        JOBS["done"] = {"id": "done", "status": "completed", "output_dir": "outputs/test"}

    client = TestClient(create_app())
    response = client.post("/jobs/done/cancel")

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    with JOBS_LOCK:
        assert "cancel_requested" not in JOBS["done"]


def test_run_job_marks_prediction_cancellation(monkeypatch):
    from segcraft.prediction import PredictionCancelled
    from segcraft.webapp import JOBS, JOBS_LOCK, _run_job

    with JOBS_LOCK:
        JOBS.clear()
        JOBS["job1"] = {"id": "job1", "status": "queued", "output_dir": "outputs/test"}

    monkeypatch.setattr("segcraft.webapp._job_config", lambda _params, _input: {})

    def cancel_prediction(*_args, **_kwargs):
        raise PredictionCancelled("Prediction was canceled.")

    monkeypatch.setattr("segcraft.webapp.run_prediction", cancel_prediction)

    _run_job(
        {
            "job_id": "job1",
            "input_path": "input.mp4",
            "youtube_url": None,
        }
    )

    with JOBS_LOCK:
        assert JOBS["job1"]["status"] == "canceled"
        assert JOBS["job1"]["cancel_requested"] is False
        assert JOBS["job1"]["progress"]["stage"] == "canceled"
