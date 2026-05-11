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
