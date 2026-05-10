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
