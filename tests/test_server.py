"""FastAPI server routing/validation (gateway faked, no inference)."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

import gateway  # noqa: E402
import server  # noqa: E402

client = TestClient(server.app)


@pytest.fixture(autouse=True)
def fake_run(monkeypatch):
    def _run(task, prompt=None, image_path=None, model=None, backend=None, **k):
        if "embed" in task:
            return {"task": task, "model": "m", "backend": "ollama", "embedding": [0.1]}
        return {"task": task, "model": "m", "backend": "ollama", "text": f"ran {task}"}
    monkeypatch.setattr(gateway, "run", _run)


def test_tasks_endpoint_shape():
    r = client.get("/tasks")
    assert r.status_code == 200
    body = r.json()
    assert "hardware" in body and "tasks" in body
    assert "reasoning" in body["tasks"]


def test_run_text_task_ok():
    r = client.post("/run/reasoning", json={"prompt": "hi"})
    assert r.status_code == 200
    assert r.json()["text"] == "ran reasoning"


def test_run_vision_task_via_json_rejected():
    # vision needs an image upload, not the JSON endpoint
    r = client.post("/run/ocr", json={"prompt": "x"})
    assert r.status_code == 400


def test_run_unknown_task_404():
    r = client.post("/run/banana", json={"prompt": "x"})
    assert r.status_code == 404


def test_run_image_endpoint_ok():
    files = {"file": ("x.png", b"\x89PNG\r\n", "image/png")}
    r = client.post("/run-image/ocr", files=files)
    assert r.status_code == 200
    assert r.json()["text"] == "ran ocr"


def test_run_image_on_text_task_rejected():
    files = {"file": ("x.png", b"data", "image/png")}
    r = client.post("/run-image/reasoning", files=files)
    assert r.status_code == 400


def test_files_endpoint_blocks_traversal():
    r = client.get("/files/../../etc/passwd")
    assert r.status_code in (403, 404)
