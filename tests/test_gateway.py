"""Gateway orchestration: dispatch by task kind (backend faked)."""

import pytest

import gateway


class FakeBackend:
    name = "fake"

    def resolve_model(self, task, override, budget):
        return override or "fake-model"

    def ensure_model(self, model, on_progress=None):
        pass

    def generate(self, model, prompt, image_path=None):
        return f"GEN[{prompt}|img={image_path}]"

    def embed(self, model, text):
        return [1.0, 2.0]


@pytest.fixture(autouse=True)
def fake_backend(monkeypatch):
    monkeypatch.setattr(gateway, "get_backend", lambda name, task: FakeBackend())


def test_text_task_returns_text():
    r = gateway.run("reasoning", prompt="2+2?")
    assert r["text"] == "GEN[2+2?|img=None]"
    assert r["backend"] == "fake" and r["model"] == "fake-model"


def test_text_task_requires_prompt():
    with pytest.raises(ValueError):
        gateway.run("reasoning")


def test_embed_task_returns_vector():
    r = gateway.run("embed", prompt="hello")
    assert r["embedding"] == [1.0, 2.0]
    assert "text" not in r


def test_vision_task_requires_image():
    with pytest.raises(ValueError):
        gateway.run("ocr", prompt="no image given")


def test_vision_task_uses_ocr_prompt_default(tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(b"x")
    r = gateway.run("ocr", image_path=str(img))
    assert gateway.OCR_PROMPT.split()[0] in r["text"]  # default prompt was used


def test_model_override_passes_through():
    r = gateway.run("reasoning", prompt="hi", model="custom:7b")
    assert r["model"] == "custom:7b"
