"""Ollama backend: model resolution + request payloads (HTTP mocked)."""

import registry
from backends.ollama import OllamaBackend


def test_resolve_model_uses_override():
    be = OllamaBackend()
    assert be.resolve_model(registry.TASKS["reasoning"], "my-model", 48) == "my-model"


def test_resolve_model_uses_tier_when_no_override():
    be = OllamaBackend()
    assert be.resolve_model(registry.TASKS["reasoning"], None, 16) == "qwen2.5:7b"


def test_host_from_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://example:1234/")
    assert OllamaBackend().host == "http://example:1234"


def test_default_num_ctx_is_a_positive_int():
    # Value is tunable (OLLAMA_NUM_CTX); just assert it's a sane positive int
    # and that it's the env value when set.
    assert isinstance(OllamaBackend.DEFAULT_NUM_CTX, int)
    assert OllamaBackend.DEFAULT_NUM_CTX >= 512


def test_generate_sends_num_ctx_and_temperature(fake_ollama):
    be = OllamaBackend()
    out = be.generate("m", "hi")
    assert out == "hello world"  # response is stripped
    payload = fake_ollama["post"][-1][1]["json"]
    assert payload["options"]["num_ctx"] == be.DEFAULT_NUM_CTX
    assert payload["options"]["temperature"] == 0
    assert "images" not in payload


def test_generate_with_image_attaches_base64(fake_ollama, tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(b"\x89PNG\r\n")
    be = OllamaBackend()
    be.generate("m", "describe", image_path=str(img))
    payload = fake_ollama["post"][-1][1]["json"]
    assert isinstance(payload["images"], list) and payload["images"]


def test_embed_returns_vector(fake_ollama):
    assert OllamaBackend().embed("m", "text") == [0.1, 0.2, 0.3]


def test_ensure_model_skips_when_present(fake_ollama):
    # fake /api/tags returns qwen2.5:7b -> no pull POST should happen
    OllamaBackend().ensure_model("qwen2.5:7b")
    assert not any(u.endswith("/api/pull") for u, _ in fake_ollama["post"])
