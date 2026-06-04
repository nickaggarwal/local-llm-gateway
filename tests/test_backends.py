"""Backend registry + selection (no inference)."""

import pytest

import backends
from backends import backend_names, get_backend
from registry import TASKS


def test_backend_names():
    assert set(backend_names()) == {"ollama", "qualcomm", "intel-npu", "amd-npu"}


def test_explicit_backend():
    assert get_backend("ollama").name == "ollama"
    assert get_backend("intel-npu").name == "intel-npu"


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        get_backend("tpu")


def test_auto_falls_back_to_ollama_when_no_npu(monkeypatch):
    monkeypatch.setattr(backends.QualcommBackend, "is_available", lambda self: False)
    assert get_backend("auto").name == "ollama"


def test_auto_prefers_qualcomm_when_available_and_supported(monkeypatch):
    monkeypatch.setattr(backends.QualcommBackend, "is_available", lambda self: True)
    monkeypatch.setattr(backends.QualcommBackend, "supports_task", lambda self, t: True)
    assert get_backend("auto", TASKS["reasoning"]).name == "qualcomm"


def test_intel_amd_not_auto_selected(monkeypatch):
    # Even if an Intel/AMD NPU were present, auto must not pick it (opt-in only).
    monkeypatch.setattr(backends.QualcommBackend, "is_available", lambda self: False)
    monkeypatch.setattr(backends.IntelNpuBackend, "is_available", lambda self: True)
    assert get_backend("auto").name == "ollama"
