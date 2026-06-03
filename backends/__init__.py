"""Pluggable inference backends.

`get_backend("auto", task)` falls back to Ollama unless a Qualcomm Hexagon NPU
is present and serves the task. The Intel/AMD NPU backends are **opt-in only**
(`--backend intel-npu` / `amd-npu`): on x86 the Ollama GPU path (NVIDIA CUDA,
AMD ROCm, Intel Arc via IPEX-LLM) is the better default, so we don't silently
route to a weaker NPU. Qualcomm is the exception — on Snapdragon there is no
Ollama GPU path, so the NPU is the accelerator and is auto-preferred.

Pass an explicit name ("ollama" | "qualcomm" | "intel-npu" | "amd-npu") to force one.
"""

from __future__ import annotations

from registry import Task

from .amd_npu import AmdNpuBackend
from .base import Backend, BackendUnavailable
from .intel_npu import IntelNpuBackend
from .ollama import OllamaBackend
from .qualcomm import QualcommBackend

_BACKENDS: dict[str, Backend] = {
    b.name: b
    for b in (OllamaBackend(), QualcommBackend(), IntelNpuBackend(), AmdNpuBackend())
}

DEFAULT = "ollama"

# NPU backends eligible for *automatic* selection. Only Qualcomm: see module docstring.
_AUTO_NPU = ("qualcomm",)


def backend_names() -> list[str]:
    return list(_BACKENDS)


def get_backend(name: str | None = "auto", task: Task | None = None) -> Backend:
    name = (name or "auto").lower()
    if name == "auto":
        for npu_name in _AUTO_NPU:
            npu = _BACKENDS[npu_name]
            if npu.is_available() and (task is None or npu.supports_task(task)):
                return npu
        return _BACKENDS[DEFAULT]
    try:
        return _BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"unknown backend '{name}'. available: auto, {', '.join(_BACKENDS)}"
        ) from None


__all__ = ["Backend", "BackendUnavailable", "get_backend", "backend_names"]
