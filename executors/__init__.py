"""Pluggable sandboxes for agent code execution.

`get_executor("auto")` picks the best available without needing Docker:
Docker (strongest, if a daemon is running) → WASM/Pyodide (portable, isolated,
no daemon) → hardened subprocess (always works; blast-radius isolation only).

Force one with `get_executor("docker"|"wasm"|"subprocess")`, the `--executor`
CLI flag, or the `LLM_GATEWAY_EXECUTOR` env var.
"""

from __future__ import annotations

import os

from .base import WORKSPACE_ROOT, BaseExecutor, ExecutorUnavailable
from .docker_exec import DockerExecutor, docker_available
from .subprocess_exec import SubprocessExecutor
from .wasm import WasmExecutor

_EXECUTORS = {
    "docker": DockerExecutor,
    "wasm": WasmExecutor,
    "subprocess": SubprocessExecutor,
}

# Auto-selection order: strongest isolation that's actually usable here.
_AUTO_ORDER = ["docker", "wasm", "subprocess"]


def executor_names() -> list[str]:
    return list(_EXECUTORS)


def available_executor_name() -> str:
    """Name auto-selection would choose right now (without instantiating a run)."""
    for name in _AUTO_ORDER:
        if _EXECUTORS[name]().is_available():
            return name
    return "subprocess"


def get_executor(name: str | None = "auto", run_id: str | None = None) -> BaseExecutor:
    name = (name or os.environ.get("LLM_GATEWAY_EXECUTOR") or "auto").lower()
    if name == "auto":
        name = available_executor_name()
    try:
        return _EXECUTORS[name](run_id)
    except KeyError:
        raise ValueError(
            f"unknown executor '{name}'. available: auto, {', '.join(_EXECUTORS)}"
        ) from None


__all__ = [
    "get_executor", "executor_names", "available_executor_name",
    "BaseExecutor", "ExecutorUnavailable", "docker_available", "WORKSPACE_ROOT",
]
