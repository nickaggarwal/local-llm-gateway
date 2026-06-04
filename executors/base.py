"""Executor abstraction: run agent-generated Python in *some* sandbox.

Mirrors the `backends/` pattern. The agent loop drives an `Executor` without
caring how isolation is achieved (Docker container, WASM/Pyodide, or a hardened
local subprocess). Each executor owns a host workspace directory; files the code
writes there persist after the run and are returned to the user.

The shared `BaseExecutor` handles the workspace + new-file diffing; subclasses
implement how to actually run `/workspace/_run.py` and install a package.
"""

from __future__ import annotations

import os
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

WORKSPACE_ROOT = Path(os.environ.get(
    "LLM_GATEWAY_WORKSPACE",
    Path.home() / ".local-llm-gateway" / "workspaces",
))

RUN_FILENAME = "_run.py"


class ExecutorUnavailable(RuntimeError):
    """Raised when an executor is selected but can't run here (missing runtime)."""


class BaseExecutor(ABC):
    """One sandboxed run with a host-mounted workspace.

    Subclasses implement `_run_script()` (execute `workspace/_run.py`) and
    `_install(package)`; everything else (workspace lifecycle, file diffing) is
    shared so every executor returns the same shape.
    """

    name: str = "base"

    def __init__(self, run_id: str | None = None) -> None:
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.workspace = WORKSPACE_ROOT / self.run_id
        self.workspace.mkdir(parents=True, exist_ok=True)

    # -- availability / lifecycle (override as needed) --

    @abstractmethod
    def is_available(self) -> bool:
        """Whether this executor can run on this machine right now."""

    def start(self) -> None:  # noqa: B027 - intentionally optional
        """Prepare the executor (start a container, build a venv, …)."""

    def stop(self) -> None:  # noqa: B027 - intentionally optional
        """Tear down any per-run resources."""

    def __enter__(self) -> "BaseExecutor":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    # -- subclass hooks --

    @abstractmethod
    def _run_script(self) -> tuple[str, str, int]:
        """Run `self.workspace/_run.py`; return (stdout, stderr, exit_code)."""

    @abstractmethod
    def _install(self, package: str) -> tuple[str, str, int]:
        """Install a package so later runs can import it; return (out, err, code)."""

    # -- public operations (shared) --

    def _workspace_files(self) -> set[str]:
        return {
            str(p.relative_to(self.workspace))
            for p in self.workspace.rglob("*")
            if p.is_file() and p.name != RUN_FILENAME
        }

    def run_code(self, code: str) -> dict:
        """Execute Python code; return stdout, stderr, exit_code, and new files."""
        before = self._workspace_files()
        (self.workspace / RUN_FILENAME).write_text(code, encoding="utf-8")
        try:
            stdout, stderr, exit_code = self._run_script()
        finally:
            (self.workspace / RUN_FILENAME).unlink(missing_ok=True)
        new_files = sorted(self._workspace_files() - before)
        return {"stdout": stdout, "stderr": stderr, "exit_code": exit_code, "files": new_files}

    def install_package(self, package: str) -> dict:
        stdout, stderr, exit_code = self._install(package)
        return {"stdout": stdout, "stderr": stderr, "exit_code": exit_code}
