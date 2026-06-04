"""Executor base orchestration + auto-selection (no Docker/Node needed)."""

import subprocess
import sys

import pytest

import executors
from executors import get_executor, executor_names, available_executor_name
from executors.base import BaseExecutor, RUN_FILENAME


class StubExecutor(BaseExecutor):
    """Runs the script with the current interpreter — no venv/Docker/Node."""

    name = "stub"

    def is_available(self) -> bool:
        return True

    def _run_script(self):
        proc = subprocess.run(
            [sys.executable, RUN_FILENAME], cwd=self.workspace,
            capture_output=True, text=True, timeout=30,
        )
        return proc.stdout, proc.stderr, proc.returncode

    def _install(self, package):
        return ("", "", 0)


def test_run_code_captures_stdout_and_new_files(isolated_workspace):
    with StubExecutor() as ex:
        r = ex.run_code("open('made.txt','w').write('hi')\nprint('done')")
    assert r["exit_code"] == 0
    assert "done" in r["stdout"]
    assert r["files"] == ["made.txt"]


def test_run_code_reports_nonzero_exit(isolated_workspace):
    with StubExecutor() as ex:
        r = ex.run_code("raise SystemExit(3)")
    assert r["exit_code"] == 3
    assert r["files"] == []


def test_run_file_is_not_reported_as_output(isolated_workspace):
    with StubExecutor() as ex:
        r = ex.run_code("print('x')")
    assert RUN_FILENAME not in r["files"]


def test_workspace_created_under_root(isolated_workspace):
    ex = StubExecutor(run_id="abc123")
    assert ex.workspace == isolated_workspace / "abc123"
    assert ex.workspace.is_dir()


# --- selection ---

def test_executor_names():
    assert set(executor_names()) == {"docker", "wasm", "subprocess"}


def test_get_executor_unknown_raises():
    with pytest.raises(ValueError):
        get_executor("banana")


def test_auto_prefers_order(monkeypatch):
    # Make docker + wasm unavailable -> auto must land on subprocess.
    monkeypatch.setattr(executors.DockerExecutor, "is_available", lambda self: False)
    monkeypatch.setattr(executors.WasmExecutor, "is_available", lambda self: False)
    assert available_executor_name() == "subprocess"


def test_auto_prefers_docker_when_available(monkeypatch):
    monkeypatch.setattr(executors.DockerExecutor, "is_available", lambda self: True)
    assert available_executor_name() == "docker"


def test_env_var_forces_executor(monkeypatch):
    monkeypatch.setenv("LLM_GATEWAY_EXECUTOR", "subprocess")
    ex = get_executor(None)
    assert ex.name == "subprocess"
