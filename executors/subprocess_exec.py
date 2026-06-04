"""Hardened subprocess executor — Docker-free, works everywhere.

Runs agent code in a dedicated venv (kept apart from the gateway's own venv,
pre-seeded with pandas/openpyxl/matplotlib), confined to the run's workspace as
its working directory, with a scrubbed environment, resource limits (POSIX), and
a wall-clock timeout.

This is **blast-radius** isolation, not escape-proof: it suits a *trusted* local
model (prevent accidental damage and runaway resource use), not arbitrary
untrusted code. It cannot fully block network access without OS-level sandboxing
— for that, use the Docker or WASM executor.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .base import RUN_FILENAME, WORKSPACE_ROOT, BaseExecutor

AGENT_VENV = WORKSPACE_ROOT.parent / "agent-venv"
BASE_PACKAGES = ["pandas", "openpyxl", "matplotlib"]
TIMEOUT_S = 120


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _resource_limits() -> None:  # POSIX preexec_fn
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT_S, TIMEOUT_S + 5))
        # 512 MB max single output file — stops a runaway from filling the disk.
        resource.setrlimit(resource.RLIMIT_FSIZE, (512 * 1024**2, 512 * 1024**2))
    except Exception:  # noqa: BLE001 - best effort; never block the run
        pass


class SubprocessExecutor(BaseExecutor):
    name = "subprocess"

    def is_available(self) -> bool:
        return True  # always; just needs a Python interpreter

    def start(self) -> None:
        self._ensure_venv()

    def _ensure_venv(self) -> None:
        py = _venv_python(AGENT_VENV)
        marker = AGENT_VENV / ".seeded"
        if py.exists() and marker.exists():
            return
        if not py.exists():
            subprocess.run([sys.executable, "-m", "venv", str(AGENT_VENV)], check=True)
        subprocess.run([str(py), "-m", "pip", "install", "-q", "--upgrade", "pip"], check=False)
        subprocess.run([str(py), "-m", "pip", "install", "-q", *BASE_PACKAGES], check=True)
        marker.write_text("ok", encoding="utf-8")

    def _env(self) -> dict:
        # Minimal environment: keep PATH/locale, point HOME+TMP at the workspace
        # so code can't read/write the real home, and drop everything else
        # (API keys, tokens) that the gateway process may have inherited.
        keep = {k: os.environ[k] for k in ("PATH", "LANG", "LC_ALL", "SYSTEMROOT") if k in os.environ}
        keep["HOME"] = str(self.workspace)
        keep["TMPDIR"] = str(self.workspace)
        keep["MPLCONFIGDIR"] = str(self.workspace / ".mpl")  # matplotlib cache in workspace
        keep["MPLBACKEND"] = "Agg"
        return keep

    def _run_script(self) -> tuple[str, str, int]:
        py = _venv_python(AGENT_VENV)
        kwargs: dict = {}
        if os.name != "nt":
            kwargs["preexec_fn"] = _resource_limits
            kwargs["start_new_session"] = True
        try:
            proc = subprocess.run(
                [str(py), RUN_FILENAME],
                cwd=self.workspace, env=self._env(),
                capture_output=True, text=True, timeout=TIMEOUT_S, **kwargs,
            )
            return proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired:
            return "", f"execution timed out after {TIMEOUT_S}s", 124

    def _install(self, package: str) -> tuple[str, str, int]:
        py = _venv_python(AGENT_VENV)
        proc = subprocess.run(
            [str(py), "-m", "pip", "install", package],
            capture_output=True, text=True, timeout=180,
        )
        return proc.stdout, proc.stderr, proc.returncode
