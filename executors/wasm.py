"""WASM executor — Pyodide (CPython compiled to WebAssembly) under Node.

The preferred Docker-free sandbox: one runtime that behaves identically on
macOS / Linux / Windows, with capability-based isolation (Pyodide sees no host
filesystem except the run's workspace, which we mount via NODEFS, and has no
network at runtime). pandas / openpyxl / matplotlib ship bundled in the Pyodide
npm package, so they load with no network; `install_package` uses micropip.

Setup (one time): needs `node` on PATH and the `pyodide` npm package. `start()`
installs it into a local cache the first time if `npm` is available.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from .base import WORKSPACE_ROOT, BaseExecutor, ExecutorUnavailable

DRIVER = Path(__file__).resolve().parent / "pyodide_driver.mjs"
PYODIDE_CACHE = WORKSPACE_ROOT.parent / "pyodide"
NODE_MODULES = PYODIDE_CACHE / "node_modules"
SENTINEL = "@@RESULT@@"
BOOT_TIMEOUT_S = 120
RUN_TIMEOUT_S = 120


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def pyodide_installed() -> bool:
    return (NODE_MODULES / "pyodide" / "package.json").exists()


def toolkit_cached() -> bool:
    """Whether the scientific wheels (pandas/numpy/…) are bundled locally.

    The slim `pyodide` npm package ships only the core + lockfile and fetches
    package wheels from the CDN at runtime. For an *offline* agent with the
    promised pandas/openpyxl/matplotlib toolkit, the full Pyodide distribution
    (with wheels) must be present — otherwise auto-selection prefers the
    subprocess executor, which has that toolkit in a real venv.
    """
    pkg = NODE_MODULES / "pyodide"
    return pkg.exists() and any(pkg.glob("pandas*.whl"))


class WasmExecutor(BaseExecutor):
    name = "wasm"

    def __init__(self, run_id: str | None = None) -> None:
        super().__init__(run_id)
        self._proc: subprocess.Popen | None = None

    def is_available(self) -> bool:
        # Auto-selection only picks WASM when it can deliver the agent's full
        # toolkit offline (pure-Python still works via explicit `--executor wasm`).
        return _have("node") and pyodide_installed() and toolkit_cached()

    def _ensure_pyodide(self) -> None:
        if pyodide_installed():
            return
        if not _have("node"):
            raise ExecutorUnavailable("Node.js not found — install Node 18+ to use the WASM executor.")
        if not _have("npm"):
            raise ExecutorUnavailable("npm not found — needed once to fetch the 'pyodide' package.")
        PYODIDE_CACHE.mkdir(parents=True, exist_ok=True)
        # One-time, ~hundreds of MB (bundles the scientific package set).
        r = subprocess.run(
            ["npm", "install", "--prefix", str(PYODIDE_CACHE), "pyodide"],
            capture_output=True, text=True, timeout=1800,
        )
        if r.returncode != 0 or not pyodide_installed():
            raise ExecutorUnavailable(f"failed to install pyodide via npm: {r.stderr[-500:]}")

    def start(self) -> None:
        self._ensure_pyodide()
        # Run with cwd=cache so the driver's bare `import "pyodide"` resolves
        # against <cache>/node_modules (paths passed to it are absolute).
        self._proc = subprocess.Popen(
            ["node", str(DRIVER), str(self.workspace), str(NODE_MODULES / "pyodide")],
            cwd=str(PYODIDE_CACHE),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        self._read_result(BOOT_TIMEOUT_S)  # wait for {"ready": true}

    def stop(self) -> None:
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    def _read_result(self, timeout_s: int) -> dict:
        """Read lines until one carries the SENTINEL result (best-effort timeout)."""
        assert self._proc and self._proc.stdout
        import select
        deadline_lines = []
        while True:
            if select.select([self._proc.stdout], [], [], timeout_s)[0]:
                line = self._proc.stdout.readline()
                if not line:
                    raise ExecutorUnavailable("Pyodide driver exited unexpectedly: "
                                              + (self._proc.stderr.read() if self._proc.stderr else ""))
                if line.startswith(SENTINEL):
                    return json.loads(line[len(SENTINEL):])
                deadline_lines.append(line)
            else:
                raise ExecutorUnavailable("Pyodide driver timed out")

    def _send(self, req: dict, timeout_s: int) -> dict:
        assert self._proc and self._proc.stdin
        self._proc.stdin.write(json.dumps(req) + "\n")
        self._proc.stdin.flush()
        return self._read_result(timeout_s)

    def _run_script(self) -> tuple[str, str, int]:
        res = self._send({"cmd": "run"}, RUN_TIMEOUT_S)
        return res.get("stdout", ""), res.get("stderr", ""), res.get("exit_code", 0)

    def _install(self, package: str) -> tuple[str, str, int]:
        res = self._send({"cmd": "install", "package": package}, RUN_TIMEOUT_S)
        return res.get("stdout", ""), res.get("stderr", ""), res.get("exit_code", 0)
