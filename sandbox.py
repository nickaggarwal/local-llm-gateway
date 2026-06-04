"""Docker sandbox for safe Python code execution.

Manages a short-lived Docker container per agent run. Code is written to a
file in a host-mounted workspace and executed via `docker exec`, avoiding
shell-escaping issues. Created files persist on the host after the container
is removed.
"""

from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path

SANDBOX_IMAGE = "llm-gateway-sandbox"
WORKSPACE_ROOT = Path(os.environ.get(
    "LLM_GATEWAY_WORKSPACE",
    Path.home() / ".local-llm-gateway" / "workspaces",
))


def docker_available() -> bool:
    return subprocess.run(
        ["docker", "info"], capture_output=True, timeout=10,
    ).returncode == 0


def ensure_image() -> None:
    """Build the sandbox Docker image if it doesn't exist."""
    check = subprocess.run(
        ["docker", "image", "inspect", SANDBOX_IMAGE],
        capture_output=True,
    )
    if check.returncode == 0:
        return
    dockerfile = Path(__file__).parent / "Dockerfile.sandbox"
    subprocess.run(
        ["docker", "build", "-f", str(dockerfile), "-t", SANDBOX_IMAGE, str(dockerfile.parent)],
        check=True,
    )


class Sandbox:
    """One Docker container for an agent run, with a host-mounted workspace."""

    def __init__(self, run_id: str | None = None) -> None:
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.workspace = WORKSPACE_ROOT / self.run_id
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._container: str | None = None

    # -- lifecycle --

    def start(self) -> None:
        name = f"llm-sandbox-{self.run_id}"
        out = subprocess.run(
            [
                "docker", "run", "-d", "--name", name,
                "-v", f"{self.workspace}:/workspace",
                "-w", "/workspace",
                "--network", "none",
                SANDBOX_IMAGE, "tail", "-f", "/dev/null",
            ],
            capture_output=True, text=True, check=True,
        )
        self._container = out.stdout.strip()

    def stop(self) -> None:
        if self._container:
            subprocess.run(
                ["docker", "rm", "-f", self._container],
                capture_output=True,
            )
            self._container = None

    def __enter__(self) -> Sandbox:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    # -- operations --

    def _workspace_files(self) -> set[str]:
        return {
            str(p.relative_to(self.workspace))
            for p in self.workspace.rglob("*") if p.is_file() and p.name != "_run.py"
        }

    def run_code(self, code: str) -> dict:
        """Execute Python code and return stdout, stderr, exit_code, and new files."""
        before = self._workspace_files()
        script = self.workspace / "_run.py"
        script.write_text(code, encoding="utf-8")
        proc = subprocess.run(
            ["docker", "exec", self._container, "python", "/workspace/_run.py"],
            capture_output=True, text=True, timeout=120,
        )
        script.unlink(missing_ok=True)
        after = self._workspace_files()
        new_files = sorted(after - before)
        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "exit_code": proc.returncode,
            "files": new_files,
        }

    def install_package(self, package: str) -> dict:
        """pip install a package inside the running container."""
        proc = subprocess.run(
            ["docker", "exec", self._container, "pip", "install", package],
            capture_output=True, text=True, timeout=120,
        )
        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "exit_code": proc.returncode,
        }
