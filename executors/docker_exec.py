"""Docker executor — the strongest isolation when a Docker daemon is present.

One short-lived container per run, host-mounted workspace, no network. This is
the original sandbox behaviour, now behind the Executor interface.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .base import RUN_FILENAME, BaseExecutor, ExecutorUnavailable

SANDBOX_IMAGE = "llm-gateway-sandbox"


def docker_available() -> bool:
    try:
        return subprocess.run(
            ["docker", "info"], capture_output=True, timeout=10
        ).returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def ensure_image() -> None:
    """Build the sandbox image if it doesn't exist."""
    if subprocess.run(["docker", "image", "inspect", SANDBOX_IMAGE], capture_output=True).returncode == 0:
        return
    dockerfile = Path(__file__).resolve().parent.parent / "Dockerfile.sandbox"
    subprocess.run(
        ["docker", "build", "-f", str(dockerfile), "-t", SANDBOX_IMAGE, str(dockerfile.parent)],
        check=True,
    )


class DockerExecutor(BaseExecutor):
    name = "docker"

    def __init__(self, run_id: str | None = None) -> None:
        super().__init__(run_id)
        self._container: str | None = None

    def is_available(self) -> bool:
        return docker_available()

    def start(self) -> None:
        if not docker_available():
            raise ExecutorUnavailable(
                "Docker is not running. Install it (https://docs.docker.com/get-docker/) "
                "or use a different executor (wasm / subprocess)."
            )
        ensure_image()
        out = subprocess.run(
            [
                "docker", "run", "-d", "--name", f"llm-sandbox-{self.run_id}",
                "-v", f"{self.workspace}:/workspace", "-w", "/workspace",
                "--network", "none", SANDBOX_IMAGE, "tail", "-f", "/dev/null",
            ],
            capture_output=True, text=True, check=True,
        )
        self._container = out.stdout.strip()

    def stop(self) -> None:
        if self._container:
            subprocess.run(["docker", "rm", "-f", self._container], capture_output=True)
            self._container = None

    def _run_script(self) -> tuple[str, str, int]:
        proc = subprocess.run(
            ["docker", "exec", self._container, "python", f"/workspace/{RUN_FILENAME}"],
            capture_output=True, text=True, timeout=120,
        )
        return proc.stdout, proc.stderr, proc.returncode

    def _install(self, package: str) -> tuple[str, str, int]:
        proc = subprocess.run(
            ["docker", "exec", self._container, "pip", "install", package],
            capture_output=True, text=True, timeout=180,
        )
        return proc.stdout, proc.stderr, proc.returncode
