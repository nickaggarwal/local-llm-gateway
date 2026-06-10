"""Desktop launcher building blocks.

Everything here is side-effect-light and unit-testable; the window/process
orchestration lives in `desktop/main.py`. Mirrors the behavior of `start.sh`
(Ollama discovery/startup with tuned env, then the Streamlit UI), minus the
venv bootstrap — in the frozen app all Python deps are bundled by PyInstaller.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

APP_NAME = "Local LLM Gateway"


def ollama_base_url() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def repo_root() -> Path:
    """Directory holding app.py: the repo in dev, sys._MEIPASS when frozen."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def ollama_running(base_url: str | None = None, timeout: float = 2.0) -> bool:
    try:
        return requests.get(f"{base_url or ollama_base_url()}/api/version", timeout=timeout).ok
    except requests.RequestException:
        return False


# Checked in order after PATH; covers the official installers on both OSes.
OLLAMA_FALLBACK_PATHS = (
    "/usr/local/bin/ollama",
    "/opt/homebrew/bin/ollama",
    "/Applications/Ollama.app/Contents/Resources/ollama",
    "~/AppData/Local/Programs/Ollama/ollama.exe",
    "%ProgramFiles%/Ollama/ollama.exe",
)


def find_ollama() -> str | None:
    found = shutil.which("ollama")
    if found:
        return found
    for raw in OLLAMA_FALLBACK_PATHS:
        path = Path(os.path.expandvars(os.path.expanduser(raw)))
        if path.is_file():
            return str(path)
    return None


def ollama_serve_env() -> dict[str, str]:
    """Env for `ollama serve`, matching start.sh: flash attention on, q8_0 KV
    cache everywhere (halves VRAM/RAM vs fp16 with negligible quality loss).
    q4_0 is too aggressive — it corrupts vision models like qwen2.5-VL, which
    then emit empty/garbage output. User-exported values win."""
    env = dict(os.environ)
    env.setdefault("OLLAMA_FLASH_ATTENTION", "1")
    env.setdefault("OLLAMA_KV_CACHE_TYPE", "q8_0")
    env.setdefault("OLLAMA_GPU_OVERHEAD", "0")
    env.setdefault("OLLAMA_CONTEXT_LENGTH", "8192")
    return env


def start_ollama(binary: str) -> subprocess.Popen:
    """Start `ollama serve` detached so it outlives the app (like start.sh)."""
    log_dir = Path.home() / ".local-llm-gateway"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = open(log_dir / "ollama.log", "ab")
    kwargs: dict = {"stdout": log, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL}
    if os.name == "posix":
        kwargs["start_new_session"] = True
    else:
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
        )
    return subprocess.Popen([binary, "serve"], env=ollama_serve_env(), **kwargs)


def wait_for_http(url: str, timeout_s: float, interval_s: float = 0.5) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if requests.get(url, timeout=2).ok:
                return True
        except requests.RequestException:
            pass
        time.sleep(interval_s)
    return False


def streamlit_command(port: int) -> list[str]:
    """Argv for the Streamlit child process. Frozen: re-exec our own binary
    with the server flag; dev: run desktop.main as a module."""
    flag = ["--streamlit-server", "--port", str(port)]
    if getattr(sys, "frozen", False):
        return [sys.executable, *flag]
    return [sys.executable, "-m", "desktop.main", *flag]


def streamlit_args(port: int) -> list[str]:
    """CLI args for `streamlit run`. developmentMode must be off explicitly in
    the frozen app (Streamlit flips it on when it can't find its pip install)."""
    return [
        "streamlit", "run", str(repo_root() / "app.py"),
        "--server.address", "127.0.0.1",
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--global.developmentMode", "false",
    ]


def run_streamlit(port: int) -> None:
    """Run the Streamlit server in this process (the child's main thread, so
    Streamlit's signal handlers install cleanly). Does not return."""
    import streamlit.web.cli as stcli

    sys.argv = streamlit_args(port)
    sys.exit(stcli.main())


def spawn_streamlit(port: int) -> subprocess.Popen:
    kwargs: dict = {"cwd": str(repo_root())}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    return subprocess.Popen(streamlit_command(port), **kwargs)
