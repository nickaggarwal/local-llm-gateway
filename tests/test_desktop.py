"""Desktop launcher logic — hermetic (no Ollama, no network, no windows)."""

from __future__ import annotations

import socket
import sys

import pytest

from desktop import launcher


# --- ports -------------------------------------------------------------------

def test_find_free_port_is_bindable():
    port = launcher.find_free_port()
    assert 1024 <= port <= 65535
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", port))  # must not raise


# --- repo root ---------------------------------------------------------------

def test_repo_root_dev_mode_holds_app_py():
    assert (launcher.repo_root() / "app.py").is_file()


def test_repo_root_frozen_uses_meipass(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert launcher.repo_root() == tmp_path


# --- Ollama discovery --------------------------------------------------------

def test_find_ollama_prefers_path(monkeypatch):
    monkeypatch.setattr(launcher.shutil, "which", lambda _: "/somewhere/ollama")
    assert launcher.find_ollama() == "/somewhere/ollama"


def test_find_ollama_checks_fallback_locations(monkeypatch, tmp_path):
    binary = tmp_path / "ollama"
    binary.write_text("")
    monkeypatch.setattr(launcher.shutil, "which", lambda _: None)
    monkeypatch.setattr(launcher, "OLLAMA_FALLBACK_PATHS", (str(binary),))
    assert launcher.find_ollama() == str(binary)


def test_find_ollama_missing(monkeypatch):
    monkeypatch.setattr(launcher.shutil, "which", lambda _: None)
    monkeypatch.setattr(launcher, "OLLAMA_FALLBACK_PATHS", ())
    assert launcher.find_ollama() is None


def test_ollama_running_true_and_false(monkeypatch):
    class Resp:
        ok = True

    monkeypatch.setattr(launcher.requests, "get", lambda *a, **k: Resp())
    assert launcher.ollama_running() is True

    def boom(*a, **k):
        raise launcher.requests.ConnectionError()

    monkeypatch.setattr(launcher.requests, "get", boom)
    assert launcher.ollama_running() is False


def test_ollama_base_url_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://example:1234")
    assert launcher.ollama_base_url() == "http://example:1234"


# --- serve env (start.sh parity) ----------------------------------------------

@pytest.mark.parametrize("gpu,kv", [(True, "q8_0"), (False, "q4_0")])
def test_ollama_serve_env_kv_cache(monkeypatch, gpu, kv):
    import hardware

    monkeypatch.setattr(hardware, "has_gpu", lambda: gpu)
    monkeypatch.delenv("OLLAMA_KV_CACHE_TYPE", raising=False)
    monkeypatch.delenv("OLLAMA_FLASH_ATTENTION", raising=False)
    env = launcher.ollama_serve_env()
    assert env["OLLAMA_KV_CACHE_TYPE"] == kv
    assert env["OLLAMA_FLASH_ATTENTION"] == "1"


def test_ollama_serve_env_user_override_wins(monkeypatch):
    import hardware

    monkeypatch.setattr(hardware, "has_gpu", lambda: True)
    monkeypatch.setenv("OLLAMA_KV_CACHE_TYPE", "f16")
    assert launcher.ollama_serve_env()["OLLAMA_KV_CACHE_TYPE"] == "f16"


# --- Streamlit child process ---------------------------------------------------

def test_streamlit_command_dev_mode():
    cmd = launcher.streamlit_command(8765)
    assert cmd[:3] == [sys.executable, "-m", "desktop.main"]
    assert "--streamlit-server" in cmd and "8765" in cmd


def test_streamlit_command_frozen(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    cmd = launcher.streamlit_command(8765)
    assert cmd[0] == sys.executable
    assert "-m" not in cmd
    assert "--streamlit-server" in cmd


def test_streamlit_args_shape():
    args = launcher.streamlit_args(9000)
    assert args[:2] == ["streamlit", "run"]
    assert args[2].endswith("app.py")
    assert "9000" in args
    # frozen apps must not run in Streamlit's development mode
    i = args.index("--global.developmentMode")
    assert args[i + 1] == "false"


def test_wait_for_http_times_out_fast(monkeypatch):
    def boom(*a, **k):
        raise launcher.requests.ConnectionError()

    monkeypatch.setattr(launcher.requests, "get", boom)
    assert launcher.wait_for_http("http://127.0.0.1:1/x", timeout_s=0.2, interval_s=0.05) is False
