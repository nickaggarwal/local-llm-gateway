"""Desktop app entry point: native window around the Streamlit UI.

Two roles, selected by argv (PyInstaller freezes a single binary):
  - default            — ensure Ollama is up, spawn the Streamlit server as a
                         child process, show it in a pywebview window (or the
                         default browser if pywebview is unavailable).
  - --streamlit-server — run the Streamlit server itself (the child process).

Build with desktop/build_macos.sh or desktop/build_windows.ps1.
"""

from __future__ import annotations

import argparse
import atexit
import multiprocessing
import subprocess
import sys
import threading
import webbrowser

from desktop import launcher

_PAGE = """
<html><body style="font-family: -apple-system, 'Segoe UI', sans-serif;
  background:#0e1117; color:#fafafa; display:flex; align-items:center;
  justify-content:center; height:100%; margin:0">
<div style="max-width:34em; text-align:center">{body}</div>
</body></html>
"""
LOADING_HTML = _PAGE.format(body="<h2>🧠 Local LLM Gateway</h2><p>Starting…</p>")
INSTALL_OLLAMA_HTML = _PAGE.format(body=(
    "<h2>🧠 Local LLM Gateway</h2>"
    "<p><b>Ollama is required</b> to run models locally, and it wasn't found "
    "on this machine.</p>"
    "<p>Install it from <a style='color:#7ab8ff' "
    "href='https://ollama.com/download'>ollama.com/download</a>, launch it, "
    "and this window will continue automatically.</p>"
))

_streamlit_proc: subprocess.Popen | None = None


def _shutdown() -> None:
    """Stop the Streamlit child. Ollama (if we started it) is left running, like
    start.sh — killing it would drop loaded models and slow the next launch."""
    global _streamlit_proc
    if _streamlit_proc and _streamlit_proc.poll() is None:
        _streamlit_proc.terminate()
        try:
            _streamlit_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _streamlit_proc.kill()
    _streamlit_proc = None


def _ensure_ollama(window) -> bool:
    if launcher.ollama_running():
        return True
    binary = launcher.find_ollama()
    if binary:
        launcher.start_ollama(binary)
        return launcher.wait_for_http(f"{launcher.ollama_base_url()}/api/version", 30)
    if window:
        window.load_html(INSTALL_OLLAMA_HTML)
    else:
        print("Ollama not found. Install it from https://ollama.com/download — waiting...")
    # Wait (generously) for the user to install and start it.
    return launcher.wait_for_http(f"{launcher.ollama_base_url()}/api/version", 15 * 60, interval_s=2)


def _prepare_npu_models() -> None:
    """start.sh parity (`convert.py --auto`): pre-build NPU artifacts when an
    NPU is present; a clean no-op everywhere else. Best-effort — Ollama is
    always the fallback."""
    try:
        import hardware
        from backends import BackendUnavailable, get_backend
        from registry import TASKS

        if hardware.has_qualcomm_npu():
            name = "qualcomm"
        elif hardware.has_intel_npu():
            name = "intel-npu"
        elif hardware.has_amd_npu():
            name = "amd-npu"
        else:
            return
        backend = get_backend(name)
        budget = hardware.memory_budget_gb()
        for task in TASKS.values():
            if backend.supports_task(task):
                try:
                    backend.ensure_model(backend.resolve_model(task, None, budget))
                except BackendUnavailable:
                    pass
    except Exception:  # noqa: BLE001
        pass


def _boot(window, port: int) -> None:
    global _streamlit_proc
    url = f"http://127.0.0.1:{port}"
    try:
        if not _ensure_ollama(window):
            raise RuntimeError("Ollama did not become reachable.")
        if window:
            window.load_html(LOADING_HTML)
        _streamlit_proc = launcher.spawn_streamlit(port)
        if not launcher.wait_for_http(f"{url}/_stcore/health", 120):
            raise RuntimeError("The UI server failed to start.")
        if window:
            window.load_url(url)
        else:
            webbrowser.open(url)
        threading.Thread(target=_prepare_npu_models, daemon=True).start()
    except Exception as e:  # noqa: BLE001 — surface anything in the window
        msg = _PAGE.format(body=f"<h2>🧠 Local LLM Gateway</h2><p>⚠️ {e}</p>")
        if window:
            window.load_html(msg)
        else:
            print(f"ERROR: {e}", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(prog=launcher.APP_NAME)
    parser.add_argument("--streamlit-server", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args(argv)

    if args.streamlit_server:
        launcher.run_streamlit(args.port or launcher.find_free_port())
        return

    port = args.port or launcher.find_free_port()
    atexit.register(_shutdown)

    try:
        import webview
    except ImportError:
        webview = None

    if webview:
        window = webview.create_window(
            launcher.APP_NAME, html=LOADING_HTML, width=1100, height=800, min_size=(700, 500)
        )
        # webview.start blocks on the main thread (required on macOS); boot in parallel.
        webview.start(_boot, (window, port))
        _shutdown()  # window closed
    else:
        _boot(None, port)
        if _streamlit_proc:
            try:
                _streamlit_proc.wait()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
