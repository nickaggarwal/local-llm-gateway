"""Integration tests — real services. Opt in with:  pytest -m integration

Each test self-skips when its dependency (Ollama / Node+Pyodide / venv build)
isn't available, so the suite is safe to run anywhere.
"""

import shutil

import pytest

pytestmark = pytest.mark.integration


def _ollama_up() -> bool:
    try:
        import requests
        requests.get("http://localhost:11434/api/tags", timeout=2).raise_for_status()
        return True
    except Exception:
        return False


def test_subprocess_executor_runs_real_code(isolated_workspace):
    """Builds the agent venv (slow, network) and runs pandas end-to-end."""
    from executors import get_executor
    with get_executor("subprocess") as ex:
        r = ex.run_code(
            "import pandas as pd\n"
            "pd.DataFrame({'a':[1,2,3]}).to_csv('t.csv', index=False)\n"
            "print('ok', 3)"
        )
    assert r["exit_code"] == 0
    assert "ok 3" in r["stdout"]
    assert "t.csv" in r["files"]


def test_wasm_executor_pure_python(isolated_workspace):
    from executors.wasm import WasmExecutor, pyodide_installed
    if not (shutil.which("node") and pyodide_installed()):
        pytest.skip("node + pyodide not set up")
    from executors import get_executor
    with get_executor("wasm") as ex:
        r = ex.run_code("open('o.txt','w').write('hi')\nprint('wasm ok')")
    assert r["exit_code"] == 0
    assert "wasm ok" in r["stdout"]
    assert "o.txt" in r["files"]


def test_ollama_embed_roundtrip():
    if not _ollama_up():
        pytest.skip("Ollama not running")
    from backends.ollama import OllamaBackend
    be = OllamaBackend()
    if "nomic-embed-text:latest" not in be._local_models() and "nomic-embed-text" not in be._local_models():
        pytest.skip("nomic-embed-text not pulled")
    vec = be.embed("nomic-embed-text", "hello world")
    assert isinstance(vec, list) and len(vec) > 10


def test_gateway_reasoning_end_to_end():
    if not _ollama_up():
        pytest.skip("Ollama not running")
    import gateway
    import registry
    import hardware
    model = registry.TASKS["reasoning"].pick_model(hardware.memory_budget_gb())
    be_models = __import__("backends.ollama", fromlist=["OllamaBackend"]).OllamaBackend()._local_models()
    if not any(model.split(":")[0] in m for m in be_models):
        pytest.skip(f"{model} not pulled")
    r = gateway.run("reasoning", prompt="What is 6 times 7? Answer with just the number.", model=model)
    assert "42" in r["text"]
