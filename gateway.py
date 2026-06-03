"""Core gateway: pick the best model for a task, download it if missing, run it.

All inference is local via Ollama (http://localhost:11434).
"""

import base64
import os

import requests

from registry import Task, get_task

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

OCR_PROMPT = (
    "Extract all text from this image exactly as it appears. "
    "Preserve reading order, line breaks, and layout. "
    "Output only the extracted text with no commentary."
)


def available_ram_gb() -> float:
    """Best-effort total RAM in GB (used to size the model tier)."""
    try:
        import psutil

        return psutil.virtual_memory().total / 1024**3
    except Exception:  # noqa: BLE001
        # Fallback for POSIX without psutil. os.sysconf is absent on Windows.
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
        except (ValueError, OSError, AttributeError):
            return 8.0


def local_models() -> set[str]:
    """Set of model tags already downloaded."""
    resp = requests.get(f"{OLLAMA}/api/tags", timeout=10)
    resp.raise_for_status()
    return {m["name"] for m in resp.json().get("models", [])}


def ensure_model(model: str, on_progress=None) -> None:
    """Download the model via Ollama if it isn't present yet."""
    if model in local_models():
        return
    if on_progress:
        on_progress(f"downloading {model} (first use)...")
    with requests.post(
        f"{OLLAMA}/api/pull", json={"model": model, "stream": True}, stream=True, timeout=3600
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line and on_progress:
                on_progress(line.decode("utf-8"))


def resolve_model(task: Task, override: str | None = None) -> str:
    """Choose the model tag for a task, honoring an explicit override."""
    return override or task.pick_model(available_ram_gb())


def run(
    task_name: str,
    prompt: str | None = None,
    image_path: str | None = None,
    model: str | None = None,
    on_progress=None,
) -> dict:
    """Run a task end to end: select model, ensure it's downloaded, execute it."""
    task = get_task(task_name)
    chosen = resolve_model(task, model)
    ensure_model(chosen, on_progress=on_progress)

    if task.kind == "embed":
        return _embed(chosen, prompt or "")
    if task.kind == "vision":
        if not image_path:
            raise ValueError(f"task '{task_name}' requires --image")
        text = _generate(chosen, prompt or OCR_PROMPT, image_path=image_path)
        return {"task": task_name, "model": chosen, "text": text}

    if not prompt:
        raise ValueError(f"task '{task_name}' requires a prompt")
    text = _generate(chosen, prompt)
    return {"task": task_name, "model": chosen, "text": text}


def _generate(model: str, prompt: str, image_path: str | None = None) -> str:
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0},
    }
    if image_path:
        from pathlib import Path

        b64 = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
        payload["images"] = [b64]
    resp = requests.post(f"{OLLAMA}/api/generate", json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def _embed(model: str, text: str) -> dict:
    resp = requests.post(
        f"{OLLAMA}/api/embeddings", json={"model": model, "prompt": text}, timeout=120
    )
    resp.raise_for_status()
    return {"task": "embed", "model": model, "embedding": resp.json().get("embedding", [])}
