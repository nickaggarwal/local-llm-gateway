"""Core gateway: pick the best model and backend for a task, then run it.

Model selection lives in `registry.py` (sized to the machine's memory budget —
GPU VRAM when present, else RAM; see `hardware.py`). Backend selection lives in
`backends/`: Ollama by default, with the Qualcomm/Hexagon-NPU backend chosen
automatically when a Snapdragon NPU is detected.
"""

from __future__ import annotations

import hardware
from backends import BackendUnavailable, get_backend
from registry import Task, get_task

# Re-exported for callers that referenced these on the gateway module.
BackendUnavailable = BackendUnavailable

OCR_PROMPT = (
    "Extract all text from this image exactly as it appears. "
    "Preserve reading order, line breaks, and layout. "
    "Output only the extracted text with no commentary."
)


def available_ram_gb() -> float:
    """Total system RAM in GB (kept for backwards compatibility)."""
    return hardware.total_ram_gb()


def resolve_model(task: Task, override: str | None = None, backend: str | None = None) -> str:
    """Choose the model id a task will run, for the selected backend."""
    chosen_backend = get_backend(backend, task)
    return chosen_backend.resolve_model(task, override, hardware.memory_budget_gb())


def run(
    task_name: str,
    prompt: str | None = None,
    image_path: str | None = None,
    model: str | None = None,
    backend: str | None = None,
    on_progress=None,
) -> dict:
    """Run a task end to end: select backend + model, ensure it's ready, execute it."""
    task = get_task(task_name)
    be = get_backend(backend, task)
    chosen = be.resolve_model(task, model, hardware.memory_budget_gb())
    be.ensure_model(chosen, on_progress=on_progress)

    result = {"task": task_name, "model": chosen, "backend": be.name}

    if task.kind == "embed":
        result["embedding"] = be.embed(chosen, prompt or "")
        return result
    if task.kind == "vision":
        if not image_path:
            raise ValueError(f"task '{task_name}' requires --image")
        result["text"] = be.generate(chosen, prompt or OCR_PROMPT, image_path=image_path)
        return result

    if task.kind == "agent":
        if not prompt:
            raise ValueError(f"task '{task_name}' requires a prompt")
        from agent import run_agent_loop
        agent_result = run_agent_loop(chosen, prompt, on_progress=on_progress)
        result["text"] = agent_result["text"]
        result["files"] = agent_result.get("files", [])
        return result

    if not prompt:
        raise ValueError(f"task '{task_name}' requires a prompt")
    result["text"] = be.generate(chosen, prompt)
    return result
