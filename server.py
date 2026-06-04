"""HTTP API for the local LLM gateway.

Run: uvicorn server:app --port 8000

Endpoints:
    GET  /tasks                      -> available tasks + selected model
    POST /run/{task}  (JSON)         -> text tasks (reasoning, code, summary, embed)
    POST /run/{task}  (multipart)    -> vision tasks (ocr, vision) with file upload
"""

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

import gateway
import hardware
from backends import get_backend
from registry import TASKS, get_task

app = FastAPI(title="Local LLM Gateway", description="Best model per task, run locally (Ollama / Qualcomm NPU).")


class RunRequest(BaseModel):
    prompt: str
    model: str | None = None
    backend: str | None = None


@app.get("/tasks")
def list_tasks(backend: str | None = None) -> dict:
    hw = hardware.describe()
    out = {}
    for name, t in TASKS.items():
        be = get_backend(backend, t)
        try:
            model = be.resolve_model(t, None, hw["budget_gb"])
        except Exception as e:  # noqa: BLE001
            model = None
            out[name] = {"kind": t.kind, "backend": be.name, "model": None,
                         "description": t.description, "error": str(e)}
            continue
        out[name] = {"kind": t.kind, "backend": be.name, "model": model, "description": t.description}
    return {"hardware": hw, "tasks": out}


@app.post("/run/{task}")
def run_text(task: str, req: RunRequest) -> dict:
    t = _get_task_or_404(task)
    if t.kind == "vision":
        raise HTTPException(400, f"task '{task}' needs an image; use multipart upload at this endpoint")
    try:
        return gateway.run(task, prompt=req.prompt, model=req.model, backend=req.backend)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, str(e)) from e


@app.post("/run-image/{task}")
async def run_image(task: str, file: UploadFile = File(...), prompt: str = Form(None),
                    model: str = Form(None), backend: str = Form(None)) -> dict:
    t = _get_task_or_404(task)
    if t.kind != "vision":
        raise HTTPException(400, f"task '{task}' is not a vision task; use POST /run/{task}")
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    import os
    import tempfile

    # Write and close the handle before reading the path (required on Windows,
    # where an open NamedTemporaryFile is locked against other readers).
    suffix = os.path.splitext(file.filename or "image.png")[1] or ".png"
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        return gateway.run(task, prompt=prompt, image_path=path, model=model, backend=backend)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, str(e)) from e
    finally:
        os.unlink(path)


@app.get("/files/{path:path}")
def get_file(path: str):
    """Download a file created by an agent run."""
    from executors import WORKSPACE_ROOT
    full = WORKSPACE_ROOT / path
    if not full.resolve().is_relative_to(WORKSPACE_ROOT.resolve()):
        raise HTTPException(403, "Access denied")
    if not full.exists() or not full.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(full, filename=full.name)


def _get_task_or_404(task: str):
    try:
        return get_task(task)
    except ValueError as e:
        raise HTTPException(404, str(e)) from e
