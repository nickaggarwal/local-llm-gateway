"""HTTP API for the local LLM gateway.

Run: uvicorn server:app --port 8000

Endpoints:
    GET  /tasks                      -> available tasks + selected model
    POST /run/{task}  (JSON)         -> text tasks (chat, code, summarize, embed)
    POST /run/{task}  (multipart)    -> vision tasks (ocr, vision) with file upload
"""

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

import gateway
from registry import TASKS, get_task

app = FastAPI(title="Local LLM Gateway", description="Best model per task, run locally via Ollama.")


class RunRequest(BaseModel):
    prompt: str
    model: str | None = None


@app.get("/tasks")
def list_tasks() -> dict:
    ram = gateway.available_ram_gb()
    return {
        "ram_gb": round(ram, 1),
        "tasks": {
            name: {"kind": t.kind, "model": t.pick_model(ram), "description": t.description}
            for name, t in TASKS.items()
        },
    }


@app.post("/run/{task}")
def run_text(task: str, req: RunRequest) -> dict:
    t = _get_task_or_404(task)
    if t.kind == "vision":
        raise HTTPException(400, f"task '{task}' needs an image; use multipart upload at this endpoint")
    try:
        return gateway.run(task, prompt=req.prompt, model=req.model)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, str(e)) from e


@app.post("/run-image/{task}")
async def run_image(task: str, file: UploadFile = File(...), prompt: str = Form(None), model: str = Form(None)) -> dict:
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
        return gateway.run(task, prompt=prompt, image_path=path, model=model)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, str(e)) from e
    finally:
        os.unlink(path)


def _get_task_or_404(task: str):
    try:
        return get_task(task)
    except ValueError as e:
        raise HTTPException(404, str(e)) from e
