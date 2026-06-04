"""Ollama backend — the default, runs everywhere Ollama runs.

Talks to the Ollama HTTP API (default http://localhost:11434). Ollama itself
transparently uses an NVIDIA GPU when one is present, so there is nothing to
enable here for GPU inference; the gateway's only GPU-specific behaviour is
sizing the model tier off VRAM (see `hardware.memory_budget_gb`).
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import requests

from registry import Task

from .base import Backend, BackendUnavailable, OnProgress


class OllamaBackend(Backend):
    name = "ollama"

    # Ollama's built-in default context is only ~2048 tokens, which truncates
    # long inputs (big OCR pages, long passages to summarize, multi-turn chat).
    # Default to an 8K window for the common case; override with OLLAMA_NUM_CTX.
    DEFAULT_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))

    def __init__(self, host: str | None = None) -> None:
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")

    def is_available(self) -> bool:
        try:
            requests.get(f"{self.host}/api/tags", timeout=3).raise_for_status()
            return True
        except requests.RequestException:
            return False

    def resolve_model(self, task: Task, override: str | None, budget_gb: float) -> str:
        return override or task.pick_model(budget_gb)

    def _local_models(self) -> set[str]:
        resp = requests.get(f"{self.host}/api/tags", timeout=10)
        resp.raise_for_status()
        return {m["name"] for m in resp.json().get("models", [])}

    def ensure_model(self, model: str, on_progress: OnProgress | None = None) -> None:
        try:
            present = model in self._local_models()
        except requests.RequestException as e:
            raise BackendUnavailable(
                f"can't reach Ollama at {self.host} — is it running? ({e})"
            ) from e
        if present:
            return
        import json as _json, sys
        print(f"Downloading model {model} (first run, this may take a few minutes)...", file=sys.stderr)
        if on_progress:
            on_progress(f"downloading {model} (first use)...")
        with requests.post(
            f"{self.host}/api/pull",
            json={"model": model, "stream": True},
            stream=True,
            timeout=3600,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    msg = _json.loads(line)
                except (ValueError, TypeError):
                    msg = {}
                total = msg.get("total")
                completed = msg.get("completed")
                status = msg.get("status", "")
                if total and completed:
                    pct = completed * 100 // total
                    done = pct // 5
                    bar = "█" * done + "░" * (20 - done)
                    size_gb = total / (1024 ** 3)
                    print(f"\r  [{bar}] {pct}% of {size_gb:.1f} GB", end="", file=sys.stderr, flush=True)
                elif status:
                    print(f"\r  {status:<60}", end="", file=sys.stderr, flush=True)
            print(file=sys.stderr)  # newline after progress

    def generate(self, model: str, prompt: str, image_path: str | None = None) -> str:
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "num_ctx": self.DEFAULT_NUM_CTX},
        }
        if image_path:
            b64 = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
            payload["images"] = [b64]
        resp = requests.post(f"{self.host}/api/generate", json=payload, timeout=600)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    def embed(self, model: str, text: str) -> list[float]:
        resp = requests.post(
            f"{self.host}/api/embeddings", json={"model": model, "prompt": text}, timeout=120
        )
        resp.raise_for_status()
        return resp.json().get("embedding", [])
