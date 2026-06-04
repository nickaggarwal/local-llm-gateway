"""Shared pytest setup and fixtures.

Puts the repo root and `eval/` on the import path (the modules import as
top-level names, e.g. `import gateway`, `import scorers`), and provides helpers
to fake the Ollama HTTP API and isolate executor workspaces.
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT, os.path.join(ROOT, "eval")):
    if p not in sys.path:
        sys.path.insert(0, p)


class FakeResp:
    """Minimal stand-in for a requests.Response."""

    def __init__(self, payload: dict | None = None, status: int = 200):
        self._payload = payload or {}
        self.status_code = status

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    # context-manager form used by the streaming /api/pull call
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_lines(self):
        return iter(())


@pytest.fixture
def fake_ollama(monkeypatch):
    """Patch backends.ollama.requests with canned responses; capture posts."""
    import backends.ollama as ob

    calls = {"post": [], "get": []}

    def fake_get(url, *a, **k):
        calls["get"].append((url, k))
        if url.endswith("/api/tags"):
            return FakeResp({"models": [{"name": "qwen2.5:7b"}]})
        return FakeResp({})

    def fake_post(url, *a, **k):
        calls["post"].append((url, k))
        if url.endswith("/api/generate"):
            return FakeResp({"response": "  hello world  "})
        if url.endswith("/api/embeddings"):
            return FakeResp({"embedding": [0.1, 0.2, 0.3]})
        return FakeResp({})

    monkeypatch.setattr(ob.requests, "get", fake_get)
    monkeypatch.setattr(ob.requests, "post", fake_post)
    return calls


@pytest.fixture
def isolated_workspace(monkeypatch, tmp_path):
    """Point executor workspaces at a temp dir."""
    import executors.base as base

    monkeypatch.setattr(base, "WORKSPACE_ROOT", tmp_path)
    return tmp_path
