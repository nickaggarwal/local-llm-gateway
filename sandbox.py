"""Backwards-compatibility shim.

The Docker sandbox moved into the pluggable `executors/` package (which also
adds Docker-free WASM and subprocess executors). Import from `executors`
instead; this module re-exports the old names so existing references keep working.
"""

from __future__ import annotations

from executors import WORKSPACE_ROOT, get_executor  # noqa: F401
from executors.docker_exec import (  # noqa: F401
    SANDBOX_IMAGE,
    DockerExecutor as Sandbox,
    docker_available,
    ensure_image,
)

__all__ = ["WORKSPACE_ROOT", "get_executor", "Sandbox", "docker_available", "ensure_image", "SANDBOX_IMAGE"]
