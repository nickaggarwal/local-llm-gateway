"""Backend abstraction: a uniform interface every inference engine implements.

The gateway picks the best *model* for a task (in `registry.py`) and the best
*backend* for this machine (Ollama by default, Qualcomm/QNN when a Hexagon NPU
is present), then drives the model through whichever backend was chosen.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from registry import Task

OnProgress = Callable[[str], None]


class BackendUnavailable(RuntimeError):
    """Raised when a backend is selected but can't run here.

    Carries an actionable message (missing SDK, missing hardware, model not yet
    compiled, …) so the CLI/UI/API can surface it verbatim to the user.
    """


class Backend(ABC):
    """Common interface for an inference backend."""

    name: str

    @abstractmethod
    def is_available(self) -> bool:
        """Whether this backend can actually serve requests on this machine."""

    @abstractmethod
    def resolve_model(self, task: "Task", override: str | None, budget_gb: float) -> str:
        """Choose the concrete model id this backend will run for `task`.

        `override` is an explicit user choice; `budget_gb` is the memory budget
        (VRAM if a GPU is present, else RAM) used to size RAM-tiered models.
        """

    @abstractmethod
    def ensure_model(self, model: str, on_progress: OnProgress | None = None) -> None:
        """Make `model` ready to run (download/compile if needed)."""

    @abstractmethod
    def generate(self, model: str, prompt: str, image_path: str | None = None) -> str:
        """Run a text or vision generation and return the response text."""

    @abstractmethod
    def embed(self, model: str, text: str) -> list[float]:
        """Return an embedding vector for `text`."""

    def supports_task(self, task: "Task") -> bool:
        """Whether this backend can serve the given task kind. Default: yes."""
        return True
