"""Task-to-model registry.

Maps each task type to the best local model for the job. On first use the
gateway downloads the model automatically (via Ollama), then runs it.

`tiers` lets the gateway pick a model sized to the machine's available RAM:
the largest tier whose `min_ram_gb` fits is selected.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelTier:
    model: str
    min_ram_gb: float


@dataclass(frozen=True)
class Task:
    name: str
    kind: str  # "text" | "vision" | "embed"
    description: str
    tiers: list[ModelTier] = field(default_factory=list)

    def pick_model(self, available_ram_gb: float) -> str:
        """Return the largest model whose RAM requirement fits; fall back to smallest."""
        affordable = [t for t in self.tiers if t.min_ram_gb <= available_ram_gb]
        if affordable:
            return max(affordable, key=lambda t: t.min_ram_gb).model
        return min(self.tiers, key=lambda t: t.min_ram_gb).model


TASKS: dict[str, Task] = {
    "ocr": Task(
        name="ocr",
        kind="vision",
        description="Extract text from an image, preserving layout.",
        tiers=[
            ModelTier("qwen2.5vl:3b", 6),
            ModelTier("qwen2.5vl:7b", 12),
        ],
    ),
    "vision": Task(
        name="vision",
        kind="vision",
        description="Describe or answer questions about an image.",
        tiers=[
            ModelTier("qwen2.5vl:3b", 6),
            ModelTier("qwen2.5vl:7b", 12),
            ModelTier("llama3.2-vision:11b", 24),
        ],
    ),
    "chat": Task(
        name="chat",
        kind="text",
        description="General conversation and Q&A.",
        tiers=[
            ModelTier("llama3.2:3b", 6),
            ModelTier("qwen2.5:7b", 12),
            ModelTier("qwen2.5:14b", 24),
        ],
    ),
    "code": Task(
        name="code",
        kind="text",
        description="Code generation, completion, and explanation.",
        tiers=[
            ModelTier("qwen2.5-coder:3b", 6),
            ModelTier("qwen2.5-coder:7b", 12),
            ModelTier("qwen2.5-coder:14b", 24),
        ],
    ),
    "summarize": Task(
        name="summarize",
        kind="text",
        description="Summarize long text into key points.",
        tiers=[
            ModelTier("llama3.2:3b", 6),
            ModelTier("qwen2.5:7b", 12),
        ],
    ),
    "embed": Task(
        name="embed",
        kind="embed",
        description="Produce vector embeddings for text.",
        tiers=[
            ModelTier("nomic-embed-text", 2),
        ],
    ),
}


def get_task(name: str) -> Task:
    try:
        return TASKS[name]
    except KeyError:
        raise ValueError(
            f"unknown task '{name}'. available: {', '.join(TASKS)}"
        ) from None
