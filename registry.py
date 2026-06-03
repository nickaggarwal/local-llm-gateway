"""Task-to-model registry.

Maps each task type to the best local model for the job. On first use the
gateway downloads the model automatically (via Ollama), then runs it.

`tiers` lets the gateway pick a model sized to the machine's total RAM:
the largest tier whose `min_ram_gb` fits is selected.

`min_ram_gb` is expressed in *total laptop RAM*, not GPU VRAM. A laptop must
leave headroom for the OS and apps, so a 16 GB machine has only ~10 GB usable
for a model — that's why a 7B (~5 GB at Q4) is the safe pick at 16 GB, a 14B
at 24 GB, and a 32B at 32–48 GB. Thresholds are set at 6 / 16 / 24 / 32 / 48.

Model choices reflect 2026 benchmarks: Qwen2.5-VL for OCR/vision (best DocVQA
among local VLMs), Qwen2.5-Coder for code (matches GPT-4o on HumanEval at 32B),
Qwen2.5 for chat/summarize, and nomic-embed-text / mxbai-embed-large for
embeddings. See README "Model selection" for sources.
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


# Resolved model per laptop RAM tier (for reference; see README):
#   TASK        16 GB                24 GB                32 GB                48 GB
#   ocr         qwen2.5vl:7b         qwen2.5vl:7b         qwen2.5vl:7b         qwen2.5vl:32b
#   vision      qwen2.5vl:7b         qwen2.5vl:7b         qwen2.5vl:32b        qwen2.5vl:32b
#   chat        qwen2.5:7b           qwen2.5:14b          qwen2.5:32b          qwen2.5:32b
#   code        qwen2.5-coder:7b     qwen2.5-coder:14b    qwen2.5-coder:32b    qwen2.5-coder:32b
#   summarize   qwen2.5:7b           qwen2.5:14b          qwen2.5:14b          qwen2.5:14b
#   embed       mxbai-embed-large    mxbai-embed-large    mxbai-embed-large    mxbai-embed-large
TASKS: dict[str, Task] = {
    "ocr": Task(
        name="ocr",
        kind="vision",
        description="Extract text from an image, preserving layout.",
        tiers=[
            ModelTier("qwen2.5vl:3b", 6),
            ModelTier("qwen2.5vl:7b", 16),
            ModelTier("qwen2.5vl:32b", 48),
        ],
    ),
    "vision": Task(
        name="vision",
        kind="vision",
        description="Describe or answer questions about an image.",
        tiers=[
            ModelTier("qwen2.5vl:3b", 6),
            ModelTier("qwen2.5vl:7b", 16),
            ModelTier("qwen2.5vl:32b", 32),
        ],
    ),
    "chat": Task(
        name="chat",
        kind="text",
        description="General conversation and Q&A.",
        tiers=[
            ModelTier("llama3.2:3b", 6),
            ModelTier("qwen2.5:7b", 16),
            ModelTier("qwen2.5:14b", 24),
            ModelTier("qwen2.5:32b", 32),
        ],
    ),
    "code": Task(
        name="code",
        kind="text",
        description="Code generation, completion, and explanation.",
        tiers=[
            ModelTier("qwen2.5-coder:3b", 6),
            ModelTier("qwen2.5-coder:7b", 16),
            ModelTier("qwen2.5-coder:14b", 24),
            ModelTier("qwen2.5-coder:32b", 32),
        ],
    ),
    "summarize": Task(
        name="summarize",
        kind="text",
        description="Summarize long text into key points.",
        tiers=[
            ModelTier("llama3.2:3b", 6),
            ModelTier("qwen2.5:7b", 16),
            ModelTier("qwen2.5:14b", 24),
        ],
    ),
    "embed": Task(
        name="embed",
        kind="embed",
        description="Produce vector embeddings for text.",
        tiers=[
            ModelTier("nomic-embed-text", 2),
            ModelTier("mxbai-embed-large", 16),
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
