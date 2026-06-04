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
Qwen2.5 for reasoning, Llama 3.1 / Qwen2.5 for summary, and bge-m3 / nomic for
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
    kind: str  # "text" | "vision" | "embed" | "agent"
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
#   ocr         qwen2.5vl:7b         qwen2.5vl:7b         qwen2.5vl:7b         qwen2.5vl:7b
#   vision      qwen2.5vl:7b         qwen2.5vl:7b         qwen2.5vl:7b         qwen2.5vl:7b
#   reasoning   qwen2.5:7b           qwen2.5:14b          qwen2.5:14b          qwen2.5:14b
#   code        qwen2.5-coder:7b     qwen2.5-coder:14b    qwen2.5-coder:32b    qwen2.5-coder:32b
#   summary     llama3.1:8b          llama3.1:8b          qwen2.5:32b          qwen2.5:32b
#   embed       bge-m3               bge-m3               bge-m3               bge-m3
TASKS: dict[str, Task] = {
    # OCR/vision top out at 7B by evidence: the bake-off (see eval/README.md)
    # shows OCR accuracy is saturated across Qwen2.5-VL sizes/quants (3B q8 ~
    # 7B q4 ~ 7B fp16), so bigger/higher-precision buys nothing measurable;
    # 32B also fails to load its CLIP projector on current Ollama. Override
    # with --model qwen2.5vl:32b for genuinely hard layouts if your runtime
    # supports it.
    "ocr": Task(
        name="ocr",
        kind="vision",
        description="Extract text from an image, preserving layout.",
        tiers=[
            ModelTier("qwen2.5vl:3b", 6),
            ModelTier("qwen2.5vl:7b", 16),
        ],
    ),
    "vision": Task(
        name="vision",
        kind="vision",
        description="Describe or answer questions about an image.",
        tiers=[
            ModelTier("qwen2.5vl:3b", 6),
            ModelTier("qwen2.5vl:7b", 16),
        ],
    ),
    # reasoning: math/multi-step word problems. Bake-off (eval/) shows Qwen2.5
    # scales 7b 0.72 -> 14b 1.0 and far outperforms Llama 3.1 8B (0.44) on math.
    # 14b already tops out (32b scored 0.94, no gain), so we cap at 14b; override
    # to qwen2.5:32b for harder problems.
    "reasoning": Task(
        name="reasoning",
        kind="text",
        description="Multi-step reasoning, math, and logic problems.",
        tiers=[
            ModelTier("qwen2.5:3b", 6),
            ModelTier("qwen2.5:7b", 16),
            ModelTier("qwen2.5:14b", 24),
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
    # summary: opposite of reasoning — the bake-off shows Llama 3.1 8B (0.875)
    # beats Qwen2.5 7B (0.75) and ties Qwen2.5 14B on key-fact coverage while
    # being smaller/faster, so it's the pick up to 24 GB; only Qwen2.5 32B (1.0)
    # justifies the jump at 32 GB+.
    "summary": Task(
        name="summary",
        kind="text",
        description="Summarize long text into key points.",
        tiers=[
            ModelTier("qwen2.5:3b", 6),
            ModelTier("llama3.1:8b", 16),
            ModelTier("qwen2.5:32b", 32),
        ],
    ),
    "embed": Task(
        name="embed",
        kind="embed",
        description="Produce vector embeddings for text.",
        # bge-m3 over mxbai-embed-large: mxbai truncates at 512 tokens, which
        # silently drops long inputs; bge-m3 has an 8192-token window, is
        # multilingual, and is competitive on MTEB. nomic is the lightweight
        # fallback (also 8192 ctx) for very low-RAM / CPU-only machines.
        tiers=[
            ModelTier("nomic-embed-text", 2),
            ModelTier("bge-m3", 8),
        ],
    ),
}

_AGENT_TASK = Task(
    name="agent",
    kind="agent",
    description="Agentic assistant: plans and executes Python code in a Docker sandbox.",
    tiers=[
        ModelTier("qwen2.5-coder:3b", 6),
        ModelTier("qwen2.5-coder:7b", 16),
        ModelTier("qwen2.5-coder:14b", 24),
        ModelTier("qwen2.5-coder:32b", 32),
    ],
)

try:
    from sandbox import docker_available
    if docker_available():
        TASKS["agent"] = _AGENT_TASK
except Exception:
    pass


def get_task(name: str) -> Task:
    try:
        return TASKS[name]
    except KeyError:
        raise ValueError(
            f"unknown task '{name}'. available: {', '.join(TASKS)}"
        ) from None
