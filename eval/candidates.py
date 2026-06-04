"""Model ladders per task, with footprints and RAM-fit logic.

Each task has a ladder spanning the size/quantization spectrum so the eval can
answer: at a given RAM budget, is a *smaller full-precision* model or a
*larger quantized* model more accurate?

`weights_gb` is the on-disk size; runtime RAM also needs KV-cache, activations,
and (for vision) image tokens, approximated by `runtime_gb`. A model "fits" a
laptop when its runtime need leaves ~5 GB headroom for the OS and apps.
"""

from dataclasses import dataclass

TIERS = [16, 24, 32, 48]
OS_HEADROOM_GB = 5


@dataclass(frozen=True)
class Cand:
    model: str
    weights_gb: float


def runtime_gb(weights_gb: float, kind: str) -> float:
    overhead = 2.5 if kind == "vision" else 1.5
    return round(weights_gb * 1.2 + overhead, 1)


def fits(weights_gb: float, kind: str, total_ram_gb: int) -> bool:
    return runtime_gb(weights_gb, kind) <= total_ram_gb - OS_HEADROOM_GB


# (kind, ladder). Ladders mix three axes so the bake-off can compare them:
#   1. size      (3B -> 32B)
#   2. quant      (q4_K_M default tag, q8_0, fp16)
#   3. family     (Qwen vs cross-family alternatives surfaced by 2026 research)
# Cross-family entries are marked; they're skipped unless downloaded (or --pull).
LADDERS: dict[str, tuple[str, list[Cand]]] = {
    "ocr": ("vision", [
        Cand("qwen2.5vl:3b-q8_0", 3.5),
        Cand("qwen2.5vl:7b", 6.0),         # q4_K_M
        Cand("qwen2.5vl:7b-q8_0", 9.4),
        Cand("qwen2.5vl:7b-fp16", 17.0),
        Cand("qwen2.5vl:32b", 21.0),       # q4_K_M
        Cand("minicpm-v", 5.5),            # alt family: strong OCR VLM
    ]),
    "vision": ("vision", [
        Cand("qwen2.5vl:3b-q8_0", 3.5),
        Cand("qwen2.5vl:7b", 6.0),
        Cand("qwen2.5vl:7b-q8_0", 9.4),
        Cand("qwen2.5vl:7b-fp16", 17.0),
        Cand("qwen2.5vl:32b", 21.0),
        Cand("minicpm-v", 5.5),            # alt family
        Cand("llama3.2-vision:11b", 7.8),  # alt family
    ]),
    "reasoning": ("text", [
        Cand("qwen2.5:3b", 2.0),
        Cand("qwen2.5:7b", 4.7),           # q4_K_M
        Cand("qwen2.5:7b-fp16", 15.0),
        Cand("qwen2.5:14b", 9.0),          # q4_K_M
        Cand("qwen2.5:32b", 20.0),         # q4_K_M
        Cand("llama3.1:8b", 4.9),          # alt family
        Cand("gemma2:9b", 5.4),            # alt family
        Cand("mistral:7b", 4.1),           # alt family: fastest at 7B
        Cand("phi4:14b", 9.1),             # alt family: STEM/reasoning
    ]),
    "summary": ("text", [
        Cand("qwen2.5:3b", 2.0),
        Cand("qwen2.5:7b", 4.7),
        Cand("qwen2.5:14b", 9.0),
        Cand("qwen2.5:32b", 20.0),
        Cand("llama3.1:8b", 4.9),          # alt family: leads summary at ~8B
        Cand("gemma2:9b", 5.4),            # alt family
    ]),
    "code": ("text", [
        Cand("qwen2.5-coder:3b", 1.9),
        Cand("qwen2.5-coder:7b", 4.7),     # q4_K_M
        Cand("qwen2.5-coder:7b-fp16", 15.0),
        Cand("qwen2.5-coder:14b", 9.0),    # q4_K_M
        Cand("qwen2.5-coder:32b", 20.0),   # q4_K_M
        Cand("deepseek-coder-v2:16b", 8.9),  # alt family: fast MoE
        Cand("codestral:22b", 13.0),         # alt family: best fill-in-the-middle
    ]),
    "embed": ("embed", [
        Cand("nomic-embed-text", 0.27),    # 8192 ctx, lightest
        Cand("bge-m3", 1.2),               # 8192 ctx, multilingual, best quality/cost
        Cand("mxbai-embed-large", 0.67),   # high English retrieval BUT 512-token ctx limit
    ]),
}
