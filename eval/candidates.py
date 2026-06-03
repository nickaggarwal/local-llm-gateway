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


# (kind, ladder). Ladders mix sizes and quants (q4 default tag, q8_0, fp16).
LADDERS: dict[str, tuple[str, list[Cand]]] = {
    "ocr": ("vision", [
        Cand("qwen2.5vl:3b-q8_0", 3.5),
        Cand("qwen2.5vl:7b", 6.0),         # q4_K_M
        Cand("qwen2.5vl:7b-q8_0", 9.4),
        Cand("qwen2.5vl:7b-fp16", 17.0),
        Cand("qwen2.5vl:32b", 21.0),       # q4_K_M
    ]),
    "vision": ("vision", [
        Cand("qwen2.5vl:3b-q8_0", 3.5),
        Cand("qwen2.5vl:7b", 6.0),
        Cand("qwen2.5vl:7b-q8_0", 9.4),
        Cand("qwen2.5vl:7b-fp16", 17.0),
        Cand("qwen2.5vl:32b", 21.0),
    ]),
    "chat": ("text", [
        Cand("qwen2.5:3b", 2.0),
        Cand("qwen2.5:7b", 4.7),           # q4_K_M
        Cand("qwen2.5:7b-fp16", 15.0),
        Cand("qwen2.5:14b", 9.0),          # q4_K_M
        Cand("qwen2.5:32b", 20.0),         # q4_K_M
    ]),
    "summarize": ("text", [
        Cand("qwen2.5:3b", 2.0),
        Cand("qwen2.5:7b", 4.7),
        Cand("qwen2.5:14b", 9.0),
        Cand("qwen2.5:32b", 20.0),
    ]),
    "code": ("text", [
        Cand("qwen2.5-coder:3b", 1.9),
        Cand("qwen2.5-coder:7b", 4.7),     # q4_K_M
        Cand("qwen2.5-coder:7b-fp16", 15.0),
        Cand("qwen2.5-coder:14b", 9.0),    # q4_K_M
        Cand("qwen2.5-coder:32b", 20.0),   # q4_K_M
    ]),
    "embed": ("embed", [
        Cand("nomic-embed-text", 0.27),
        Cand("mxbai-embed-large", 0.67),
    ]),
}
