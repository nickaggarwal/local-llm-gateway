"""AMD Ryzen AI (XDNA NPU) backend (scaffold) — ONNX Runtime + VitisAI EP.

Runs quantized ONNX models on the AMD XDNA NPU via ONNX Runtime's
``VitisAIExecutionProvider``. Models come from the Ryzen AI flow — either
quantized locally with ``vai_q_onnx`` or pulled from a prebuilt ``amd/*`` ONNX
repo. As with the Qualcomm ONNX path, the per-model tokenizer + decode loop for
LLMs is model-specific, so this scaffold loads/validates the VitisAI session and
stops short of a generic generation loop (that's the part to wire on hardware).

Gated by `hardware.has_amd_npu()`: on any machine without a Ryzen AI NPU it
reports unavailable and the gateway falls back to Ollama. It is **opt-in only**
(`--backend amd-npu`) and never auto-selected — on x86 the Ollama GPU path
(ROCm) is the better default, so we don't silently route to the NPU.

The slugs below are Ryzen AI model-zoo / ``amd/*`` ONNX repo names; confirm them
per Ryzen AI release, as the catalog evolves.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import hardware
from registry import Task

from .base import Backend, BackendUnavailable, OnProgress


@dataclass(frozen=True)
class XdnaModel:
    onnx_repo: str  # Ryzen AI / amd/* ONNX model id
    runtime: str = "onnx-vitisai"


# Text tasks share one small instruct model. ocr/vision/embed fall back to Ollama.
AMD_MODELS: dict[str, XdnaModel] = {
    "chat": XdnaModel("amd/Llama-3.2-3B-Instruct-onnx-ryzenai"),
    "summarize": XdnaModel("amd/Llama-3.2-3B-Instruct-onnx-ryzenai"),
    "code": XdnaModel("amd/Llama-3.2-3B-Instruct-onnx-ryzenai"),
}

_CACHE = Path(os.environ.get("RYZEN_AI_GATEWAY_CACHE", Path.home() / ".cache" / "ryzen-ai-gateway"))


class AmdNpuBackend(Backend):
    name = "amd-npu"

    def is_available(self) -> bool:
        return hardware.has_amd_npu()

    def supports_task(self, task: Task) -> bool:
        return task.name in AMD_MODELS

    def resolve_model(self, task: Task, override: str | None, budget_gb: float) -> str:
        if override:
            return override
        spec = AMD_MODELS.get(task.name)
        if spec is None:
            raise BackendUnavailable(
                f"task '{task.name}' has no AMD Ryzen AI model mapped; use --backend ollama instead."
            )
        return spec.onnx_repo

    def _model_dir(self, model: str) -> Path:
        return _CACHE / model.replace("/", "__")

    def _require_available(self) -> None:
        if not self.is_available():
            raise BackendUnavailable(
                "no AMD XDNA NPU detected. This backend needs a Ryzen AI device with the "
                "Ryzen AI SW stack and ONNX Runtime's VitisAI EP. Use --backend ollama here."
            )

    def ensure_model(self, model: str, on_progress: OnProgress | None = None) -> None:
        self._require_available()
        model_dir = self._model_dir(model)
        if model_dir.exists() and any(model_dir.glob("*.onnx")):
            return
        # Not cached yet: download the pre-quantized Ryzen AI ONNX repo (amd/*).
        from . import convert

        convert.prepare_vitisai(model, model_dir, on_progress=on_progress)

    def generate(self, model: str, prompt: str, image_path: str | None = None) -> str:
        self._require_available()
        if image_path is not None:
            raise BackendUnavailable("the AMD XDNA text backend does not accept image input.")
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise BackendUnavailable(
                "onnxruntime with the VitisAI EP is required (install the Ryzen AI SW stack)."
            ) from e
        model_dir = self._model_dir(model)
        onnx_path = next(model_dir.glob("*.onnx"), None) if model_dir.exists() else None
        if onnx_path is None:
            raise BackendUnavailable(
                f"no .onnx graph for '{model}' in {model_dir}; run the Ryzen AI export first."
            )
        if "VitisAIExecutionProvider" not in ort.get_available_providers():
            raise BackendUnavailable(
                "VitisAIExecutionProvider is not available in this onnxruntime build; "
                "install the Ryzen AI onnxruntime."
            )
        config = os.environ.get("VAIP_CONFIG", "vaip_config.json")
        ort.InferenceSession(
            str(onnx_path),
            providers=["VitisAIExecutionProvider"],
            provider_options=[{"config_file": config}],
        )
        raise BackendUnavailable(
            f"VitisAI session for '{model}' loads, but the per-model tokenizer + decode loop "
            f"still needs wiring (Ryzen AI LLMs use a model-specific generation runner)."
        )

    def embed(self, model: str, text: str) -> list[float]:
        self._require_available()
        raise BackendUnavailable(
            "embeddings are not wired on the AMD XDNA backend; use --backend ollama for 'embed'."
        )
