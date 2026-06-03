"""Qualcomm AI Hub / Hexagon NPU backend (scaffold).

Models on Qualcomm AI Hub (https://aihub.qualcomm.com) are pre-optimized and
compiled for the Snapdragon Hexagon NPU. They do **not** run through Ollama:
text LLMs run on the Genie runtime (`genie-t2t-run`), while CNN/transformer
graphs run on ONNX Runtime's QNN execution provider. Both require Snapdragon
hardware plus the QNN SDK, so on any other machine this backend reports
unavailable and is never auto-selected — `gateway` falls back to Ollama.

What is real here:
  * hardware/SDK detection (`is_available`) gating every code path;
  * the task -> AI Hub model map and the per-model runtime choice;
  * the Genie CLI invocation and the ONNX Runtime QNN session setup.

What requires a Snapdragon device to complete:
  * `ensure_model` expects the compiled artifact + Genie config produced by the
    AI Hub export step (instructions are raised if it's missing — we don't fake
    a cloud compile job here);
  * actually executing the NPU graph.

The AI Hub model slugs below are the `qai_hub_models` module names; adjust per
your target device from the AI Hub catalog.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import hardware
from registry import Task

from .base import Backend, BackendUnavailable, OnProgress


@dataclass(frozen=True)
class QnnModel:
    aihub_slug: str  # qai_hub_models module name, e.g. "llama_v3_2_3b_chat_quantized"
    runtime: str  # "genie" for text LLMs, "onnx-qnn" for ONNX graphs


# Task -> NPU-optimized model. Slugs come from the AI Hub catalog; verify the
# exact name/device target before exporting, as the catalog evolves.
QUALCOMM_MODELS: dict[str, QnnModel] = {
    "chat": QnnModel("llama_v3_2_3b_chat_quantized", "genie"),
    "summarize": QnnModel("llama_v3_2_3b_chat_quantized", "genie"),
    "code": QnnModel("llama_v3_2_3b_chat_quantized", "genie"),
    "ocr": QnnModel("trocr", "onnx-qnn"),
    "vision": QnnModel("openai_clip", "onnx-qnn"),
    # No NPU-optimized embedding model is wired yet; `embed` stays on Ollama.
}

_CACHE = Path(os.environ.get("QAI_HUB_GATEWAY_CACHE", Path.home() / ".cache" / "qai-hub-gateway"))


class QualcommBackend(Backend):
    name = "qualcomm"

    def is_available(self) -> bool:
        return hardware.has_qualcomm_npu()

    def supports_task(self, task: Task) -> bool:
        return task.name in QUALCOMM_MODELS

    def resolve_model(self, task: Task, override: str | None, budget_gb: float) -> str:
        if override:
            return override
        spec = QUALCOMM_MODELS.get(task.name)
        if spec is None:
            raise BackendUnavailable(
                f"task '{task.name}' has no Qualcomm AI Hub model mapped; "
                f"run it on the Ollama backend (--backend ollama) instead."
            )
        return spec.aihub_slug

    def _spec(self, model: str) -> QnnModel:
        for spec in QUALCOMM_MODELS.values():
            if spec.aihub_slug == model:
                return spec
        # Allow an arbitrary override slug; default it to the Genie runtime.
        return QnnModel(model, "genie")

    def _require_available(self) -> None:
        if not self.is_available():
            raise BackendUnavailable(
                "no Qualcomm Hexagon NPU detected. This backend needs a Snapdragon "
                "device with the QNN SDK (set QNN_SDK_ROOT) or ONNX Runtime's "
                "QNNExecutionProvider. Use --backend ollama on this machine."
            )

    def ensure_model(self, model: str, on_progress: OnProgress | None = None) -> None:
        self._require_available()
        model_dir = _CACHE / model
        if model_dir.exists() and any(model_dir.iterdir()):
            return
        # We deliberately do not run a cloud compile job here (it needs an AI Hub
        # API token and minutes of NPU compile time). Point the user at the
        # documented export step that produces the artifact + Genie config.
        raise BackendUnavailable(
            f"compiled artifact for '{model}' not found in {model_dir}. Export it "
            f"for your device first, e.g.:\n"
            f"  pip install qai-hub-models\n"
            f'  python -m qai_hub_models.models.{model}.export --device "Snapdragon X Elite CRD"\n'
            f"then place the produced .bin/.onnx and genie_config.json under {model_dir}."
        )

    def generate(self, model: str, prompt: str, image_path: str | None = None) -> str:
        self._require_available()
        spec = self._spec(model)
        if spec.runtime == "genie":
            if image_path is not None:
                raise BackendUnavailable(
                    "the Genie text runtime does not accept image input; "
                    "map this task to an onnx-qnn vision model instead."
                )
            return self._run_genie(model, prompt)
        if spec.runtime == "onnx-qnn":
            return self._run_onnx(model, prompt, image_path)
        raise BackendUnavailable(f"unknown QNN runtime '{spec.runtime}' for model '{model}'")

    def embed(self, model: str, text: str) -> list[float]:
        self._require_available()
        raise BackendUnavailable(
            "embeddings are not yet wired on the Qualcomm backend; "
            "use --backend ollama for the 'embed' task."
        )

    def _run_genie(self, model: str, prompt: str) -> str:
        genie = shutil.which("genie-t2t-run")
        if not genie:
            raise BackendUnavailable(
                "genie-t2t-run not found on PATH. Install the QNN/Genie runtime "
                "from the Qualcomm AI Engine Direct SDK and add it to PATH."
            )
        config = _CACHE / model / "genie_config.json"
        if not config.exists():
            raise BackendUnavailable(
                f"missing Genie config at {config}; run the AI Hub export for '{model}' first."
            )
        out = subprocess.run(
            [genie, "-c", str(config), "-p", prompt],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if out.returncode != 0:
            raise BackendUnavailable(f"genie-t2t-run failed: {out.stderr.strip() or out.stdout.strip()}")
        return out.stdout.strip()

    def _run_onnx(self, model: str, prompt: str, image_path: str | None) -> str:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise BackendUnavailable(
                "onnxruntime with the QNN execution provider is required for this "
                "model. Install onnxruntime-qnn on your Snapdragon device."
            ) from e
        onnx_path = next((_CACHE / model).glob("*.onnx"), None) if (_CACHE / model).exists() else None
        if onnx_path is None:
            raise BackendUnavailable(
                f"no .onnx graph for '{model}' in {_CACHE / model}; run the AI Hub export first."
            )
        if "QNNExecutionProvider" not in ort.get_available_providers():
            raise BackendUnavailable(
                "QNNExecutionProvider is not available in this onnxruntime build; "
                "install onnxruntime-qnn on a Snapdragon device."
            )
        backend_path = os.environ.get("QNN_BACKEND_PATH", "QnnHtp.so")
        ort.InferenceSession(
            str(onnx_path),
            providers=["QNNExecutionProvider"],
            provider_options=[{"backend_path": backend_path}],
        )
        # Model-specific tokenization / image pre- & post-processing is required
        # to turn the raw graph I/O into text and is not generic across models.
        raise BackendUnavailable(
            f"QNN session for '{model}' loads, but per-model pre/post-processing "
            f"(tokenizer or image transform + decoding) still needs wiring for this graph."
        )
