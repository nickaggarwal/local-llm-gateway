"""Intel NPU backend (scaffold) — Core Ultra NPU via OpenVINO.

Runs text LLMs on the Intel NPU through OpenVINO's ``"NPU"`` device. Generation
goes through optimum-intel (``OVModelForCausalLM``), which handles tokenization
and the decode loop end to end — so on real hardware this path is functional,
not just a stub. Models must first be exported to OpenVINO IR (ideally INT4
weight-compressed) with ``optimum-cli export openvino``.

Gated by `hardware.has_intel_npu()`: on any machine without an Intel NPU it
reports unavailable and the gateway falls back to Ollama. It is **opt-in only**
(`--backend intel-npu`) and never auto-selected — on x86 the Ollama GPU path is
the better default, so we don't silently route to the NPU.

The task -> model map uses Hugging Face ids that get exported to IR; confirm NPU
support per model from the OpenVINO docs, as coverage evolves.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import hardware
from registry import Task

from .base import Backend, BackendUnavailable, OnProgress


@dataclass(frozen=True)
class OvModel:
    hf_model: str  # source id for `optimum-cli export openvino`
    runtime: str = "openvino"
    precompiled: str | None = None  # optional HF repo of ready-made OpenVINO IR to download instead


# Text tasks share one small instruct model (as the Qualcomm backend does).
# ocr/vision/embed are not wired on the NPU yet and fall back to Ollama.
INTEL_MODELS: dict[str, OvModel] = {
    "chat": OvModel("meta-llama/Llama-3.2-3B-Instruct"),
    "summarize": OvModel("meta-llama/Llama-3.2-3B-Instruct"),
    "code": OvModel("Qwen/Qwen2.5-Coder-3B-Instruct"),
}

_CACHE = Path(os.environ.get("OV_NPU_GATEWAY_CACHE", Path.home() / ".cache" / "ov-npu-gateway"))


class IntelNpuBackend(Backend):
    name = "intel-npu"

    def is_available(self) -> bool:
        return hardware.has_intel_npu()

    def supports_task(self, task: Task) -> bool:
        return task.name in INTEL_MODELS

    def resolve_model(self, task: Task, override: str | None, budget_gb: float) -> str:
        if override:
            return override
        spec = INTEL_MODELS.get(task.name)
        if spec is None:
            raise BackendUnavailable(
                f"task '{task.name}' has no Intel NPU model mapped; use --backend ollama instead."
            )
        return spec.hf_model

    def _model_dir(self, model: str) -> Path:
        return _CACHE / model.replace("/", "__")

    def _spec_for(self, model: str) -> OvModel:
        for spec in INTEL_MODELS.values():
            if spec.hf_model == model:
                return spec
        return OvModel(model)  # an override id: export from source, no precompiled repo

    def _require_available(self) -> None:
        if not self.is_available():
            raise BackendUnavailable(
                "no Intel NPU detected. This backend needs an Intel Core Ultra NPU with "
                "OpenVINO exposing the 'NPU' device. Use --backend ollama on this machine."
            )

    def ensure_model(self, model: str, on_progress: OnProgress | None = None) -> None:
        self._require_available()
        model_dir = self._model_dir(model)
        if (model_dir / "openvino_model.xml").exists():
            return
        # Not cached yet: download a pre-compiled IR if mapped, else export from source.
        from . import convert

        convert.prepare_openvino(
            model, model_dir, precompiled=self._spec_for(model).precompiled, on_progress=on_progress
        )

    def generate(self, model: str, prompt: str, image_path: str | None = None) -> str:
        self._require_available()
        if image_path is not None:
            raise BackendUnavailable("the Intel NPU text backend does not accept image input.")
        try:
            from optimum.intel import OVModelForCausalLM
            from transformers import AutoTokenizer
        except ImportError as e:
            raise BackendUnavailable(
                "optimum-intel is required for this backend: pip install optimum[openvino]"
            ) from e
        model_dir = self._model_dir(model)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        ov_model = OVModelForCausalLM.from_pretrained(model_dir, device="NPU")
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        output = ov_model.generate(inputs, max_new_tokens=512, do_sample=False)
        return tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True).strip()

    def embed(self, model: str, text: str) -> list[float]:
        self._require_available()
        raise BackendUnavailable(
            "embeddings are not wired on the Intel NPU backend; use --backend ollama for 'embed'."
        )
