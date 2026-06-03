"""Model preparation pipelines for the NPU backends.

When an NPU model isn't cached yet, prepare it before first use: prefer
downloading a **pre-compiled equivalent**, otherwise run the **vendor conversion
pipeline** into the backend's cache dir. Each pipeline shells out to (or imports)
the vendor tool, so it needs that SDK on the target device; when the tool is
missing it raises `BackendUnavailable` with the manual command to run.

These pipelines target real hardware + SDKs and can't run on a Mac/x86 host
without the NPU toolchain installed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .base import BackendUnavailable, OnProgress


def _emit(on_progress: OnProgress | None, msg: str) -> None:
    if on_progress:
        on_progress(msg)


def _run(cmd: list[str], on_progress: OnProgress | None = None) -> None:
    """Run a conversion command, streaming its output through on_progress."""
    _emit(on_progress, "$ " + " ".join(cmd))
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
    except OSError as e:
        raise BackendUnavailable(f"failed to launch '{cmd[0]}': {e}") from e
    assert proc.stdout is not None
    for line in proc.stdout:
        _emit(on_progress, line.rstrip())
    if proc.wait() != 0:
        raise BackendUnavailable(f"conversion failed (exit {proc.returncode}): {' '.join(cmd)}")


def _download_hf(
    repo: str, out_dir: Path, on_progress: OnProgress | None = None, allow_patterns=None
) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise BackendUnavailable(
            "huggingface_hub is required to fetch a pre-compiled model: pip install huggingface_hub"
        ) from e
    _emit(on_progress, f"downloading pre-compiled model {repo} ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo, local_dir=str(out_dir), allow_patterns=allow_patterns)


def prepare_openvino(
    model: str, out_dir: Path, *, precompiled: str | None = None, on_progress: OnProgress | None = None
) -> None:
    """Intel: download a pre-compiled OpenVINO IR repo, else export from source via optimum-cli."""
    if precompiled:
        _download_hf(
            precompiled, out_dir, on_progress,
            allow_patterns=["*.xml", "*.bin", "*.json", "*.txt", "tokenizer*", "*.model"],
        )
        return
    cli = shutil.which("optimum-cli")
    if not cli:
        raise BackendUnavailable(
            "optimum-cli not found — pip install optimum[openvino] (or map a pre-compiled "
            "OpenVINO repo via OvModel.precompiled)."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [cli, "export", "openvino", "--model", model, "--weight-format", "int4", str(out_dir)],
        on_progress,
    )


def prepare_vitisai(model: str, out_dir: Path, *, on_progress: OnProgress | None = None) -> None:
    """AMD: download a pre-compiled Ryzen AI ONNX repo (amd/*), else require local quantization."""
    if "/" in model:  # looks like a Hugging Face repo of pre-quantized ONNX
        _download_hf(
            model, out_dir, on_progress,
            allow_patterns=["*.onnx", "*.onnx.data", "*.json", "*config*", "tokenizer*", "*.model"],
        )
        return
    raise BackendUnavailable(
        f"'{model}' is not a downloadable Ryzen AI ONNX repo. Quantize locally with the "
        f"Ryzen AI flow (vai_q_onnx) and place the *.onnx + VitisAI config in {out_dir}."
    )


def prepare_qnn(
    model: str, out_dir: Path, *, device: str | None = None, on_progress: OnProgress | None = None
) -> None:
    """Qualcomm: run the AI Hub export/compile pipeline for the target device."""
    try:
        import qai_hub_models  # noqa: F401
    except ImportError as e:
        raise BackendUnavailable(
            "qai-hub-models not installed — pip install qai-hub-models (and configure an "
            "AI Hub API token)."
        ) from e
    device = device or os.environ.get("QAI_HUB_DEVICE", "Snapdragon X Elite CRD")
    out_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [sys.executable, "-m", f"qai_hub_models.models.{model}.export",
         "--device", device, "--output-dir", str(out_dir)],
        on_progress,
    )


def prepare(
    backend: str,
    model: str,
    out_dir: Path,
    *,
    device: str | None = None,
    precompiled: str | None = None,
    on_progress: OnProgress | None = None,
) -> None:
    """Dispatch to the right conversion pipeline for `backend`."""
    if backend == "intel-npu":
        prepare_openvino(model, out_dir, precompiled=precompiled, on_progress=on_progress)
    elif backend == "amd-npu":
        prepare_vitisai(model, out_dir, on_progress=on_progress)
    elif backend == "qualcomm":
        prepare_qnn(model, out_dir, device=device, on_progress=on_progress)
    else:
        raise BackendUnavailable(f"no conversion pipeline for backend '{backend}'")
