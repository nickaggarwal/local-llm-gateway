"""Hardware detection: system RAM, GPU VRAM (NVIDIA/AMD/Intel), and Qualcomm NPU.

Two jobs:

1. **Size the model tier to the real compute budget.** Ollama/llama.cpp splits a
   model across GPU layers (VRAM) and CPU layers (system RAM) via partial offload,
   so the binding constraint is the *combined* pool, not VRAM alone. We size off
   `RAM + VRAM`: a GPU only ever *adds* capacity (and accelerates the layers that
   fit in VRAM) — it can never shrink the model you'd have run on CPU. OS headroom
   is already baked into the registry's RAM thresholds. Only *dedicated* VRAM is
   added; integrated GPUs share system RAM, so counting them would double-count.
2. **Decide which inference backend is usable here.** Ollama is always the
   fallback; the Qualcomm/QNN backend is only selectable when a Hexagon NPU is
   actually present (see `has_qualcomm_npu`).

GPU *acceleration* itself is handled by the inference runtime, not here: Ollama
uses NVIDIA (CUDA) and AMD (ROCm) automatically; Intel Arc goes through the
IPEX-LLM Ollama build (point `OLLAMA_HOST` at it). This module only detects the
hardware so the model tier can be sized for it.
"""

from __future__ import annotations

import functools
import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


def total_ram_gb() -> float:
    """Best-effort total system RAM in GB."""
    try:
        import psutil

        return psutil.virtual_memory().total / 1024**3
    except Exception:  # noqa: BLE001
        # POSIX fallback without psutil. os.sysconf is absent on Windows.
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
        except (ValueError, OSError, AttributeError):
            return 8.0


@dataclass(frozen=True)
class Gpu:
    vendor: str  # "nvidia" | "amd" | "intel"
    vram_gb: float  # 0.0 when present but dedicated VRAM is unknown (e.g. integrated)
    name: str | None = None


def _detect_nvidia() -> list[Gpu]:
    """NVIDIA GPUs via NVML, falling back to nvidia-smi."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            gpus = []
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                total = pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1024**3
                name = pynvml.nvmlDeviceGetName(handle)
                gpus.append(Gpu("nvidia", total, name.decode() if isinstance(name, bytes) else name))
            return gpus
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001
        pass

    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    try:
        out = subprocess.run(
            [smi, "--query-gpu=memory.total,name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    gpus = []
    for line in out.stdout.splitlines():
        if not line.strip():
            continue
        mem, _, name = line.partition(",")
        try:
            gpus.append(Gpu("nvidia", float(mem) / 1024, name.strip() or None))  # MiB -> GiB
        except ValueError:
            continue
    return gpus


def _detect_sysfs(vendor_id: str, vendor: str) -> list[Gpu]:
    """Linux DRM sysfs probe: match PCI vendor id, read dedicated VRAM if exposed.

    `mem_info_vram_total` (bytes) is exposed by the amdgpu driver and Intel's xe
    driver for discrete cards; integrated GPUs lack it, so they report 0.0 VRAM
    (detected, but not added to the sizing budget).
    """
    gpus = []
    for dev in sorted(Path("/sys/class/drm").glob("card*/device")):
        try:
            if (dev / "vendor").read_text().strip().lower() != vendor_id:
                continue
        except OSError:
            continue
        vram = 0.0
        try:
            vram = int((dev / "mem_info_vram_total").read_text().strip()) / 1024**3
        except (OSError, ValueError):
            pass
        gpus.append(Gpu(vendor, vram))
    return gpus


def _detect_amd() -> list[Gpu]:
    """AMD GPUs via DRM sysfs, falling back to rocm-smi."""
    gpus = _detect_sysfs("0x1002", "amd")
    if any(g.vram_gb > 0 for g in gpus):
        return gpus
    smi = shutil.which("rocm-smi")
    if smi:
        try:
            out = subprocess.run(
                [smi, "--showmeminfo", "vram", "--json"], capture_output=True, text=True, timeout=5
            )
            if out.returncode == 0 and out.stdout.strip():
                data = json.loads(out.stdout)
                rocm = []
                for fields in data.values():
                    for key, val in fields.items():
                        if "vram" in key.lower() and "total" in key.lower():
                            try:
                                rocm.append(Gpu("amd", int(val) / 1024**3))
                            except (ValueError, TypeError):
                                continue
                if rocm:
                    return rocm
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            pass
    return gpus


def _detect_intel() -> list[Gpu]:
    """Intel GPUs via DRM sysfs (PCI vendor 0x8086)."""
    return _detect_sysfs("0x8086", "intel")


@functools.lru_cache(maxsize=1)
def detect_gpus() -> tuple[Gpu, ...]:
    """All detected GPUs across vendors. Empty on machines without one (e.g. this Mac)."""
    return tuple(_detect_nvidia() + _detect_amd() + _detect_intel())


def gpu_vram_gb() -> float | None:
    """Dedicated VRAM (GB) of the largest GPU, or None if none has known VRAM.

    This is the figure added to the RAM sizing budget, so integrated GPUs that
    share system RAM (reported as 0.0) are intentionally excluded.
    """
    vrams = [g.vram_gb for g in detect_gpus() if g.vram_gb > 0]
    return max(vrams) if vrams else None


def has_gpu() -> bool:
    return bool(detect_gpus())


@functools.lru_cache(maxsize=1)
def has_qualcomm_npu() -> bool:
    """True if a Qualcomm Hexagon NPU is usable for inference on this machine.

    Detection is layered (any one is sufficient): an installed QNN SDK
    (``QNN_SDK_ROOT``), ONNX Runtime exposing the QNN execution provider, or a
    Snapdragon Windows-on-ARM CPU. On non-Snapdragon hosts this returns False,
    so the Qualcomm backend is never auto-selected there.
    """
    if os.environ.get("QNN_SDK_ROOT") or os.environ.get("QNN_BACKEND_PATH"):
        return True
    try:
        import onnxruntime as ort

        if "QNNExecutionProvider" in ort.get_available_providers():
            return True
    except Exception:  # noqa: BLE001
        pass
    proc = (platform.processor() or "").lower()
    machine = (platform.machine() or "").lower()
    if platform.system() == "Windows" and (
        "arm" in machine or "snapdragon" in proc or "qualcomm" in proc
    ):
        return True
    return False


def _accel_vendors() -> set[str]:
    """Vendors of devices on the Linux DRM *accel* subsystem (NPUs live here).

    Both Intel's `intel_vpu` and AMD's `amdxdna` drivers register under
    `/sys/class/accel`, so we read each device's PCI vendor id to tell them
    apart (AMD `0x1002`, Intel `0x8086`). Empty off Linux / without an NPU.
    """
    ids = {"0x1002": "amd", "0x8086": "intel"}
    vendors: set[str] = set()
    for dev in Path("/sys/class/accel").glob("accel*/device"):
        try:
            vid = (dev / "vendor").read_text().strip().lower()
        except OSError:
            continue
        if vid in ids:
            vendors.add(ids[vid])
    return vendors


@functools.lru_cache(maxsize=1)
def has_intel_npu() -> bool:
    """True if an Intel NPU (Core Ultra) is usable via OpenVINO's 'NPU' device.

    Off-Intel-NPU hosts return False, so the intel-npu backend stays dormant.
    """
    try:
        import openvino

        if "NPU" in openvino.Core().available_devices:
            return True
    except Exception:  # noqa: BLE001
        pass
    return "intel" in _accel_vendors()


@functools.lru_cache(maxsize=1)
def has_amd_npu() -> bool:
    """True if an AMD XDNA NPU (Ryzen AI) is usable via ONNX Runtime's VitisAI EP.

    Layered detection: Ryzen AI env vars, the VitisAI execution provider, or an
    AMD device on the DRM accel subsystem. False elsewhere.
    """
    if os.environ.get("XLNX_VART_FIRMWARE") or os.environ.get("RYZEN_AI_INSTALLATION_PATH"):
        return True
    try:
        import onnxruntime as ort

        if "VitisAIExecutionProvider" in ort.get_available_providers():
            return True
    except Exception:  # noqa: BLE001
        pass
    return "amd" in _accel_vendors()


def memory_budget_gb() -> float:
    """Combined memory budget for tier sizing: system RAM plus any GPU VRAM.

    Reflects how llama.cpp/Ollama offload layers across GPU and CPU — the model
    can use both pools, so VRAM is additive on top of RAM rather than a
    replacement for it.
    """
    return total_ram_gb() + (gpu_vram_gb() or 0.0)


def describe() -> dict:
    """Compact snapshot of the compute environment for display/reporting."""
    gpus = detect_gpus()
    vram = gpu_vram_gb()
    return {
        "ram_gb": round(total_ram_gb(), 1),
        "gpu": bool(gpus),
        "gpus": [
            {"vendor": g.vendor, "name": g.name, "vram_gb": round(g.vram_gb, 1) if g.vram_gb else None}
            for g in gpus
        ],
        "vram_gb": round(vram, 1) if vram else None,
        "npus": {
            "qualcomm": has_qualcomm_npu(),
            "intel": has_intel_npu(),
            "amd": has_amd_npu(),
        },
        "budget_gb": round(memory_budget_gb(), 1),
    }
