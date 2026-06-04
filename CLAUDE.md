# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local LLM gateway: pick the best local model per **task** (`ocr`, `chat`, `code`,
`summarize`, `vision`, `embed`), download/prepare it on first use, and run it entirely on
the user's machine. Three surfaces share one core: a CLI (`cli.py`), a FastAPI server
(`server.py`), and a Streamlit UI (`app.py`).

## Commands

```bash
# One-command native launch (host Ollama + venv + UI). No flags.
./start.sh                         # macOS/Linux/WSL/Git Bash
.\start.ps1                        # Windows PowerShell

# CLI
python cli.py tasks                       # list tasks + detected hardware + chosen model/backend
python cli.py tasks --backend qualcomm    # what a specific backend would pick
python cli.py run chat "..."              # run a text task
python cli.py run ocr --image x.png       # vision task
python cli.py run chat "..." --backend intel-npu --model <id>   # force backend/model

# HTTP API
uvicorn server:app --port 8000

# Streamlit UI
streamlit run app.py

# Prepare NPU models ahead of first use (also auto-runs from the launcher)
python convert.py chat                 # detected NPU
python convert.py --all --backend amd-npu

# Eval / bake-off (drives Ollama directly)
python eval/run_eval.py --task all
python eval/run_eval.py --task ocr           # single task
python eval/run_eval.py --task chat --pull   # download missing models first
python eval/run_eval.py --task all --limit 2 # quick smoke test
```

There is no build step, linter config, or unit-test suite. "Tests" means the eval
bake-off in `eval/`, which requires a running Ollama and downloads models.

## Architecture

The flow is **task → model → backend**, with two independent selection decisions:

1. **Model selection** lives in `registry.py` (`TASKS`). Each task has RAM-tiered Ollama
   model candidates; `Task.pick_model(budget_gb)` returns the largest that fits the budget.
2. **Backend selection** lives in `backends/` (`get_backend(name, task)`). `auto` falls
   back to Ollama unless a Qualcomm NPU is present; the other NPU backends are opt-in.

`gateway.run()` is the single orchestrator both the CLI/API/UI call: it resolves the
backend, asks the backend to resolve the model (using the memory budget), calls
`backend.ensure_model()`, then dispatches to `generate`/`embed`. `gateway.py` deliberately
keeps a stable public surface (`run`, `available_ram_gb`, `OCR_PROMPT`) — the three front
ends depend on it.

### Backends (`backends/`)

All implement the `Backend` ABC in `backends/base.py` (`is_available`, `resolve_model`,
`ensure_model`, `generate`, `embed`, `supports_task`). Failures raise `BackendUnavailable`
with an actionable message that the front ends surface verbatim.

- **`ollama.py`** — the default; talks to the Ollama HTTP API at `OLLAMA_HOST`
  (default `localhost:11434`). NVIDIA (CUDA) and AMD (ROCm) acceleration are automatic in
  stock Ollama; Intel Arc/Xe goes through an IPEX-LLM Ollama build (same API/port).
- **`qualcomm.py`** (Hexagon NPU), **`intel_npu.py`** (OpenVINO `device="NPU"`),
  **`amd_npu.py`** (ONNX Runtime VitisAI EP) — each has its own per-task model map
  (`QUALCOMM_MODELS` / `INTEL_MODELS` / `AMD_MODELS`) and its own cache dir env var. These
  are **hardware-gated**: on a machine without that NPU they report unavailable and the
  gateway falls back to Ollama.

**Auto-selection rule (important):** only Qualcomm is auto-preferred when present (Snapdragon
has no Ollama GPU path). `intel-npu` and `amd-npu` are **opt-in only** (`--backend …`),
because on x86 the Ollama GPU path is the better default — don't change this without reason.

### Hardware detection (`hardware.py`)

Detects RAM, GPUs (NVIDIA via NVML/`nvidia-smi`, AMD/Intel via DRM sysfs `/sys/class/drm`),
and NPUs (Qualcomm/Intel/AMD). `memory_budget_gb()` returns **`RAM + dedicated VRAM`** —
this is additive because llama.cpp/Ollama offload layers across GPU and CPU, so a GPU only
ever lets you run a bigger model. Integrated GPUs report 0 VRAM and are excluded to avoid
double-counting shared memory.

### NPU model preparation (`backends/convert.py` + `convert.py`)

NPU models aren't plain Ollama pulls — they need device-specific artifacts. When a backend's
cache is empty, `ensure_model` calls `backends/convert.py` to **download a pre-compiled
equivalent or run the vendor conversion pipeline** (optimum-cli for Intel, qai-hub export
for Qualcomm, HF download for AMD) into that cache. `convert.py` (root) is the standalone
CLI; the launcher runs `convert.py --auto` at startup (a clean no-op when no NPU is present).

### Launcher GPU routing

`start.sh` / `start.ps1` run native only (no Docker, no flags). When **Intel is the only
GPU** (no NVIDIA/AMD for stock Ollama to use), they launch an IPEX-LLM Ollama build instead
of stock Ollama (set `IPEX_LLM_OLLAMA=/path/to/ollama`) with the SYCL env exported.

## Conventions

- The NPU backends are honest scaffolds: the Intel text path runs end-to-end on hardware,
  but the Qualcomm ONNX and AMD paths load/validate the session and stop short of the
  model-specific tokenizer + decode loop, and `embed`/vision aren't mapped on any NPU
  (they fall back to Ollama). None of the NPU paths can be exercised on a Mac/x86 host
  without the vendor NPU + SDK — keep this in mind; don't claim them verified.
- To change a model per task, edit the map for that backend (Ollama: `registry.py`; NPUs:
  the `*_MODELS` dict in the backend file), or override per call with `--model`.
- Optional/platform deps (pynvml, optimum[openvino], onnxruntime-qnn, qai-hub-models, Ryzen
  AI) are intentionally **not** in `requirements.txt` so the base install stays clean on
  Mac/x86; they're documented as comments there and imported lazily inside the NPU paths.
