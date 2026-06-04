# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local LLM gateway: pick the best local model per **task** (`ocr`, `reasoning`, `code`,
`summary`, `vision`, `embed`), download/prepare it on first use, and run it entirely on
the user's machine — no data leaves the box. Three front ends share one core:

- **CLI** — `cli.py`
- **HTTP API** (FastAPI) — `server.py`
- **Web UI** (Streamlit) — `app.py`

Inference goes through a pluggable **backend** (Ollama by default; Qualcomm/Intel/AMD NPU
backends are present but hardware-gated). An evaluation harness in `eval/` ("the tests")
runs model bake-offs to justify the model choices in `registry.py`.

## Repository layout

```
gateway.py        Orchestrator: run() ties task -> backend -> model. Stable public surface.
registry.py       TASKS: per-task RAM-tiered Ollama model candidates + pick_model().
hardware.py       Detect RAM / GPU VRAM / NPU; memory_budget_gb() = RAM + dedicated VRAM.
convert.py        CLI to pre-prepare NPU models (wraps backends/convert.py).

backends/
  base.py         Backend ABC + BackendUnavailable.
  ollama.py       Default backend (HTTP API). Sets num_ctx; NVIDIA/AMD auto, Intel via IPEX.
  qualcomm.py     Hexagon NPU (Genie/QNN). QUALCOMM_MODELS map. Hardware-gated.
  intel_npu.py    OpenVINO device="NPU". INTEL_MODELS map. Hardware-gated.
  amd_npu.py      ONNX Runtime VitisAI EP (Ryzen AI). AMD_MODELS map. Hardware-gated.
  convert.py      Per-backend NPU model preparation (download/compile artifacts).

agent.py          Agentic tool-use loop (Ollama /api/chat + run_python/install_package).
executors/        Pluggable sandboxes for agent code (Docker-free by default):
  base.py         Executor ABC + shared workspace + new-file diffing.
  docker_exec.py  Docker container per run (strongest isolation, if a daemon is present).
  wasm.py         Pyodide (CPython→WASM) under Node; pyodide_driver.mjs is the Node host.
  subprocess_exec.py  Hardened venv subprocess (always works; blast-radius isolation).
sandbox.py        Back-compat shim re-exporting the moved Docker sandbox.
Dockerfile.sandbox  Image for the Docker executor (optional; not required anymore).

cli.py server.py app.py    The three front ends (all call gateway.run()).
start.sh start.ps1         Native launchers (host Ollama + venv + UI; no flags, no Docker).

eval/
  candidates.py   LADDERS: per-task model ladder (sizes x quants x families) + RAM-fit math.
  scorers.py      One objective scorer per task kind (returns 0..1).
  run_eval.py     The bake-off: runs each downloaded candidate over a dataset, per-tier winner.
  gen_images.py   Generates the OCR test images + ocr.json (deterministic).
  fetch_vqa.py    Builds vision.json from public VQAv2 via the HF datasets-server API.
  datasets/*.json Ground-truth datasets, one per task.
  images/         OCR images (generated) + images/vqa/ (public).
  README.md       Eval methodology + latest bake-off results and findings.
```

## Commands

```bash
# One-command native launch (host Ollama + venv + UI). No flags.
./start.sh                         # macOS/Linux/WSL/Git Bash
.\start.ps1                        # Windows PowerShell

# CLI
python cli.py tasks                       # list tasks + detected hardware + chosen model/backend
python cli.py tasks --backend qualcomm    # what a specific backend would pick
python cli.py run reasoning "..."         # run a text task
python cli.py run ocr --image x.png       # vision task
python cli.py run reasoning "..." --backend intel-npu --model <id>   # force backend/model

# HTTP API
uvicorn server:app --port 8000
#   GET  /tasks                 -> hardware + per-task chosen model
#   POST /run/{task}    (JSON {prompt, model?, backend?})       -> text/embed tasks
#   POST /run-image/{task} (multipart file + prompt?/model?/backend?) -> vision/ocr

# Streamlit UI
streamlit run app.py

# Prepare NPU models ahead of first use (also auto-runs from the launcher)
python convert.py reasoning            # detected NPU
python convert.py --all --backend amd-npu

# Eval / bake-off (drives Ollama directly)
python eval/gen_images.py            # regenerate OCR images + ocr.json (once)
python eval/fetch_vqa.py             # build vision.json from public VQAv2 (once)
python eval/run_eval.py --task all
python eval/run_eval.py --task ocr           # single task
python eval/run_eval.py --task reasoning --pull   # download missing models first
python eval/run_eval.py --task all --limit 2 # quick smoke test
```

There is no build step, linter config, or unit-test suite. "Tests" means the eval
bake-off in `eval/`, which requires a running Ollama and downloads models.

## Tasks

Six tasks, each with a `kind` that decides how input/output flow:

| Task | kind | Input | Output | Notes |
|------|------|-------|--------|-------|
| `ocr` | vision | image (+ optional prompt) | text | default prompt is `gateway.OCR_PROMPT` |
| `vision` | vision | image + question | text | general VQA / description |
| `reasoning` | text | prompt | text | math / multi-step / logic |
| `code` | text | prompt | text | code generation |
| `summary` | text | text | text | summarization |
| `embed` | embed | text | float vector | retrieval / RAG |
| `agent` | agent | prompt | text + files | tool-use loop; runs Python in a sandbox (see Executors) |

A task's `kind` is the dispatch key in `gateway.run()` and in the eval — there is **no
per-task-name special-casing in the gateway** (the summarize-style prompt wrapper lives
only in the eval). Front ends iterate `registry.TASKS` dynamically, so adding/renaming a
task in the registry propagates to CLI/API/UI automatically.

## Architecture

The flow is **task → model → backend**, with two independent selection decisions:

1. **Model selection** lives in `registry.py` (`TASKS`). Each `Task` has `tiers`: a list of
   `ModelTier(model, min_ram_gb)`. `Task.pick_model(budget_gb)` returns the largest tier
   whose `min_ram_gb` fits the budget (falling back to the smallest). Thresholds are set at
   6 / 16 / 24 / 32 GB; tiers can mix model families (e.g. `summary` uses Llama at 16 GB,
   Qwen at 32 GB) because the bake-off showed different families win different tasks.
2. **Backend selection** lives in `backends/` (`get_backend(name, task)`). `auto` falls
   back to Ollama unless a Qualcomm NPU is present; the other NPU backends are opt-in.

`gateway.run()` is the single orchestrator the CLI/API/UI call: it resolves the backend,
asks the backend to resolve the model (using the memory budget from `hardware`), calls
`backend.ensure_model()`, then dispatches on `task.kind` to `generate` (text/vision) or
`embed`. `gateway.py` keeps a stable public surface — `run`, `resolve_model`,
`available_ram_gb`, `OCR_PROMPT`, `BackendUnavailable` — that the three front ends and the
eval depend on; don't break these signatures.

### Request data flow

```
front end → gateway.run(task, prompt/image, model?, backend?)
  → get_backend(backend, task)            # backends/__init__.py
  → backend.resolve_model(task, model, hardware.memory_budget_gb())
  → backend.ensure_model(model)           # pull (Ollama) or prepare (NPU)
  → backend.generate(...) | backend.embed(...)
  → {"task", "model", "backend", "text" | "embedding"}
```

### Backends (`backends/`)

All implement the `Backend` ABC in `backends/base.py` (`is_available`, `resolve_model`,
`ensure_model`, `generate`, `embed`, `supports_task`). Failures raise `BackendUnavailable`
with an actionable message that the front ends surface verbatim.

- **`ollama.py`** — the default; talks to the Ollama HTTP API at `OLLAMA_HOST`
  (default `localhost:11434`). NVIDIA (CUDA) and AMD (ROCm) acceleration are automatic in
  stock Ollama; Intel Arc/Xe goes through an IPEX-LLM Ollama build (same API/port). Sets a
  default context window (`DEFAULT_NUM_CTX`, env `OLLAMA_NUM_CTX`, default 4096) on every
  `generate` so long inputs aren't truncated by Ollama's ~2048 default. When a dedicated
  GPU is detected, passes `num_gpu: 999` to offload all layers to VRAM. The launchers
  also set `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE` (q8_0 with GPU, q4_0
  CPU-only) to keep the KV cache small enough for 8 GB VRAM cards.
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
CLI (`tasks…`, `--all`, `--auto`, `--backend`, `--model`); the launcher runs
`convert.py --auto` at startup (a clean no-op when no NPU is present).

### Launcher GPU routing

`start.sh` / `start.ps1` run native only (no Docker, no flags). When **Intel is the only
GPU** (no NVIDIA/AMD for stock Ollama to use), they launch an IPEX-LLM Ollama build instead
of stock Ollama (set `IPEX_LLM_OLLAMA=/path/to/ollama`) with the SYCL env exported.

### Agent execution (`executors/`)

The `agent` task (`agent.py`) is an Ollama `/api/chat` tool-use loop with two
tools — `run_python` and `install_package` — executed in a sandbox. The sandbox
is **pluggable and no longer requires Docker**; `get_executor("auto")` picks the
strongest option that's actually usable:

```
docker  →  wasm  →  subprocess
```

- **`docker`** — one container per run, `--network none`, host-mounted workspace.
  Strongest isolation + full CPython; used when a Docker daemon is present.
- **`wasm`** — Pyodide (CPython→WASM) under Node, via `pyodide_driver.mjs`. The
  workspace is the only host path it can see (NODEFS) and there's no runtime
  network — strong, portable isolation with no daemon. The slim `pyodide` npm
  package ships only the core, so **auto only picks WASM when the scientific
  wheels are cached locally**; otherwise it's selectable explicitly for
  pure-Python work (`--executor wasm`).
- **`subprocess`** — a dedicated venv (pre-seeded with pandas/openpyxl/matplotlib),
  run in the workspace cwd with a scrubbed env, POSIX resource limits, and a
  timeout. **Blast-radius** isolation, not escape-proof — fits a *trusted* local
  model, and it's the Docker-free default because it delivers the full toolkit
  offline. Force any executor with `--executor` / `LLM_GATEWAY_EXECUTOR`.

All executors share the `Executor` ABC in `executors/base.py` (workspace +
new-file diffing live in the base; subclasses implement `_run_script`/`_install`)
and return `{stdout, stderr, exit_code, files}`. Agent code must write outputs as
**bare filenames** in the current directory (the workspace) — not absolute paths
like `/workspace`, which only exist in the Docker executor.

## Eval / bake-off subsystem (`eval/`)

Answers, per task and per RAM tier (16/24/32/48 GB): which downloaded model is most
accurate among those that *fit* that budget. It drives Ollama directly via
`OllamaBackend` (not `gateway.run`), so it can iterate raw model ids.

- **`candidates.py`** — `LADDERS[task] = (kind, [Cand(model, weights_gb), …])`. Ladders mix
  three axes: size, quantization (q4 default / q8_0 / fp16), and model family.
  `runtime_gb(weights, kind)` estimates RAM need (`weights*1.2 + 2.5` vision / `+1.5` text);
  `fits(weights, kind, total_ram)` requires ~5 GB OS headroom. Keys must match `registry.TASKS`.
- **`scorers.py`** — one scorer per task kind, all returning 0..1:
  - `ocr`: 0.5·(1−char-error-rate) + 0.5·key-token coverage (Levenshtein, dash/quote-normalized).
  - `vision` / `reasoning`: word-boundary match against an accept list (so "1" ≠ "10").
  - `summary`: key-fact coverage − length penalty.
  - `code`: functional — runs the extracted solution against assert tests in a subprocess.
  - `embed` is scored inline in `run_eval.py` (cosine: does the relevant doc rank #1?).
- **`run_eval.py`** — iterates the ladder, skips models not downloaded (or `--pull`), scores
  each over the dataset, prints a table + the per-tier winner (best score, ties broken by speed).
- **Datasets** — `gen_images.py` builds OCR images + `ocr.json`; `fetch_vqa.py` builds
  `vision.json` from public VQAv2; the rest (`reasoning/code/summary/embed.json`) are
  hand-authored with verifiable answers.

**Latest findings (see `eval/README.md` for tables):** model family wins flip by task —
Qwen2.5-VL leads OCR (MiniCPM-V trails), Qwen2.5 dominates math reasoning (Llama 3.1 8B is
weak), Llama 3.1 8B *beats* Qwen on summary, DeepSeek-Coder-V2 ties Qwen on code and is
faster. Quantization barely matters until a task is hard; OCR saturates at 7B, reasoning at
14B, code at 7B. These results drive the tiers in `registry.py`.

## Environment variables

| Var | Used by | Purpose |
|-----|---------|---------|
| `OLLAMA_HOST` | ollama backend | Ollama API base (default `http://localhost:11434`) |
| `OLLAMA_NUM_CTX` | ollama backend | Context window for generation (default 8192) |
| `LLM_GATEWAY_BACKEND` | cli | Default backend when `--backend` is omitted |
| `LLM_GATEWAY_EXECUTOR` | executors | Force agent sandbox (`docker`/`wasm`/`subprocess`); default `auto` |
| `LLM_GATEWAY_WORKSPACE` | executors | Root dir for agent run workspaces (default `~/.local-llm-gateway/workspaces`) |
| `IPEX_LLM_OLLAMA` | launchers | Path to IPEX-LLM Ollama for Intel-only-GPU machines |
| `QNN_SDK_ROOT`, `QNN_BACKEND_PATH`, `QAI_HUB_DEVICE`, `QAI_HUB_GATEWAY_CACHE` | qualcomm | QNN SDK + AI Hub export/cache |
| `OV_NPU_GATEWAY_CACHE` | intel_npu | OpenVINO compiled-model cache dir |
| `RYZEN_AI_INSTALLATION_PATH`, `RYZEN_AI_GATEWAY_CACHE`, `VAIP_CONFIG`, `XLNX_VART_FIRMWARE` | amd_npu | Ryzen AI / VitisAI config + cache |

## Extending the system

- **Change a model for a task** — Ollama: edit the `tiers` in `registry.py`; NPUs: edit the
  `*_MODELS` dict in that backend file. Or override per call with `--model`.
- **Add a task** — add a `Task` to `registry.TASKS` (pick a `kind`); the front ends pick it
  up automatically. For eval coverage also add a `LADDERS` entry, a `datasets/<task>.json`,
  and (for a new kind) a scorer in `eval/scorers.py`.
- **Add a backend** — implement the `Backend` ABC in `backends/`, register it in
  `backends/__init__.py`. Keep it hardware-gated and Ollama as the fallback.
- **Strengthen an eval** — if every model scores ~1.0 the dataset is too easy (it can't
  discriminate); add harder examples. `embed`/short-doc and `code`/standard-problems
  saturate quickly.

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
- Front ends must keep using `gateway.run()` and the stable public surface; don't reach into
  backend internals (the eval's use of `OllamaBackend._local_models()` is the one exception,
  and it's deliberately eval-only).
- Generated/fetched eval images are committed so the bake-off runs without a prep step;
  regenerate with `gen_images.py` / `fetch_vqa.py` if you change the datasets.
