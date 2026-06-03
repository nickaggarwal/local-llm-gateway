# Local LLM Gateway

Pick the **best local model for each task type**, download it automatically on first
use, and run it — entirely on your laptop. No data leaves the machine. Powered by
[Ollama](https://ollama.com).

## One-command start (native, fast)

The launcher runs **natively** for full GPU speed: it starts host Ollama, sets up a
local virtualenv, installs deps, prepares NPU models if an NPU is present (`convert.py
--auto`, a no-op otherwise), and opens the UI. No flags, no Docker.

**macOS / Linux / WSL / Git Bash:**

```bash
./start.sh
```

**Windows (PowerShell):**

```powershell
.\start.ps1
```

Then open **http://localhost:8501**. (Prefer containers? See [Docker](#quick-start-docker--easiest) below.)

## Quick start (Docker — easiest)

Requires only [Docker](https://docs.docker.com/get-docker/). No Python, no Ollama install.

**Linux / Windows:**

```bash
git clone https://github.com/nickaggarwal/local-llm-gateway
cd local-llm-gateway
docker compose up --build
```

**macOS** (Docker can't use the Apple GPU, so run Ollama natively for speed):

```bash
brew install --cask ollama && ollama serve   # in one terminal
# in another:
git clone https://github.com/nickaggarwal/local-llm-gateway
cd local-llm-gateway
docker compose -f docker-compose.host-ollama.yml up --build
```

Then open **http://localhost:8501**. The first time you use a task, its model
downloads automatically (progress shown in the UI). That's it.

> To pre-download a model instead of waiting on first use:
> `docker exec ollama ollama pull qwen2.5vl:7b`

## How it works

1. You ask for a **task** (`ocr`, `chat`, `code`, `summarize`, `vision`, `embed`).
2. The gateway picks the best **backend** for your machine (see below) and the best
   **model** for that task in [`registry.py`](registry.py), sized to your machine's
   memory budget (it has tiers; bigger budget → bigger/better model).
3. If the model isn't downloaded yet, it pulls it automatically.
4. It runs the model and returns the result.

## Backends (Ollama GPU, Qualcomm / Intel / AMD NPU)

Inference runs through a pluggable backend ([`backends/`](backends/)). The gateway
auto-selects, or you can force one with `--backend` (CLI), the **Backend** dropdown
(UI), a `backend` field (API), or the `LLM_GATEWAY_BACKEND` env var. Available names:
`auto`, `ollama`, `qualcomm`, `intel-npu`, `amd-npu`.

- **`ollama`** (default, everywhere) — runs models via Ollama. What the gateway adds is
  **GPU-aware model sizing**: Ollama/llama.cpp offloads model layers across GPU VRAM and
  CPU RAM, so the gateway sizes the model tier off the **combined `RAM + VRAM` budget**.
  A GPU therefore only ever lets you run a *bigger* model and accelerates the layers that
  fit in VRAM — it never shrinks the model you'd have run on CPU alone. Layers beyond VRAM
  are CPU-offloaded (slower, but they still run). Only **dedicated** VRAM is added to the
  budget; integrated GPUs share system RAM, so counting them would double-count.

  GPU acceleration itself is the runtime's job, not the gateway's — what differs per vendor
  is which Ollama runtime you run:
  - **NVIDIA** (CUDA) — used automatically by stock Ollama. Detected via NVML / `nvidia-smi`.
  - **AMD Radeon** (ROCm) — used automatically by stock Ollama where ROCm is supported.
    Detected via DRM sysfs (`/sys/class/drm`) or `rocm-smi`.
  - **Intel Arc** (XMX / SYCL) — **not** in stock Ollama; run the
    [IPEX-LLM Ollama build](https://github.com/intel-analytics/ipex-llm) and point
    `OLLAMA_HOST` at it (same HTTP API, so this backend talks to it unchanged). Detected
    via DRM sysfs.

  > The acceleration above is via the **GPU** (CUDA / ROCm / SYCL-XMX). The *dedicated
  > NPUs* on these platforms (AMD XDNA / "Ryzen AI", Intel Core Ultra NPU) are a separate
  > path Ollama doesn't use — they have their own backends below.

The remaining backends target **dedicated NPUs**. Each is hardware-gated: on a machine
without that NPU it reports unavailable (with setup instructions) and the gateway falls
back to Ollama. Only **`qualcomm`** is auto-selected when present — on Snapdragon there's
no Ollama GPU path, so the NPU is the accelerator. **`intel-npu`** and **`amd-npu`** are
**opt-in only** (`--backend …`): on x86 the Ollama GPU path is the better default, so we
don't silently route to a weaker NPU.

- **`qualcomm`** (Snapdragon, auto) — runs [Qualcomm AI Hub](https://aihub.qualcomm.com)
  models pre-compiled for the **Hexagon NPU** (text LLMs via the Genie runtime; ONNX
  graphs via ONNX Runtime's QNN execution provider). Detected via QNN SDK / `QNN_SDK_ROOT`,
  ONNX Runtime's `QNNExecutionProvider`, or a Snapdragon Windows-on-ARM CPU. See
  [Qualcomm NPU setup](#qualcomm-npu-setup).
- **`intel-npu`** (Intel Core Ultra, opt-in) — runs text LLMs on the **Intel NPU** via
  **OpenVINO** (`device="NPU"`) through optimum-intel, which handles tokenization and the
  decode loop. Detected via OpenVINO's `NPU` device or the DRM accel subsystem. Needs models
  exported to OpenVINO IR — see [Intel NPU setup](#intel-npu-setup).
- **`amd-npu`** (AMD Ryzen AI, opt-in) — runs quantized ONNX LLMs on the **AMD XDNA NPU**
  via ONNX Runtime's **VitisAI** execution provider. Detected via the VitisAI EP, Ryzen AI
  env vars, or the DRM accel subsystem. Needs Ryzen-AI-quantized ONNX models — see
  [AMD Ryzen AI setup](#amd-ryzen-ai-npu-setup).

> **NPU scaffolds:** the Intel path is functional end-to-end on hardware (optimum-intel
> drives generation). The Qualcomm ONNX path and the AMD path load/validate the NPU session
> but stop short of the model-specific tokenizer + decode loop — that last step is wired on
> the target device. Embeddings and vision tasks are not yet mapped on any NPU and fall back
> to Ollama. All NPU backends require their vendor SDK + per-device model conversion, so they
> can't run on a Mac/x86 host without an NPU.

`python cli.py tasks` (and `GET /tasks`) print the detected hardware, the active backend,
and the model each task would use.

### Model per task and laptop RAM

The model is chosen from your machine's **total RAM** (leaving headroom for the OS),
so a 16 GB laptop runs ~7B models, 24 GB runs ~14B, and 32–48 GB runs ~32B.

| Task | 16 GB | 24 GB | 32 GB | 48 GB |
|------|-------|-------|-------|-------|
| `ocr` | qwen2.5vl:7b | qwen2.5vl:7b | qwen2.5vl:7b | qwen2.5vl:7b |
| `vision` | qwen2.5vl:7b | qwen2.5vl:7b | qwen2.5vl:7b | qwen2.5vl:7b |
| `chat` | qwen2.5:7b | qwen2.5:14b | qwen2.5:32b | qwen2.5:32b |
| `code` | qwen2.5-coder:7b | qwen2.5-coder:14b | qwen2.5-coder:32b | qwen2.5-coder:32b |
| `summarize` | qwen2.5:7b | qwen2.5:14b | qwen2.5:14b | qwen2.5:14b |
| `embed` | bge-m3 | bge-m3 | bge-m3 | bge-m3 |

Machines under 16 GB fall back to 3B models (and `nomic-embed-text` for embeddings).
You can always override with `--model` (CLI) or the sidebar (UI).

**Why these models (2026 research):**

- **OCR / vision** — Qwen2.5-VL leads local VLMs on document benchmarks (DocVQA 95.7);
  the top OCR finetune olmOCR-2 is built on Qwen2.5-VL-7B. Our [bake-off](eval/README.md)
  confirms it: Qwen2.5-VL scored ~0.95 vs MiniCPM-V at 0.625, and OCR accuracy was flat
  across sizes (3B≈7B) and quants (q4≈q8≈fp16) — so OCR/vision top out at **7B q4**;
  bigger/higher-precision bought nothing measurable (32B also fails to load on Ollama 0.30.2).
- **code** — Qwen2.5-Coder is state-of-the-art open source (32B scores 92.7% HumanEval,
  beating GPT-4o); DeepSeek-Coder-V2 is a faster MoE alternative and Codestral is best at
  fill-in-the-middle completion.
- **chat / summarize** — Qwen2.5 is strongest per-size on consumer hardware; Phi-4 14B is
  notably strong on STEM/reasoning, with Llama 3.1, Gemma 2, and Mistral as alternatives.
- **embed** — switched the default to **bge-m3** (8192-token context, multilingual):
  mxbai-embed-large scores higher on English retrieval but **silently truncates at 512
  tokens**, which corrupts embeddings of long text. nomic-embed-text is the lightweight
  fallback. Qwen3-Embedding tops MTEB but is heavy.

These alternatives are wired into the [eval ladders](eval/candidates.py) so the bake-off
compares model *families*, not just Qwen sizes. Sources are linked at the bottom.

## Manual setup (without Docker)

```bash
# Ollama (one time): brew install --cask ollama   (or see ollama.com for Linux)
ollama serve            # keep running in a terminal

cd local-llm-gateway
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## CLI

```bash
python cli.py tasks                       # show tasks + which model each will use
python cli.py run ocr --image receipt.png
python cli.py run chat "Explain RAG in one sentence."
python cli.py run code "Reverse a linked list in Python."
python cli.py run summarize "<long text...>"
python cli.py run embed "hello world"
python cli.py run ocr --image x.png --model qwen2.5vl:3b   # force a model
python cli.py run chat "Hi" --backend qualcomm             # force a backend (Snapdragon NPU)
python cli.py tasks --backend qualcomm                     # see NPU model per task
```

First call for a new task downloads the model (progress shown), then runs it.

## Streamlit UI

```bash
streamlit run app.py
```

Opens at http://localhost:8501. Pick a task from the dropdown. For `ocr`/`vision`,
upload an image then use the **chat box** to send extraction directions — e.g.
"just the total", "the table as markdown", "the invoice number" — and iterate across
turns. For `chat`/`code`/`summarize` the chat box drives the conversation; `embed`
uses a one-shot input. The chosen model and download progress show inline, and a
**Clear chat** button in the sidebar resets the conversation.

## HTTP API

```bash
uvicorn server:app --port 8000
```

```bash
curl -s localhost:8000/tasks | jq

# text tasks
curl -s -X POST localhost:8000/run/chat \
  -H 'content-type: application/json' \
  -d '{"prompt":"Explain RAG in one sentence."}'

# vision/ocr tasks (file upload)
curl -s -X POST localhost:8000/run-image/ocr -F "file=@receipt.png"
```

## Platform notes

- **Windows**: works via Docker (`docker compose up --build`) or natively. Use
  `start.ps1` in PowerShell, or `start.sh` under Git Bash / WSL. `host.docker.internal`
  is available on Docker Desktop, so the host-Ollama compose file works too.
- **macOS**: Docker can't access the Apple GPU, so prefer native Ollama (the launcher
  does this automatically) or the `docker-compose.host-ollama.yml` override.
- **Linux**: `docker compose up` bundles Ollama; uncomment the GPU block in
  `docker-compose.yml` to use an NVIDIA GPU (needs nvidia-container-toolkit).

## Preparing NPU models (conversion)

NPU models aren't plain Ollama pulls — each vendor needs a device-specific artifact
(OpenVINO IR, a QNN/Genie bundle, or VitisAI-quantized ONNX). The gateway **prepares it
automatically on first use**: each NPU backend's model cache is checked, and if it's empty
the converter ([`backends/convert.py`](backends/convert.py)) either **downloads a
pre-compiled equivalent** or **runs the vendor conversion pipeline** into that cache.

You can also prepare ahead of time with the [`convert.py`](convert.py) script (it
auto-detects the NPU, or pass `--backend`):

```bash
python convert.py chat                  # prepare 'chat' for the detected NPU
python convert.py --all                 # prepare every task the backend supports
python convert.py ocr --backend qualcomm
```

Either way needs the vendor SDK installed on the target device — if a conversion tool is
missing, the converter raises the exact command to run. The vendor SDK + per-device
setup is described below.

## Qualcomm NPU setup

The Qualcomm backend targets the **Hexagon NPU** on Snapdragon devices (it can't run on a
Mac/x86 host):

1. **Qualcomm AI Engine Direct (QNN) SDK** with the **Genie** runtime — set
   `QNN_SDK_ROOT` and put `genie-t2t-run` on `PATH`. For ONNX models also install
   `onnxruntime-qnn`.
2. **Model artifact.** Prepared automatically on first use (or via `python convert.py
   <task> --backend qualcomm`), which runs the AI Hub export/compile for your device — set
   `QAI_HUB_DEVICE` (default `"Snapdragon X Elite CRD"`). Needs `pip install qai-hub-models`
   and an AI Hub API token. Artifacts cache under `~/.cache/qai-hub-gateway/<model>/`
   (override `QAI_HUB_GATEWAY_CACHE`).
3. Run it: `python cli.py run chat "Hi" --backend qualcomm`.

The task→model map lives in [`backends/qualcomm.py`](backends/qualcomm.py)
(`QUALCOMM_MODELS`); the slugs are `qai_hub_models` module names — confirm the exact
name/device in the AI Hub catalog, as it evolves. `embed` is not yet wired on the NPU and
stays on Ollama.

## Intel NPU setup

The `intel-npu` backend targets the **Intel Core Ultra NPU** via OpenVINO:

1. Install the runtime: `pip install optimum[openvino]` (plus an OpenVINO build that
   exposes the `NPU` device).
2. **Model artifact.** Prepared automatically on first use (or via `python convert.py
   <task> --backend intel-npu`): the converter runs `optimum-cli export openvino
   --weight-format int4`, or downloads a ready-made IR if an `OvModel.precompiled` repo is
   mapped. IR caches under `~/.cache/ov-npu-gateway/<model>/` (override `OV_NPU_GATEWAY_CACHE`).
3. Run it: `python cli.py run chat "Hi" --backend intel-npu`.

Generation runs end to end through optimum-intel (`OVModelForCausalLM`, `device="NPU"`).
The task→model map lives in [`backends/intel_npu.py`](backends/intel_npu.py)
(`INTEL_MODELS`).

## AMD Ryzen AI (NPU) setup

The `amd-npu` backend targets the **AMD XDNA NPU** via ONNX Runtime's VitisAI EP:

1. Install the **Ryzen AI SW** stack (provides the VitisAI-enabled `onnxruntime`).
2. **Model artifact.** Prepared automatically on first use (or via `python convert.py
   <task> --backend amd-npu`): the converter downloads the pre-quantized `amd/*` ONNX repo
   mapped for the task. (For a model you quantize yourself with `vai_q_onnx`, drop the
   `*.onnx` in the cache dir instead.) Caches under `~/.cache/ryzen-ai-gateway/<model>/`
   (override `RYZEN_AI_GATEWAY_CACHE`; point `VAIP_CONFIG` at the config file).
3. Run it: `python cli.py run chat "Hi" --backend amd-npu`.

The scaffold loads/validates the VitisAI session; the per-model tokenizer + decode loop is
wired on the device (Ryzen AI LLMs use a model-specific generation runner). The task→model
map lives in [`backends/amd_npu.py`](backends/amd_npu.py) (`AMD_MODELS`) — confirm slugs per
Ryzen AI release.

## Customizing

- Switch backends per call with `--backend {auto,ollama,qualcomm,intel-npu,amd-npu}`
  (or `LLM_GATEWAY_BACKEND`).
- **Change the model per task** — the map depends on the backend:
  - Ollama: [`registry.py`](registry.py) (`TASKS`), with memory-sized tiers.
  - Qualcomm: `QUALCOMM_MODELS` in [`backends/qualcomm.py`](backends/qualcomm.py).
  - Intel NPU: `INTEL_MODELS` in [`backends/intel_npu.py`](backends/intel_npu.py).
  - AMD NPU: `AMD_MODELS` in [`backends/amd_npu.py`](backends/amd_npu.py).
  - Or override per call with `--model <id>` on any backend.
- Add a whole new task type by adding an entry to `TASKS`.
- Add a new backend by implementing the `Backend` interface in [`backends/base.py`](backends/base.py).
- Add a conversion pipeline for a new NPU runtime in [`backends/convert.py`](backends/convert.py).

## Model selection — sources

- [Best Ollama Models in 2026: A Practical Guide by Use Case](https://mljourney.com/best-ollama-models-in-2026-a-practical-guide-by-use-case/)
- [Best Local Vision-Language Models for Offline AI (Roboflow)](https://blog.roboflow.com/local-vision-language-models/) — Qwen2.5-VL for OCR/documents
- [Best Local Coding Models Ranked, Every VRAM Tier (InsiderLLM)](https://insiderllm.com/guides/best-local-coding-models-2026/) — Qwen2.5-Coder per VRAM tier
- [Ollama VRAM Requirements: Complete 2026 Guide (LocalLLM.in)](https://localllm.in/blog/ollama-vram-requirements-for-local-llms)
- [Ollama Embedding Models: Benchmarks, VRAM (Morph)](https://www.morphllm.com/ollama-embedding-models) — nomic-embed-text / mxbai-embed-large
- [Which Embedding Model Should You Use in 2026? (10-model benchmark)](https://zc277584121.github.io/rag/2026/03/20/embedding-models-benchmark-2026.html) — bge-m3 vs mxbai 512-ctx limit
- [Best Open-Weight Embedding Models 2026 (Presenc)](https://presenc.ai/research/best-open-weight-embedding-models-2026) — Qwen3-Embedding / bge-m3 / MTEB
- [Qwen2.5-Coder vs DeepSeek vs Codestral (2026)](https://www.aimadetools.com/blog/best-open-source-coding-model-2026/)
- [Show HN: Qwen-2.5-32B best open source OCR model](https://news.ycombinator.com/item?id=43549072) — olmOCR / Qwen2.5-VL
