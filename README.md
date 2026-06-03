# Local LLM Gateway

Pick the **best local model for each task type**, download it automatically on first
use, and run it — entirely on your laptop. No data leaves the machine. Powered by
[Ollama](https://ollama.com).

## One-command start (auto-detects everything)

The launcher detects your OS, chooses native vs Docker, starts Ollama, sets up the
environment, and opens the UI.

**macOS / Linux / WSL / Git Bash:**

```bash
./start.sh                # auto
./start.sh --docker        # force Docker
./start.sh --native        # force native (venv + host Ollama)
```

**Windows (PowerShell):**

```powershell
.\start.ps1                 # auto
.\start.ps1 -Mode docker
.\start.ps1 -Mode native
```

Then open **http://localhost:8501**.

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
2. The gateway looks up the best model for that task in [`registry.py`](registry.py),
   sized to your machine's RAM (it has tiers; bigger RAM → bigger/better model).
3. If the model isn't downloaded yet, it pulls it automatically.
4. It runs the model and returns the result.

### Model per task and laptop RAM

The model is chosen from your machine's **total RAM** (leaving headroom for the OS),
so a 16 GB laptop runs ~7B models, 24 GB runs ~14B, and 32–48 GB runs ~32B.

| Task | 16 GB | 24 GB | 32 GB | 48 GB |
|------|-------|-------|-------|-------|
| `ocr` | qwen2.5vl:7b | qwen2.5vl:7b | qwen2.5vl:7b | qwen2.5vl:32b |
| `vision` | qwen2.5vl:7b | qwen2.5vl:7b | qwen2.5vl:32b | qwen2.5vl:32b |
| `chat` | qwen2.5:7b | qwen2.5:14b | qwen2.5:32b | qwen2.5:32b |
| `code` | qwen2.5-coder:7b | qwen2.5-coder:14b | qwen2.5-coder:32b | qwen2.5-coder:32b |
| `summarize` | qwen2.5:7b | qwen2.5:14b | qwen2.5:14b | qwen2.5:14b |
| `embed` | bge-m3 | bge-m3 | bge-m3 | bge-m3 |

Machines under 16 GB fall back to 3B models (and `nomic-embed-text` for embeddings).
You can always override with `--model` (CLI) or the sidebar (UI).

**Why these models (2026 research):**

- **OCR / vision** — Qwen2.5-VL leads local VLMs on document benchmarks (DocVQA 95.7);
  the top OCR finetune olmOCR-2 is built on Qwen2.5-VL-7B. Main cross-family alternative:
  MiniCPM-V. (LLaVA / Granite-vision / Llama3.2-vision rate lower with more hallucination.)
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

## Customizing

- Add or change models per task in [`registry.py`](registry.py) — each task has RAM-sized tiers.
- Add a whole new task type by adding an entry to `TASKS`.

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
