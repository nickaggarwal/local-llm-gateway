#!/usr/bin/env bash
#
# Local LLM Gateway launcher (native, fast mode).
# Starts host Ollama (GPU-accelerated) + the UI in a local virtualenv.
#
# Usage:
#   ./start.sh
#
set -euo pipefail
cd "$(dirname "$0")"

if [ "$#" -gt 0 ]; then
  echo "usage: ./start.sh   (native mode only; no flags)"
  exit 1
fi

OLLAMA_URL="http://localhost:11434"
UI_URL="http://localhost:8501"

have()      { command -v "$1" >/dev/null 2>&1; }
ollama_up() { curl -fs "$OLLAMA_URL/api/version" >/dev/null 2>&1; }

# Pick a python command (Windows/Git Bash often has only `python`).
PYTHON=""
for c in python3 python; do have "$c" && { PYTHON="$c"; break; }; done
[ -n "$PYTHON" ] || { echo "ERROR: Python not found."; exit 1; }

# Choose the Ollama binary. When Intel is the *only* GPU (no NVIDIA/AMD that stock
# Ollama would accelerate), prefer an IPEX-LLM Ollama build for Arc/Xe (SYCL/XMX).
# It speaks the same HTTP API on the same port, so the gateway is unchanged.
# Enable it by setting IPEX_LLM_OLLAMA=/path/to/ollama.
OLLAMA_BIN="${OLLAMA_BIN:-ollama}"
intel_only_gpu() {
  "$PYTHON" -c "import hardware,sys; v={g.vendor for g in hardware.detect_gpus()}; sys.exit(0 if 'intel' in v and not (v & {'nvidia','amd'}) else 1)" 2>/dev/null
}

if intel_only_gpu; then
  if [ -n "${IPEX_LLM_OLLAMA:-}" ] && [ -x "${IPEX_LLM_OLLAMA}" ]; then
    OLLAMA_BIN="$IPEX_LLM_OLLAMA"
    export OLLAMA_NUM_GPU="${OLLAMA_NUM_GPU:-999}"
    export ZES_ENABLE_SYSMAN="${ZES_ENABLE_SYSMAN:-1}"
    export SYCL_CACHE_PERSISTENT="${SYCL_CACHE_PERSISTENT:-1}"
    echo "==> Intel GPU detected — using IPEX-LLM Ollama: $OLLAMA_BIN"
  else
    echo "==> Intel GPU detected. For GPU acceleration, install the IPEX-LLM Ollama build and"
    echo "    set IPEX_LLM_OLLAMA=/path/to/ollama  (https://github.com/intel-analytics/ipex-llm)."
    echo "    Continuing with stock Ollama for now."
  fi
fi

ensure_ollama_host() {
  have "$OLLAMA_BIN" || { echo "ERROR: Ollama not found ($OLLAMA_BIN). Install: https://ollama.com/download"; exit 1; }
  if ollama_up; then echo "==> Ollama already running."; return; fi
  echo "==> Starting Ollama ($OLLAMA_BIN)..."
  "$OLLAMA_BIN" serve >"${TMPDIR:-/tmp}/ollama.log" 2>&1 &
  for _ in $(seq 1 30); do ollama_up && break; sleep 1; done
  ollama_up || { echo "ERROR: Ollama failed to start. See ${TMPDIR:-/tmp}/ollama.log"; exit 1; }
  echo "==> Ollama is up."
}

ensure_ollama_host

if [ ! -d .venv ]; then
  echo "==> Creating virtualenv (.venv)..."
  "$PYTHON" -m venv .venv
fi
# Activate (POSIX path or Windows/Git Bash Scripts path).
if [ -f .venv/bin/activate ]; then . .venv/bin/activate
elif [ -f .venv/Scripts/activate ]; then . .venv/Scripts/activate
fi

echo "==> Installing dependencies..."
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt

# Prepare NPU models if an NPU is present and its cache is empty (no-op otherwise).
echo "==> Checking NPU model cache..."
python convert.py --auto || echo "WARN: NPU model preparation skipped/failed; continuing with Ollama."

echo "==> Launching UI at $UI_URL  (Ctrl+C to stop)"
exec streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
