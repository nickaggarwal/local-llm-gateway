#!/usr/bin/env bash
#
# Local LLM Gateway launcher.
# Figures out your environment and starts everything (Ollama + the UI).
#
# Usage:
#   ./start.sh              # auto-detect the best way to run
#   ./start.sh --native     # force native (venv + host Ollama)
#   ./start.sh --docker      # force Docker Compose
#
set -euo pipefail
cd "$(dirname "$0")"

OLLAMA_URL="http://localhost:11434"
UI_URL="http://localhost:8501"

have()       { command -v "$1" >/dev/null 2>&1; }
ollama_up()  { curl -fs "$OLLAMA_URL/api/version" >/dev/null 2>&1; }

detect_os() {
  case "$(uname -s 2>/dev/null)" in
    Darwin) echo "mac" ;;
    Linux)  grep -qi microsoft /proc/version 2>/dev/null && echo "wsl" || echo "linux" ;;
    MINGW*|MSYS*|CYGWIN*) echo "windows" ;;
    *) echo "unknown" ;;
  esac
}

# Pick a python command (Windows/Git Bash often has only `python`).
PYTHON=""
for c in python3 python; do have "$c" && { PYTHON="$c"; break; }; done

# Parse mode.
MODE="auto"
case "${1:-}" in
  ""|auto|--auto) MODE="auto" ;;
  native|--native) MODE="native" ;;
  docker|--docker) MODE="docker" ;;
  *) echo "usage: ./start.sh [--native|--docker]"; exit 1 ;;
esac

OS="$(detect_os)"
echo "==> OS: $OS"

# Auto-pick a mode: prefer native (fastest, GPU); fall back to Docker.
if [ "$MODE" = "auto" ]; then
  if have ollama && [ -n "$PYTHON" ]; then
    MODE="native"
  elif have docker; then
    MODE="docker"
  else
    echo "ERROR: need either (Ollama + Python) or Docker installed."
    echo "  - Ollama: https://ollama.com/download"
    echo "  - Docker: https://docs.docker.com/get-docker/"
    exit 1
  fi
fi
echo "==> Mode: $MODE"

ensure_ollama_host() {
  have ollama || { echo "ERROR: Ollama not found. Install: https://ollama.com/download"; exit 1; }
  if ollama_up; then echo "==> Ollama already running."; return; fi
  echo "==> Starting Ollama..."
  ollama serve >"${TMPDIR:-/tmp}/ollama.log" 2>&1 &
  for _ in $(seq 1 30); do ollama_up && break; sleep 1; done
  ollama_up || { echo "ERROR: Ollama failed to start. See ${TMPDIR:-/tmp}/ollama.log"; exit 1; }
  echo "==> Ollama is up."
}

start_native() {
  [ -n "$PYTHON" ] || { echo "ERROR: Python not found."; exit 1; }
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

  echo "==> Launching UI at $UI_URL  (Ctrl+C to stop)"
  exec streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
}

start_docker() {
  have docker || { echo "ERROR: Docker not found."; exit 1; }
  if [ "$OS" = "mac" ] && have ollama; then
    ensure_ollama_host
    echo "==> Host Ollama (GPU) + gateway container. UI at $UI_URL"
    exec docker compose -f docker-compose.host-ollama.yml up --build
  fi
  if [ "$OS" = "mac" ]; then
    echo "NOTE: Ollama not installed on host — the container will run it on CPU (slow on Mac)."
    echo "      For GPU speed: brew install --cask ollama, then re-run."
  fi
  echo "==> Starting Ollama + gateway via Docker Compose. UI at $UI_URL"
  exec docker compose up --build
}

case "$MODE" in
  native) start_native ;;
  docker) start_docker ;;
esac
