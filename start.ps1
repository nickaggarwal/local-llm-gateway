# Local LLM Gateway launcher for Windows PowerShell (native, fast mode).
# Starts host Ollama (GPU-accelerated) + the UI in a local virtualenv.
#
# Usage:
#   .\start.ps1
#
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$OllamaUrl = "http://localhost:11434"
$UiUrl     = "http://localhost:8501"

function Have($cmd) { $null -ne (Get-Command $cmd -ErrorAction SilentlyContinue) }
function OllamaUp {
  try { Invoke-RestMethod "$OllamaUrl/api/version" -TimeoutSec 2 | Out-Null; $true }
  catch { $false }
}

$python = @("python", "python3") | Where-Object { Have $_ } | Select-Object -First 1
if (-not $python) { Write-Host "ERROR: Python not found."; exit 1 }

function Ensure-Ollama {
  if (-not (Have "ollama")) { Write-Host "ERROR: Ollama not found. https://ollama.com/download"; exit 1 }
  if (OllamaUp) { Write-Host "==> Ollama already running."; return }
  Write-Host "==> Starting Ollama..."
  Start-Process -NoNewWindow ollama -ArgumentList "serve" | Out-Null
  for ($i = 0; $i -lt 30; $i++) { if (OllamaUp) { break }; Start-Sleep 1 }
  if (-not (OllamaUp)) { Write-Host "ERROR: Ollama failed to start."; exit 1 }
  Write-Host "==> Ollama is up."
}

Ensure-Ollama
if (-not (Test-Path ".venv")) { Write-Host "==> Creating virtualenv..."; & $python -m venv .venv }
& ".venv\Scripts\Activate.ps1"
Write-Host "==> Installing dependencies..."
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt
# Prepare NPU models if an NPU is present and its cache is empty (no-op otherwise).
Write-Host "==> Checking NPU model cache..."
python convert.py --auto
if ($LASTEXITCODE -ne 0) { Write-Host "WARN: NPU model preparation skipped/failed; continuing with Ollama." }
Write-Host "==> Launching UI at $UiUrl  (Ctrl+C to stop)"
streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
