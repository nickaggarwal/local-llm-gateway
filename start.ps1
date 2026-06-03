# Local LLM Gateway launcher for Windows PowerShell.
#
# Usage:
#   .\start.ps1            # auto-detect (native if Ollama+Python, else Docker)
#   .\start.ps1 -Mode native
#   .\start.ps1 -Mode docker
#
param([ValidateSet("auto", "native", "docker")] [string]$Mode = "auto")

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

if ($Mode -eq "auto") {
  if ((Have "ollama") -and $python) { $Mode = "native" }
  elseif (Have "docker")            { $Mode = "docker" }
  else {
    Write-Host "ERROR: need either (Ollama + Python) or Docker installed."
    Write-Host "  - Ollama: https://ollama.com/download"
    Write-Host "  - Docker: https://docs.docker.com/get-docker/"
    exit 1
  }
}
Write-Host "==> Mode: $Mode"

function Ensure-Ollama {
  if (-not (Have "ollama")) { Write-Host "ERROR: Ollama not found. https://ollama.com/download"; exit 1 }
  if (OllamaUp) { Write-Host "==> Ollama already running."; return }
  Write-Host "==> Starting Ollama..."
  Start-Process -NoNewWindow ollama -ArgumentList "serve" | Out-Null
  for ($i = 0; $i -lt 30; $i++) { if (OllamaUp) { break }; Start-Sleep 1 }
  if (-not (OllamaUp)) { Write-Host "ERROR: Ollama failed to start."; exit 1 }
  Write-Host "==> Ollama is up."
}

if ($Mode -eq "native") {
  if (-not $python) { Write-Host "ERROR: Python not found."; exit 1 }
  Ensure-Ollama
  if (-not (Test-Path ".venv")) { Write-Host "==> Creating virtualenv..."; & $python -m venv .venv }
  & ".venv\Scripts\Activate.ps1"
  Write-Host "==> Installing dependencies..."
  python -m pip install -q --upgrade pip
  python -m pip install -q -r requirements.txt
  Write-Host "==> Launching UI at $UiUrl  (Ctrl+C to stop)"
  streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
}
elseif ($Mode -eq "docker") {
  if (-not (Have "docker")) { Write-Host "ERROR: Docker not found."; exit 1 }
  Write-Host "==> Starting Ollama + gateway via Docker Compose. UI at $UiUrl"
  docker compose up --build
}
