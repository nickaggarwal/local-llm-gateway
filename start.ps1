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

# Choose the Ollama binary. When Intel is the *only* GPU (no NVIDIA/AMD that stock
# Ollama would accelerate), prefer an IPEX-LLM Ollama build for Arc/Xe (SYCL/XMX);
# it speaks the same HTTP API on the same port. Set IPEX_LLM_OLLAMA to enable it.
$OllamaBin = if ($env:OLLAMA_BIN) { $env:OLLAMA_BIN } else { "ollama" }
function IntelOnlyGpu {
  & $python -c "import hardware,sys; v={g.vendor for g in hardware.detect_gpus()}; sys.exit(0 if 'intel' in v and not (v & {'nvidia','amd'}) else 1)" 2>$null
  return ($LASTEXITCODE -eq 0)
}

if (IntelOnlyGpu) {
  if ($env:IPEX_LLM_OLLAMA -and (Test-Path $env:IPEX_LLM_OLLAMA)) {
    $OllamaBin = $env:IPEX_LLM_OLLAMA
    if (-not $env:OLLAMA_NUM_GPU)       { $env:OLLAMA_NUM_GPU = "999" }
    if (-not $env:ZES_ENABLE_SYSMAN)    { $env:ZES_ENABLE_SYSMAN = "1" }
    if (-not $env:SYCL_CACHE_PERSISTENT){ $env:SYCL_CACHE_PERSISTENT = "1" }
    Write-Host "==> Intel GPU detected — using IPEX-LLM Ollama: $OllamaBin"
  } else {
    Write-Host "==> Intel GPU detected. Install the IPEX-LLM Ollama build and set IPEX_LLM_OLLAMA"
    Write-Host "    (https://github.com/intel-analytics/ipex-llm). Continuing with stock Ollama."
  }
}

function Ensure-Ollama {
  if (-not (Have $OllamaBin)) { Write-Host "ERROR: Ollama not found ($OllamaBin). https://ollama.com/download"; exit 1 }
  if (OllamaUp) { Write-Host "==> Ollama already running."; return }
  Write-Host "==> Starting Ollama ($OllamaBin)..."
  Start-Process -NoNewWindow $OllamaBin -ArgumentList "serve" | Out-Null
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
