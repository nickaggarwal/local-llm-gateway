# Local LLM Gateway launcher for Windows PowerShell (native, fast mode).
# Starts host Ollama (GPU-accelerated) + the UI in a local virtualenv.
#
# Usage:
#   .\start.ps1
#
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$OllamaUrl = "http://127.0.0.1:11434"
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
# If ollama isn't on PATH, check the default install location.
if (-not (Have $OllamaBin)) {
  $defaultPath = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
  if (Test-Path $defaultPath) { $OllamaBin = $defaultPath }
}
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
    Write-Host "==> Intel GPU detected - using IPEX-LLM Ollama: $OllamaBin"
  } else {
    Write-Host "==> Intel GPU detected. Install the IPEX-LLM Ollama build and set IPEX_LLM_OLLAMA"
    Write-Host "    (https://github.com/intel-analytics/ipex-llm). Continuing with stock Ollama."
  }
}

function Install-Ollama {
  Write-Host "==> Ollama not found. Installing..."
  # Try winget first (available on Windows 10 1709+ / Windows 11)
  if (Have winget) {
    winget install --id Ollama.Ollama --source winget --silent --accept-package-agreements --accept-source-agreements
    # Refresh PATH so the new binary is visible in this session
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("PATH", "User")
  } else {
    # Fallback: download the official installer and run it silently
    $installer = "$env:TEMP\OllamaSetup.exe"
    Write-Host "==> Downloading Ollama installer..."
    Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile $installer
    Write-Host "==> Running installer (silent)..."
    Start-Process -FilePath $installer -ArgumentList "/S" -Wait
    Remove-Item $installer -Force
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("PATH", "User")
  }
  if (-not (Have $OllamaBin)) {
    Write-Host "ERROR: Ollama installation failed. Install manually: https://ollama.com/download"
    exit 1
  }
  Write-Host "==> Ollama installed successfully."
}

# Persist GPU-optimized Ollama env vars so the tray app and any future
# Ollama start (manual, on-boot) always uses the right config.
function Ensure-Ollama-Env {
  $hasGpu = & $python -c "import hardware,sys; sys.exit(0 if hardware.has_gpu() else 1)" 2>$null; ($LASTEXITCODE -eq 0)
  $kvType = if ($hasGpu) { "q8_0" } else { "q4_0" }
  $vars = @{
    OLLAMA_FLASH_ATTENTION = "1"
    OLLAMA_KV_CACHE_TYPE   = $kvType
    OLLAMA_GPU_OVERHEAD    = "0"
  }
  foreach ($k in $vars.Keys) {
    $current = [System.Environment]::GetEnvironmentVariable($k, "User")
    if ($current -ne $vars[$k]) {
      [System.Environment]::SetEnvironmentVariable($k, $vars[$k], "User")
      Write-Host "    Set $k=$($vars[$k]) (persistent)"
    }
    $env:$k = $vars[$k]
  }
  Write-Host "==> Ollama config: FLASH_ATTENTION=1  KV_CACHE=$kvType  GPU_OVERHEAD=0"
}

function Ensure-Ollama {
  if (-not (Have $OllamaBin)) { Install-Ollama }
  Ensure-Ollama-Env
  if (OllamaUp) {
    Write-Host "==> Ollama already running."
    # If tray app started Ollama without our env vars, restart it.
    $needsRestart = $false
    try {
      $ver = Invoke-RestMethod "$OllamaUrl/api/version" -TimeoutSec 2
    } catch { $needsRestart = $true }
    if (-not $needsRestart) { return }
  }
  Write-Host "==> Starting Ollama ($OllamaBin)..."
  $ollamaLog = "$env:TEMP\ollama.log"
  Start-Process -FilePath $OllamaBin -ArgumentList "serve" -WindowStyle Hidden -RedirectStandardOutput $ollamaLog -RedirectStandardError "$env:TEMP\ollama-err.log"
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
# Pre-warm the default model on GPU so the first request is instant.
Write-Host "==> Pre-loading model on GPU..."
python -c "from backends.ollama import OllamaBackend; import registry, hardware; b=OllamaBackend(); m=registry.TASKS['reasoning'].pick_model(hardware.memory_budget_gb()); b.ensure_model(m); b.warm_model(m); print(f'  {m} loaded')"
if ($LASTEXITCODE -ne 0) { Write-Host "WARN: Model pre-load failed; first request will be slower." }
Write-Host "==> Launching UI at $UiUrl  (Ctrl+C to stop)"
streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
