# Build the installable Windows app:
#   dist\LocalLLMGateway\                  (the app folder)
#   dist\LocalLLMGateway-Setup.exe         (installer, if Inno Setup is installed)
#   dist\LocalLLMGateway-win64.zip         (portable zip fallback otherwise)
#
# Usage: .\desktop\build_windows.ps1
# PowerShell 5.1 compatible.

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$python = "python"
if (Get-Command py -ErrorAction SilentlyContinue) { $python = "py" }

if (-not (Test-Path ".venv-desktop")) {
    Write-Host "==> Creating build virtualenv (.venv-desktop)..."
    & $python -m venv .venv-desktop
}
$venvPy = ".venv-desktop\Scripts\python.exe"

Write-Host "==> Installing build dependencies..."
& $venvPy -m pip install -q --upgrade pip
& $venvPy -m pip install -q -r requirements.txt -r requirements-desktop.txt

Write-Host "==> Freezing app with PyInstaller..."
& $venvPy -m PyInstaller --noconfirm --clean desktop\gateway_app.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed." }

# Inno Setup compiler: PATH first, then the default install locations.
$iscc = $null
$cmd = Get-Command ISCC.exe -ErrorAction SilentlyContinue
if ($cmd) { $iscc = $cmd.Source }
if (-not $iscc) {
    foreach ($p in @("$env:ProgramFiles(x86)\Inno Setup 6\ISCC.exe",
                     "$env:ProgramFiles\Inno Setup 6\ISCC.exe")) {
        if (Test-Path $p) { $iscc = $p; break }
    }
}

if ($iscc) {
    Write-Host "==> Building installer with Inno Setup..."
    & $iscc desktop\installer.iss
    if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed." }
    Write-Host ""
    Write-Host "Installer: dist\LocalLLMGateway-Setup.exe"
} else {
    Write-Host "==> Inno Setup not found (https://jrsoftware.org/isinfo.php); building portable zip instead..."
    if (Test-Path "dist\LocalLLMGateway-win64.zip") { Remove-Item "dist\LocalLLMGateway-win64.zip" }
    Compress-Archive -Path "dist\LocalLLMGateway" -DestinationPath "dist\LocalLLMGateway-win64.zip"
    Write-Host ""
    Write-Host "Portable build: dist\LocalLLMGateway-win64.zip"
}
