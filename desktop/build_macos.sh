#!/usr/bin/env bash
#
# Build the installable macOS app:
#   dist/Local LLM Gateway.app   (the app)
#   dist/LocalLLMGateway.dmg     (drag-to-Applications installer)
#
# Usage: ./desktop/build_macos.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"

if [ ! -d .venv-desktop ]; then
  echo "==> Creating build virtualenv (.venv-desktop)..."
  "$PYTHON" -m venv .venv-desktop
fi
. .venv-desktop/bin/activate

echo "==> Installing build dependencies..."
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt -r requirements-desktop.txt

echo "==> Freezing app with PyInstaller..."
pyinstaller --noconfirm --clean desktop/gateway_app.spec

APP="dist/Local LLM Gateway.app"
DMG="dist/LocalLLMGateway.dmg"

# Ad-hoc signature so Gatekeeper doesn't refuse outright on the build machine.
# For distribution, re-sign with a Developer ID cert and notarize:
#   codesign --force --deep --options runtime -s "Developer ID Application: ..." "$APP"
#   xcrun notarytool submit "$DMG" --keychain-profile ... --wait
echo "==> Signing (ad-hoc)..."
codesign --force --deep -s - "$APP"

echo "==> Building DMG..."
rm -f "$DMG"
hdiutil create -volname "Local LLM Gateway" -srcfolder "$APP" -ov -format UDZO "$DMG"

echo
echo "Built:     $APP"
echo "Installer: $DMG"
