# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the desktop app (macOS .app / Windows onedir).

Build via desktop/build_macos.sh or desktop/build_windows.ps1, or directly:
    pyinstaller --noconfirm --clean desktop/gateway_app.spec

Streamlit executes app.py from disk at runtime, so app.py ships as a data
file while the modules it imports (gateway, registry, ...) are bundled as
hidden imports — PyInstaller's analysis can't see imports inside a data file.
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

ROOT = Path(SPECPATH).parent  # noqa: F821 — SPECPATH is injected by PyInstaller

datas = [
    (str(ROOT / "app.py"), "."),
    (str(ROOT / "executors" / "pyodide_driver.mjs"), "executors"),
]
# Streamlit resolves its frontend assets and version from package data/metadata.
datas += copy_metadata("streamlit")
datas += collect_data_files("streamlit")

hiddenimports = [
    "gateway", "registry", "hardware", "agent", "sandbox",
    "backends", "backends.base", "backends.ollama", "backends.convert",
    "backends.qualcomm", "backends.intel_npu", "backends.amd_npu",
    "executors", "executors.base", "executors.docker_exec",
    "executors.subprocess_exec", "executors.wasm",
]
hiddenimports += collect_submodules("streamlit")

a = Analysis(
    [str(ROOT / "desktop" / "main.py")],
    pathex=[str(ROOT)],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["pytest", "tests", "eval"],
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="Local LLM Gateway" if sys.platform == "darwin" else "LocalLLMGateway",
    console=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="LocalLLMGateway")

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Local LLM Gateway.app",
        bundle_identifier="io.github.local-llm-gateway",
        info_plist={
            "NSHighResolutionCapable": True,
            "LSApplicationCategoryType": "public.app-category.developer-tools",
            "CFBundleShortVersionString": "1.0.0",
        },
    )
