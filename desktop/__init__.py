"""Installable desktop app (macOS/Windows) wrapping the Streamlit UI.

`launcher.py` holds the unit-testable pieces (port/process/Ollama discovery);
`main.py` is the window + process orchestration entry point that PyInstaller
freezes via `desktop/gateway_app.spec`.
"""
