"""Agentic tool-use loop: LLM plans and executes code via a Docker sandbox.

The loop sends the user's prompt (plus tool definitions) to Ollama's
``/api/chat`` endpoint. When the model returns tool calls the sandbox
executes them, feeds the results back, and the model continues until it
produces a final text answer.
"""

from __future__ import annotations

import json
import os

import requests

from sandbox import Sandbox, docker_available, ensure_image

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))
MAX_TURNS = 20

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute Python code in a sandboxed Docker container. "
                "pandas, openpyxl, and matplotlib are pre-installed. "
                "Save output files to /workspace/ — they will be returned to the user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "install_package",
            "description": "Install a Python pip package in the sandbox so subsequent run_python calls can import it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "package": {"type": "string", "description": "Package name, e.g. 'scipy' or 'scikit-learn'"},
                },
                "required": ["package"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are a helpful assistant that can execute Python code to accomplish tasks.\n"
    "You have a sandboxed Python environment with pandas, openpyxl, and matplotlib.\n"
    "You can install additional packages with install_package.\n"
    "Save any output files (Excel, CSV, images, etc.) to /workspace/.\n"
    "Think step by step, use tools to execute code, and when finished summarize "
    "what you did and list any files you created."
)


def run_agent_loop(
    model: str,
    prompt: str,
    on_progress: object = None,
) -> dict:
    """Drive the agent loop. Returns ``{"text": ..., "files": [host paths]}``."""
    if not docker_available():
        raise RuntimeError(
            "Docker is not running. The agent task requires Docker for sandboxed code execution.\n"
            "Install Docker: https://docs.docker.com/get-docker/"
        )
    ensure_image()

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    all_files: list[str] = []

    with Sandbox() as sandbox:
        for turn in range(MAX_TURNS):
            if on_progress:
                on_progress(f"Agent thinking (turn {turn + 1})...")

            resp = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "tools": TOOLS,
                    "stream": False,
                    "options": {"temperature": 0, "num_ctx": NUM_CTX},
                },
                timeout=600,
            )
            resp.raise_for_status()
            msg = resp.json()["message"]
            messages.append(msg)

            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                return {
                    "text": msg.get("content", ""),
                    "files": [str(sandbox.workspace / f) for f in all_files],
                }

            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                fn_args = tc["function"].get("arguments", {})
                if isinstance(fn_args, str):
                    try:
                        fn_args = json.loads(fn_args)
                    except (json.JSONDecodeError, TypeError):
                        fn_args = {}

                if fn_name == "run_python":
                    result = sandbox.run_code(fn_args.get("code", ""))
                    new_files = result.pop("files", [])
                    all_files.extend(new_files)
                    if new_files:
                        result["created_files"] = new_files
                elif fn_name == "install_package":
                    result = sandbox.install_package(fn_args.get("package", ""))
                else:
                    result = {"error": f"Unknown tool: {fn_name}"}

                if on_progress:
                    on_progress(f"  tool {fn_name} -> exit {result.get('exit_code', '?')}")

                messages.append({
                    "role": "tool",
                    "content": json.dumps(result),
                })

    return {
        "text": "Agent reached the maximum number of turns without finishing.",
        "files": [str(sandbox.workspace / f) for f in all_files],
    }
