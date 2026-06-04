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


TOOL_NAMES = {t["function"]["name"] for t in TOOLS}


def _parse_content_tool_calls(content: str) -> list[dict] | None:
    """Some models return tool calls as JSON text in content instead of using
    the ``tool_calls`` field.  Try to extract them."""
    text = content.strip()
    # Strip markdown code fences if present.
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    # Single tool call as {"name": ..., "arguments": ...}
    if isinstance(obj, dict) and obj.get("name") in TOOL_NAMES:
        return [{"function": {"name": obj["name"], "arguments": obj.get("arguments", {})}}]
    # List of tool calls
    if isinstance(obj, list) and all(isinstance(o, dict) and o.get("name") in TOOL_NAMES for o in obj):
        return [{"function": {"name": o["name"], "arguments": o.get("arguments", {})}} for o in obj]
    return None


def _execute_tool(sandbox, fn_name: str, fn_args: dict, on_progress) -> tuple[dict, list[str]]:
    """Execute a single tool call, return (result_dict, new_files)."""
    new_files: list[str] = []
    if fn_name == "run_python":
        result = sandbox.run_code(fn_args.get("code", ""))
        new_files = result.pop("files", [])
        if new_files:
            result["created_files"] = new_files
    elif fn_name == "install_package":
        result = sandbox.install_package(fn_args.get("package", ""))
    else:
        result = {"error": f"Unknown tool: {fn_name}"}
    if on_progress:
        on_progress(f"  tool {fn_name} -> exit {result.get('exit_code', '?')}")
    return result, new_files


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

            # Ollama may return tool calls in the tool_calls field (standard)
            # or as JSON text in the content field (some models do this).
            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                content = msg.get("content", "")
                parsed = _parse_content_tool_calls(content)
                if parsed:
                    tool_calls = parsed
                else:
                    return {
                        "text": content,
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

                result, new_files = _execute_tool(sandbox, fn_name, fn_args, on_progress)
                all_files.extend(new_files)

                messages.append({
                    "role": "tool",
                    "content": json.dumps(result),
                })

    return {
        "text": "Agent reached the maximum number of turns without finishing.",
        "files": [str(sandbox.workspace / f) for f in all_files],
    }
