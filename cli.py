"""CLI for the local LLM gateway.

Examples:
    python cli.py tasks
    python cli.py run ocr --image receipt.png
    python cli.py run chat "Explain RAG in one sentence."
    python cli.py run code "Write a Python function to reverse a linked list."
    python cli.py run embed "hello world"
"""

import argparse
import json
import os
import sys

import gateway
import hardware
from backends import backend_names, get_backend
from registry import TASKS


def _progress(msg: str) -> None:
    # Ollama pull lines are JSON; print a compact status when possible.
    try:
        obj = json.loads(msg)
        status = obj.get("status", "")
        if "completed" in obj and "total" in obj and obj["total"]:
            pct = obj["completed"] / obj["total"] * 100
            print(f"\r  {status}: {pct:4.1f}%", end="", file=sys.stderr, flush=True)
        elif status:
            print(f"\r  {status}", end="", file=sys.stderr, flush=True)
    except (json.JSONDecodeError, TypeError):
        print(msg, file=sys.stderr)


def cmd_tasks(args) -> int:
    hw = hardware.describe()
    parts = [f"RAM {hw['ram_gb']:.0f} GB"]
    for g in hw["gpus"]:
        label = g["vendor"].upper()
        parts.append(f"{label} GPU {g['vram_gb']:.0f} GB VRAM" if g["vram_gb"] else f"{label} GPU")
    for vendor, present in hw["npus"].items():
        if present:
            parts.append(f"{vendor.capitalize()} NPU")
    print(f"Hardware: {', '.join(parts)}  •  sizing budget: {hw['budget_gb']:.0f} GB\n")
    print(f"{'TASK':<12}{'KIND':<9}{'BACKEND':<10}{'SELECTED MODEL':<34}DESCRIPTION")
    for name, task in TASKS.items():
        be = get_backend(args.backend, task)
        try:
            model = be.resolve_model(task, None, hw["budget_gb"])
        except Exception as e:  # noqa: BLE001
            model = f"(unavailable: {e})"[:33]
        print(f"{name:<12}{task.kind:<9}{be.name:<10}{model:<34}{task.description}")
    return 0


def cmd_run(args) -> int:
    try:
        result = gateway.run(
            args.task,
            prompt=args.prompt,
            image_path=args.image,
            model=args.model,
            backend=args.backend,
            on_progress=_progress,
        )
    except Exception as e:  # noqa: BLE001
        print(f"\nerror: {e}", file=sys.stderr)
        return 1

    print("", file=sys.stderr)  # newline after progress
    if "embedding" in result:
        emb = result["embedding"]
        print(f"[{result['backend']}/{result['model']}] {len(emb)}-dim embedding: {emb[:5]}...")
    else:
        print(result["text"])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Local LLM gateway: best model per task, run locally.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    backend_choices = ["auto", *backend_names()]
    backend_default = os.environ.get("LLM_GATEWAY_BACKEND", "auto")

    tasks_p = sub.add_parser("tasks", help="List task types and the model each will use")
    tasks_p.add_argument("--backend", choices=backend_choices, default=backend_default,
                         help="Inference backend (default: auto)")

    run_p = sub.add_parser("run", help="Run a task")
    run_p.add_argument("task", choices=list(TASKS), help="Task type")
    run_p.add_argument("prompt", nargs="?", default=None, help="Prompt / input text")
    run_p.add_argument("--image", help="Image path (for ocr/vision tasks)")
    run_p.add_argument("--model", help="Force a specific model tag/id")
    run_p.add_argument("--backend", choices=backend_choices, default=backend_default,
                       help="Inference backend (default: auto)")

    args = parser.parse_args()
    if args.cmd == "tasks":
        return cmd_tasks(args)
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
