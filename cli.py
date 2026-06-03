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
import sys

import gateway
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


def cmd_tasks(_args) -> int:
    ram = gateway.available_ram_gb()
    print(f"Detected RAM: {ram:.0f} GB\n")
    print(f"{'TASK':<12}{'KIND':<9}{'SELECTED MODEL':<22}DESCRIPTION")
    for name, task in TASKS.items():
        model = task.pick_model(ram)
        print(f"{name:<12}{task.kind:<9}{model:<22}{task.description}")
    return 0


def cmd_run(args) -> int:
    try:
        result = gateway.run(
            args.task,
            prompt=args.prompt,
            image_path=args.image,
            model=args.model,
            on_progress=_progress,
        )
    except Exception as e:  # noqa: BLE001
        print(f"\nerror: {e}", file=sys.stderr)
        return 1

    print("", file=sys.stderr)  # newline after progress
    if "embedding" in result:
        emb = result["embedding"]
        print(f"[{result['model']}] {len(emb)}-dim embedding: {emb[:5]}...")
    else:
        print(result["text"])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Local LLM gateway: best model per task, run locally.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("tasks", help="List task types and the model each will use")

    run_p = sub.add_parser("run", help="Run a task")
    run_p.add_argument("task", choices=list(TASKS), help="Task type")
    run_p.add_argument("prompt", nargs="?", default=None, help="Prompt / input text")
    run_p.add_argument("--image", help="Image path (for ocr/vision tasks)")
    run_p.add_argument("--model", help="Force a specific Ollama model tag")

    args = parser.parse_args()
    if args.cmd == "tasks":
        return cmd_tasks(args)
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
