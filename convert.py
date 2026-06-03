"""Prepare (convert / download) NPU models ahead of first use.

Detects the NPU on this machine (or use --backend) and, for each task, resolves
the backend's model and prepares it: downloads a pre-compiled equivalent, or
runs the vendor conversion pipeline into the backend's cache. The same step runs
automatically on first use via each backend's `ensure_model`; this script just
lets you do it up front.

Examples:
    python convert.py chat                  # prepare 'chat' for the detected NPU
    python convert.py chat code             # prepare several tasks
    python convert.py --all                 # prepare every task the backend supports
    python convert.py ocr --backend qualcomm
"""

import argparse
import sys

import hardware
from backends import BackendUnavailable, get_backend
from registry import TASKS


def _detect_backend() -> str | None:
    if hardware.has_qualcomm_npu():
        return "qualcomm"
    if hardware.has_intel_npu():
        return "intel-npu"
    if hardware.has_amd_npu():
        return "amd-npu"
    return None


def _progress(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare NPU models (convert or download) before use.")
    parser.add_argument("tasks", nargs="*", help="Task(s) to prepare (e.g. chat code ocr)")
    parser.add_argument("--all", action="store_true", help="Prepare every task the backend supports")
    parser.add_argument(
        "--auto", action="store_true",
        help="Startup mode: prepare all tasks for the detected NPU; exit cleanly if there is none.",
    )
    parser.add_argument(
        "--backend", default="auto", choices=["auto", "qualcomm", "intel-npu", "amd-npu"],
        help="Target NPU backend (default: auto-detect)",
    )
    parser.add_argument("--model", help="Override the model id to prepare (single task only)")
    args = parser.parse_args()

    # --auto is the launcher hook: no NPU is not an error, just nothing to do.
    if args.auto:
        backend_name = _detect_backend()
        if backend_name is None:
            print("==> No NPU detected; skipping model preparation.")
            return 0
    elif args.backend != "auto":
        backend_name = args.backend
    else:
        backend_name = _detect_backend()
        if backend_name is None:
            raise SystemExit(
                "no NPU detected on this machine. Pass --backend qualcomm|intel-npu|amd-npu "
                "to prepare models for a specific target anyway."
            )

    backend = get_backend(backend_name)
    print(f"==> Target backend: {backend_name}")

    if args.all or args.auto:
        task_names = [name for name, t in TASKS.items() if backend.supports_task(t)]
    else:
        task_names = args.tasks
    if not task_names:
        parser.error("specify one or more tasks, or pass --all")
    if args.model and len(task_names) != 1:
        parser.error("--model can only be used with a single task")

    budget = hardware.memory_budget_gb()
    failures = 0
    for name in task_names:
        task = TASKS.get(name)
        if task is None:
            print(f"  ! unknown task '{name}', skipping", file=sys.stderr)
            failures += 1
            continue
        try:
            model = backend.resolve_model(task, args.model, budget)
            print(f"==> {name}: preparing '{model}' ...")
            backend.ensure_model(model, on_progress=_progress)
            print(f"==> {name}: ready ({model})")
        except BackendUnavailable as e:
            print(f"  ! {name}: {e}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
