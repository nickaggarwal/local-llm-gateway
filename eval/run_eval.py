"""Per-RAM-tier model bake-off across all task types.

For each task it runs a ladder of models (different sizes and quantizations)
over a scored dataset, then reports the most accurate model that fits each
laptop RAM tier (16/24/32/48 GB) — answering whether a smaller full-precision
or a larger quantized model wins at each budget.

Usage (from project root, with the venv active):
    python eval/run_eval.py --task ocr
    python eval/run_eval.py --task all
    python eval/run_eval.py --task chat --pull        # download missing models
    python eval/run_eval.py --task all --limit 2      # quick smoke test
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gateway  # noqa: E402

from candidates import LADDERS, TIERS, fits, runtime_gb  # noqa: E402
from scorers import SCORERS, cosine  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARIZE_PROMPT = "Summarize the following text in {n} words or fewer. Output only the summary.\n\n{text}"


def load_dataset(task: str) -> list[dict]:
    path = os.path.join(HERE, "datasets", f"{task}.json")
    return json.load(open(path))


def model_present(model: str, have: set[str]) -> bool:
    return model in have or f"{model}:latest" in have


def run_example(task: str, kind: str, model: str, ex: dict) -> float:
    if kind == "vision":
        img = os.path.join(HERE, ex["image"])
        prompt = gateway.OCR_PROMPT if task == "ocr" else ex["question"]
        out = gateway._generate(model, prompt, image_path=img)
        return SCORERS[task](out, ex)
    if kind == "text":
        if task == "summarize":
            prompt = SUMMARIZE_PROMPT.format(n=ex.get("max_words", 60), text=ex["text"])
        else:
            prompt = ex["prompt"]
        out = gateway._generate(model, prompt)
        return SCORERS[task](out, ex)
    if kind == "embed":
        q = gateway._embed(model, ex["query"])["embedding"]
        rel = gateway._embed(model, ex["relevant"])["embedding"]
        rel_sim = cosine(q, rel)
        dist_sims = [cosine(q, gateway._embed(model, d)["embedding"]) for d in ex["distractors"]]
        return 1.0 if rel_sim > max(dist_sims) else 0.0
    raise ValueError(kind)


def eval_task(task: str, pull: bool, limit: int | None) -> None:
    kind, ladder = LADDERS[task]
    dataset = load_dataset(task)
    if limit:
        dataset = dataset[:limit]
    have = gateway.local_models()

    print(f"\n{'='*70}\nTASK: {task}  (kind={kind}, {len(dataset)} examples)\n{'='*70}")
    results = []
    for cand in ladder:
        if not model_present(cand.model, have):
            if pull:
                print(f"  pulling {cand.model} ...")
                try:
                    gateway.ensure_model(cand.model)
                    have = gateway.local_models()
                except Exception as e:  # noqa: BLE001
                    print(f"  !! could not pull {cand.model}: {e} (skipping)")
                    continue
            else:
                print(f"  skip {cand.model} (not downloaded; use --pull)")
                continue
        t0 = time.time()
        scores = []
        for ex in dataset:
            try:
                scores.append(run_example(task, kind, cand.model, ex))
            except Exception as e:  # noqa: BLE001
                print(f"    !! {cand.model} errored: {e}")
                scores.append(0.0)
        dt = (time.time() - t0) / max(1, len(dataset))
        acc = sum(scores) / len(scores)
        results.append({"model": cand.model, "weights": cand.weights_gb,
                        "runtime": runtime_gb(cand.weights_gb, kind),
                        "score": round(acc, 3), "sec_per_ex": round(dt, 1)})
        print(f"  {cand.model:<24} score={acc:5.3f}  {dt:5.1f}s/ex")

    if not results:
        print("  (no models available — run with --pull or download them first)")
        return

    print(f"\n  {'model':<24}{'wt':>5}{'rtRAM':>7}{'score':>7}{'s/ex':>7}")
    for r in results:
        print(f"  {r['model']:<24}{r['weights']:>5}{r['runtime']:>7}{r['score']:>7}{r['sec_per_ex']:>7}")

    print("\n  PER-RAM-TIER WINNER (best score that fits):")
    for ram in TIERS:
        fitting = [r for r in results if fits(r["weights"], kind, ram)]
        if not fitting:
            print(f"    {ram}GB: (none fit)")
            continue
        best = sorted(fitting, key=lambda r: (-r["score"], r["sec_per_ex"]))[0]
        print(f"    {ram}GB: {best['model']}  (score={best['score']}, {best['sec_per_ex']}s/ex)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all", help="task name or 'all'")
    ap.add_argument("--pull", action="store_true", help="download missing models")
    ap.add_argument("--limit", type=int, default=None, help="cap examples per task")
    args = ap.parse_args()

    tasks = list(LADDERS) if args.task == "all" else [args.task]
    for t in tasks:
        if t not in LADDERS:
            print(f"unknown task: {t}")
            return 1
        eval_task(t, args.pull, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
