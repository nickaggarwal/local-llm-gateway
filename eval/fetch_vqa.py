"""Build the vision eval set from a public VQA dataset (default: VQAv2).

Uses the HuggingFace datasets-server REST API (no `datasets` package needed),
downloads a small, deterministic image subset locally, and writes
datasets/vision.json with {image, question, accept:[answers]}.

Run from the project root:  python eval/fetch_vqa.py
Source: lmms-lab/VQAv2 (validation) — images are COCO (CC BY 4.0).
"""

import json
import os
import re

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(HERE, "images", "vqa")
ROWS_API = "https://datasets-server.huggingface.co/rows"

DATASET = "lmms-lab/VQAv2"
CONFIG = "default"
SPLIT = "validation"
SCAN = 100     # how many rows to scan
WANT = 15      # how many to keep


def fetch_rows(offset: int, length: int) -> list[dict]:
    r = requests.get(ROWS_API, params={
        "dataset": DATASET, "config": CONFIG, "split": SPLIT,
        "offset": offset, "length": length}, timeout=60)
    r.raise_for_status()
    return [x["row"] for x in r.json().get("rows", [])]


def _counts(row: dict) -> dict[str, int]:
    c: dict[str, int] = {}
    for a in row.get("answers", []):
        k = (a.get("answer") or "").strip().lower()
        if k:
            c[k] = c.get(k, 0) + 1
    return c


def accept_list(row: dict) -> list[str]:
    """Keep only answers ≥3 of 10 humans gave — drops one-off noise."""
    c = _counts(row)
    mc = (row.get("multiple_choice_answer") or "").strip().lower()
    out = sorted([a for a, n in c.items() if n >= 3], key=lambda a: -c[a])
    if mc and mc not in out:
        out.insert(0, mc)
    return out


def majority_agree(row: dict) -> int:
    mc = (row.get("multiple_choice_answer") or "").strip().lower()
    return _counts(row).get(mc, 0)


def good(row: dict) -> bool:
    mc = (row.get("multiple_choice_answer") or "").strip().lower()
    if not mc or len(mc.split()) > 3 or mc == "unanswerable":
        return False
    if majority_agree(row) < 6:  # clear gold answer
        return False
    acc = accept_list(row)
    if "yes" in acc and "no" in acc:  # contradictory -> not discriminating
        return False
    return True


def build():
    os.makedirs(IMG_DIR, exist_ok=True)
    rows = fetch_rows(0, SCAN)
    kept = []
    for i, row in enumerate(rows):
        if len(kept) >= WANT:
            break
        if not good(row):
            continue
        src = row["image"]["src"]
        ext = ".jpg"
        fname = f"vqa_{i:03d}{ext}"
        dst = os.path.join(IMG_DIR, fname)
        try:
            img = requests.get(src, timeout=60)
            img.raise_for_status()
            with open(dst, "wb") as f:
                f.write(img.content)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {i}: image download failed ({e})")
            continue
        kept.append({
            "image": f"images/vqa/{fname}",
            "question": row["question"].strip() + " Answer in a single word or short phrase.",
            "accept": accept_list(row),
            "_source": f"{DATASET} qid={row.get('question_id')}",
        })
        print(f"  + {fname}: {row['question']!r} -> {kept[-1]['accept'][:4]}")

    json.dump(kept, open(os.path.join(HERE, "datasets", "vision.json"), "w"), indent=2)
    print(f"\nWrote {len(kept)} examples to datasets/vision.json")


if __name__ == "__main__":
    build()
