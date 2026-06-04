# Eval: per-RAM-tier model bake-off

This pipeline answers a concrete question for every task type: **at a given laptop
RAM budget, is a smaller full-precision model or a larger quantized model more
accurate?** It runs a ladder of models (different sizes and quantizations) over a
scored dataset and reports the best model that *fits* each RAM tier (16/24/32/48 GB).

## Layout

```
eval/
  candidates.py     model ladders per task + footprints + RAM-fit logic
  scorers.py        objective scoring function per task (returns 0..1)
  datasets/         ground-truth datasets (one JSON per task)
  images/           OCR test images (generated) + images/vqa/ (public VQA)
  gen_images.py     regenerates OCR images + ocr.json deterministically
  fetch_vqa.py      builds vision.json from a public VQA dataset (VQAv2)
  run_eval.py       the pipeline
```

## How each task is scored

| Task | Dataset | Metric |
|------|---------|--------|
| `ocr` | 6 generated images (invoice, paragraph, code, table, receipt, address) | 0.5·(1−char-error-rate) + 0.5·key-token coverage |
| `vision` | 15 real images from **VQAv2** (public) + short answers | answer word-boundary-matches an accepted human answer |
| `reasoning` | math / multi-step word problems with exact answers | answer word-boundary-matches the expected value |
| `code` | function specs + assert tests | functional correctness (asserts pass) |
| `summary` | passages with required key facts + length cap | key-fact coverage − length penalty |
| `embed` | query + 1 relevant doc + 3 distractors | relevant doc ranks #1 by cosine similarity |

## Running

From the project root with the venv active and Ollama running:

```bash
python eval/gen_images.py                 # once: build OCR images + ocr.json
python eval/fetch_vqa.py                    # once: build vision.json from public VQAv2
python eval/run_eval.py --task embed       # one task (uses already-downloaded models)
python eval/run_eval.py --task all          # every task
python eval/run_eval.py --task ocr --pull   # download any missing models in the ladder
python eval/run_eval.py --task all --limit 2  # quick smoke test
```

Models not yet downloaded are skipped unless you pass `--pull`. The per-tier winner
is computed only over models that actually fit that tier's RAM budget.

## Results (Apple Silicon 48 GB, Ollama 0.30.2)

**OCR** — 6 generated images, composite of (1−char-error-rate) and key-token coverage:

| Model | Quant | Weights | OCR score | s/ex |
|-------|-------|--------:|----------:|-----:|
| qwen2.5vl:3b | q8_0 | 3.5 GB | **0.987** | 9.6 |
| qwen2.5vl:7b | q4_K_M | 6.0 GB | **0.983** | 12.1 |
| qwen2.5vl:7b | q8_0 | 9.4 GB | 0.969 | 12.8 |
| qwen2.5vl:7b | fp16 | 17 GB | 0.969 | 16.5 |
| minicpm-v | q4 | 5.5 GB | 0.767 | 6.2 |
| qwen2.5vl:32b | q4_K_M | 21 GB | — | fails to load CLIP projector on 0.30.2 |

**Vision (VQAv2, 15 real images)** — answer matches an accepted human answer:

| Model | Quant | Vision score | s/ex |
|-------|-------|-------------:|-----:|
| minicpm-v | q4 | **0.867** | 1.6 |
| qwen2.5vl:3b | q8_0 | 0.800 | 3.2 |
| qwen2.5vl:7b | q4_K_M | 0.800 | 4.0 |
| qwen2.5vl:7b | q8_0 | 0.800 | 3.9 |
| qwen2.5vl:7b | fp16 | 0.800 | 4.4 |

**Text tasks** (reasoning 18 Q, code 12 problems, summary 4 passages, embed 15 retrievals):

| Model | reasoning | code | summary |
|-------|----------:|-----:|--------:|
| qwen2.5:3b / coder:3b | 0.556 | 0.917 | 0.750 |
| qwen2.5:7b / coder:7b | 0.722 | 1.000 | 0.750 |
| qwen2.5:14b / coder:14b | **1.000** | 1.000 | 0.875 |
| qwen2.5:32b / coder:32b | 0.944 | 1.000 | **1.000** |
| llama3.1:8b | 0.444 | — | 0.875 |
| deepseek-coder-v2:16b | — | 1.000 | — |

embed: nomic **1.000** · bge-m3 0.933 · mxbai 0.867.

**Findings:**

1. **Best model depends on the sub-task.** Qwen2.5-VL wins OCR (~0.98 vs MiniCPM-V's
   0.77); MiniCPM-V edges general VQA (0.867 vs 0.800) and is ~2.5× faster. Pick per use case.
2. **Family strength flips by task.** On math **reasoning**, Qwen2.5 dominates and Llama
   3.1 8B is weak (0.44); on **summary**, Llama 3.1 8B (0.875) *beats* Qwen2.5 7B (0.75).
   On **code**, DeepSeek-Coder-V2 matches Qwen and is faster. This is why the registry
   picks a different family per task.
3. **Reasoning saturates at 14B** (32B no gain), **code at 7B**, so the registry caps
   there rather than always reaching for the biggest model.
2. **Quantization barely matters.** Across OCR and vision, q4 / q8 / fp16 of the same
   model score within noise (higher precision was if anything a hair lower). So for the
   small-full-precision vs big-quantized question: **q4 is the right choice** — higher
   precision just costs RAM and speed for no measurable accuracy gain.
3. **Size is saturated for these tasks.** 3B ≈ 7B, so the gateway tops OCR/vision out at
   7B; 32B adds cost with no measurable gain (and won't load on this Ollama build).

Caveats: 6 OCR + 15 VQA examples — small, so treat sub-point differences as noise; the
robust signals are the per-task family ranking and the quant plateau. Grow the datasets
(`gen_images.py`, `fetch_vqa.py`) to tighten the numbers.

## Tuning the fit model

`candidates.py` estimates runtime RAM as `weights·1.2 + overhead` (overhead 2.5 GB for
vision, 1.5 GB for text) and treats a model as fitting a laptop when that leaves ~5 GB
for the OS. Adjust `OS_HEADROOM_GB` / overhead there if your environment differs.

## Extending

- Add examples by editing the JSON in `datasets/` (or `gen_images.py` for image tasks).
- Add a model to a ladder in `candidates.py`.
- Add a new task: add a ladder entry, a dataset, and a scorer in `scorers.py`.
