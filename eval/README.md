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
  images/           generated test images for ocr/vision
  gen_images.py     regenerates images + ocr.json/vision.json deterministically
  run_eval.py       the pipeline
```

## How each task is scored

| Task | Dataset | Metric |
|------|---------|--------|
| `ocr` | invoice + paragraph images with ground-truth text | 0.5·(1−char-error-rate) + 0.5·key-token coverage |
| `vision` | shape images + counting/color questions | answer matches accepted set (exact) |
| `chat` | factual questions with canonical answers | answer contains the canonical value |
| `code` | function specs + assert tests | functional correctness (asserts pass) |
| `summarize` | passages with required key facts + length cap | key-fact coverage − length penalty |
| `embed` | query + 1 relevant doc + 3 distractors | relevant doc ranks #1 by cosine similarity |

## Running

From the project root with the venv active and Ollama running:

```bash
python eval/gen_images.py                 # once: build images + ocr/vision datasets
python eval/run_eval.py --task embed       # one task (uses already-downloaded models)
python eval/run_eval.py --task all          # every task
python eval/run_eval.py --task ocr --pull   # download any missing models in the ladder
python eval/run_eval.py --task all --limit 2  # quick smoke test
```

Models not yet downloaded are skipped unless you pass `--pull`. The per-tier winner
is computed only over models that actually fit that tier's RAM budget.

## Results so far (OCR bake-off)

Run on an Apple Silicon 48 GB machine, Ollama 0.30.2, 2 OCR images (composite of
char-error-rate and key-token coverage; higher is better):

| Model | Quant | Weights | OCR score | s/example |
|-------|-------|--------:|----------:|----------:|
| qwen2.5vl:3b | q8_0 | 3.5 GB | **0.976** | 11.3 |
| qwen2.5vl:7b | q4_K_M | 6.0 GB | **0.976** | 13.8 |
| qwen2.5vl:7b | q8_0 | 9.4 GB | 0.934 | 16.1 |
| qwen2.5vl:7b | fp16 | 17 GB | 0.934 | 21.6 |
| minicpm-v | q4 | 5.5 GB | 0.625 | 9.0 |
| qwen2.5vl:32b | q4_K_M | 21 GB | — | fails to load CLIP projector on 0.30.2 |

**Findings:**

1. **Family matters most.** Qwen2.5-VL (~0.95) clearly beats MiniCPM-V (0.625) on
   OCR — consistent with 2026 research.
2. **Quantization barely matters for OCR.** q4 / q8 / fp16 of the same 7B model are
   within noise of each other (and the higher-precision runs were actually a hair
   lower here). So for the small-full-precision vs big-quantized question on OCR:
   neither helps — **q4 is the right choice**, higher precision just costs RAM and speed.
3. **Size is saturated.** 3B and 7B tie on this set, so the gateway tops OCR/vision
   out at 7B; 32B adds cost with no measurable OCR gain (and won't load here).

Caveat: only 2 OCR examples, so treat the small score differences as noise — the
robust signals are (1) and the family gap. Add more examples to `datasets/ocr.json`
to tighten the numbers. The `vision` shape-counting set is too easy (all VLMs score
1.0) and needs harder questions to discriminate.

## Tuning the fit model

`candidates.py` estimates runtime RAM as `weights·1.2 + overhead` (overhead 2.5 GB for
vision, 1.5 GB for text) and treats a model as fitting a laptop when that leaves ~5 GB
for the OS. Adjust `OS_HEADROOM_GB` / overhead there if your environment differs.

## Extending

- Add examples by editing the JSON in `datasets/` (or `gen_images.py` for image tasks).
- Add a model to a ladder in `candidates.py`.
- Add a new task: add a ladder entry, a dataset, and a scorer in `scorers.py`.
