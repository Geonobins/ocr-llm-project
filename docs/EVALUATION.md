# Evaluation method

## 1. Dataset

600 invoice images with one JSON ground-truth file each.

| Subset | Invoices | Origin | Formats |
|---|---|---|---|
| Real | 1–15, 316–600 (300) | publicly available invoice templates and scanned invoices, labelled by hand | WebP, PNG, JPG; 1131×1600 to 2481×3509 px |
| Synthetic | 16–315 (300) | `generate_dataset.py` – Pillow renders a company name, invoice number, date, optional line items and noise lines, total | PNG 800×600 or 800×1000 |

The synthetic half is deliberately easy (clean white background, one font, five possible dates). It serves as a sanity floor: a pipeline that cannot reach ~100% on it is broken. The real half is where the architectures are actually tested, so every accuracy metric is reported per subset as well as overall.

Ground truth is a JSON object with the three fields in the format shown on the invoice. The `total` field means the gross total (the amount due). The original labels of invoices 316–600 mixed net and gross totals; `evaluation/relabel_totals.py` corrected 26 of them to the gross total, accepting a value only when net + VAT = gross on the OCR output (logs: `results/relabel_totals_log_pass1.csv` and `results/relabel_totals_log.csv`, originals: `ground_truth_backup_20261018/`). `dataset.load_documents` orders documents by their numeric index (1, 2, 3, … 600), never by string order, and refuses to run if any ground-truth file is missing.

## 2. Scoring (`evaluation/scorer.py`)

Each field of each document is compared with its ground truth **after normalisation** (see `docs/ARCHITECTURE.md` §2.4) and assigned one outcome:

| Truth | Prediction | Outcome |
|---|---|---|
| null | null | TN |
| null | value | FP |
| value | null | FN |
| value | matching value | TP |
| value | wrong value | FP and FN (a wrong value is both a false positive and a miss) |

"Matching" means:

- **invoice number**: canonical strings equal, or their digit sequences equal (≥ 3 digits), or RapidFuzz `ratio ≥ 90` on the canonical strings – this tolerates OCR confusions such as `l`/`1` or `O`/`0` in one character of a long number.
- **date**: both parse and denote the same calendar date (day/month swap tolerated); if either cannot be parsed, the strings are compared after whitespace/case normalisation with the same fuzzy threshold on their digit sequences.
- **total**: both parse as amounts and differ by less than 0.005; otherwise the fuzzy threshold on digit sequences.

From the outcomes:

```
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 · precision · recall / (precision + recall)
```

computed per field and micro-averaged over all fields ("overall"). Two further numbers are reported: field accuracy `(TP + TN) / fields` and document accuracy (share of documents with all three fields correct).

The same scorer, with the same thresholds, is applied to every configuration. This is what makes the A/B accuracy comparison valid.

## 3. Performance metrics (`pipeline/metrics.py`)

| Metric | Definition |
|---|---|
| Wall time | seconds from submitting the first document to receiving the last result, models already loaded and warmed up |
| Throughput | documents ÷ wall time × 60 (docs/min) |
| Service time | compute spent on one document: OCR + embedding share + LLM share |
| Latency | time from submission to result, including queueing (equals service time in A) |
| p50 / p95 latency | percentiles over documents |
| Mean OCR / embedding / LLM time | per uncached document |
| Peak RAM | maximum resident set size of the process tree during the run, sampled at 4 Hz |
| Pipeline GPU peak | PyTorch's peak allocated GPU memory in the run – the pipeline's own use |
| Whole-GPU peak | maximum GPU memory in use on the card (NVML) during the run, including other programs; the baseline before the run is recorded separately |
| Straight-through rate | share of invoices without a review flag (accepted automatically) |
| Accuracy auto-accepted | share of accepted invoices with all three fields correct |
| Wrong invoices caught | share of invoices with at least one wrong field that the review flag sent to a person |
| Cost | wall time × €0.35 per GPU-hour, per document and per 100 documents |

Model loading and warm-up are timed separately (`model_load_time_s`) and excluded from wall time, because they are paid once per process, not per document.

## 4. Experiment design

Every configuration is run **three times** on the full dataset; tables report mean ± standard deviation. The cache is cleared before each cold run.

| Configuration | What it isolates |
|---|---|
| `A` | baseline |
| `B` | all optimisations, cold cache |
| `B-warm` | the same run repeated with a warm cache |
| `B-nocache` | overhead of the cache lookup on a cold pass |
| `B-allfull` | B without the quantized model (every document fp32) |
| `B-allfast` | B with only the quantized model (every document 4-bit) |
| `B-fp16` | half precision instead of quantization |
| `B-8bit` | LLM.int8 instead of NF4 |
| `B-1worker` | B without OCR parallelism |
| `B-6workers` | B with twice the default OCR parallelism |
| `B-nobatch` | B without GPU batching |
| `B-tuned` | B with thresholds chosen by the routing analysis of the same benchmark (run last, after `B-allfast` and `B-allfull`) |

Comparing `B` with each ablation attributes the overall gain to its components; comparing `B-allfull`, `B-allfast`, `B-fp16` and `B-8bit` gives the precision trade-off (latency, memory, F1) on this GPU.

## 5. Routing-threshold analysis (`evaluation/routing_analysis.py`)

1. Take the per-document results of `B-allfast` and `B-allfull`, averaged over their runs. For every document this gives: token count, OCR confidence, fields correct under FAST, fields correct under FULL, LLM time under each.
2. For every `(max_tokens_fast, min_confidence_fast)` on a grid (tokens 50–800 step 25; confidence 70–95) simulate the rule, and compute the share of documents sent FAST, the resulting field accuracy and the resulting mean LLM time.
3. Keep the policies whose accuracy is within 0.5 percentage points of all-FULL; among them choose the one with the largest FAST share.
4. Report accuracy per token-count bin and per confidence bin, so the thesis can show *where* the quantized model starts to lose accuracy.

Output, in the benchmark folder: `routing_analysis.md`, `routing_grid.csv`, `routing_documents.csv` and `routing_thresholds.json`. With `--configs all` the runner performs this analysis automatically before `B-tuned` and uses the chosen thresholds for it.

## 6. Reproducibility

- Every run writes to a new folder `outputs/<date>_<time>_<label>/` with `run_info.json` (command, pipeline settings, hardware, library versions), `log.txt`, per-run `documents.csv` and `summary.json`, the report tables and figures, and its own cache file.
- Models are loaded offline by default, so a run cannot be disturbed by network access.
- Runs disturbed by other programs are recognised by OCR time per invoice (about 0.55–0.63 s with 3 workers on an idle machine), moved to `discarded/` with the reason in the folder name, and repeated with `--out <folder> --skip-existing`.

- Hardware, driver and library versions are written into every `summary.json`.
- Generation is deterministic (`do_sample=False`), so accuracy differences between runs come only from timing-independent code paths; timing differences come from the OS scheduler and GPU clocks.
- The random seed of the synthetic generator is not fixed in `generate_dataset.py`; the generated files are therefore committed with the project and must not be regenerated.
- Nothing in the pipeline depends on the file name beyond the numeric index used to pair images with ground truth.

## 7. Additional experiments

Three experiments answer questions the main benchmark cannot. Each writes its own folder under `outputs/` with `run_info.json`, `log.txt`, per-invoice CSV files and a `summary.md`.

### 7.1 Latency with invoices arriving one at a time (`evaluation/latency_experiment.py`)

The main benchmark submits all 600 invoices at once, so B's latency there is mostly queueing. This experiment measures latency the way a user would see it.

- 60 invoices spread evenly over the dataset (`spread_indices`: 31 real, 29 synthetic).
- Arrivals follow a Poisson process (random exponential gaps) at an average of 30, 48, 120 and 240 invoices per minute. A handles about 53 per minute, B about 300, so the rates cover "both idle", "A near its limit", "A overloaded" and "B near its limit".
- Each run shuffles the invoices and draws the arrival times with a fixed seed (`1000 × run + rate`); A and both B variants see exactly the same arrivals. 3 runs per rate.
- Latency = time from an invoice's arrival to its result. Reported: median, p95 and maximum per run, then mean ± standard deviation over runs.
- Variants: `A` (one invoice at a time in arrival order), `B` (as benchmarked: a GPU batch starts once 8 invoices are read, or when no more invoices are coming), `B-greedy` (when the GPU is free it takes every invoice already read, up to 8 – dynamic batching, `max_batch_wait_s = 0`).
- The B result cache is off, so every invoice is computed.

### 7.2 OCR engines: Tesseract vs EasyOCR (`evaluation/ocr_comparison.py`)

- Both engines read all 600 invoices once, with the model loaded before timing starts. Tesseract runs on the CPU as in the pipeline; EasyOCR runs on the GPU, its normal mode.
- EasyOCR's text boxes go through the same row-by-row layout as Tesseract's (`_rows_from_segments`), so the comparison is about recognition, not layout.
- Accuracy is measured twice: (1) whether each ground-truth value appears in the OCR text at all (`field_found`: canonical invoice number, the total as a parsed amount, the date by fuzzy partial match ≥ 90) – an upper bound for any extractor; (2) the F1 of the full Architecture A downstream (MiniLM filter, Flan-T5 fp32, total check) fed with each engine's text, scored by the same scorer. With Tesseract this reproduces A's benchmark F1, which checks the experiment.
- Speed: OCR seconds per invoice (mean, median, p95, real vs synthetic); EasyOCR's GPU memory.

### 7.3 Vision-language model baseline (`evaluation/vlm_baseline.py`)

- Model: `Qwen/Qwen2-VL-2B-Instruct`, loaded in 4-bit NF4 with bitsandbytes (the fp16 weights alone, 4.4 GB, do not fit the 4 GB GPU).
- The model sees only the image (no OCR) and is asked for the three fields as JSON. Answers pass through the same normalisation (`clean_prediction`) and scorer as A and B.
- Image resolution is capped by a maximum number of visual tokens (one token per 28 × 28 pixels); the cap is a parameter (`--tokens`) because it trades accuracy against memory and time.
- One invoice at a time; time per invoice, peak GPU memory, share of valid JSON answers and F1 are recorded. Out-of-memory errors are caught and counted.

### 7.4 Re-scoring stored runs

Every experiment stores its predictions per invoice, so a scorer change never requires re-running a model:

- main benchmark: `python -m evaluation.rescore --results <folder>` then `python -m evaluation.report --results <folder>`
- latency: `python -m evaluation.latency_experiment --rescore --out <folder>`
- OCR comparison: `python -m evaluation.ocr_comparison --rescore --out <folder>` (also recomputes whether each value appears in the stored OCR text)
- VLM: `python -m evaluation.vlm_baseline --rescore --out <folder>`

This was used once: the OCR comparison exposed that dates written without a space after the comma (`April 19,2024`) were parsed with the current year. `parse_date` now inserts the space first (unit test in `tests/test_normalize.py`), and all stored runs were re-scored. The main benchmark did not change, because its labels and predictions in that format always appeared together; EasyOCR's and the VLM's scores rose.

Run each VLM setting in its own process (`--tokens 512`, then `--tokens 1024`, …): when several settings ran in one process, the GPU peak of the later settings included the model of the earlier one. The probe in `outputs/_superseded/` has that fault; `outputs/2026-10-21_00-35-48_vlm_resolution_probe_20docs` is the corrected one.
