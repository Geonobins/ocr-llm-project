# Results

Final benchmark of the current code: 600 invoices (300 real, 300 synthetic), 12 configurations, 3 runs each, mean ± standard deviation. Everything below comes from the benchmark folder:

```
outputs/2026-10-18_23-52-41_full_benchmark/
  summary.md          all tables
  figures/            all charts
  runs/<config>/run_<k>/documents.csv, summary.json
  routing_analysis.md, routing_thresholds.json
  run_info.json       command, settings, hardware, library versions
  discarded/          runs disturbed by other programs (see section 8)
```

Hardware: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB), 12 logical CPU cores, 15.7 GB RAM, Windows 11. Software: Python 3.12.2, PyTorch 2.5.1 + CUDA 12.1, transformers 5.7, bitsandbytes 0.50, sentence-transformers, Tesseract 5.4. Models loaded offline; laptop plugged in and kept awake.

The earlier benchmark in `results/` was measured before the accuracy upgrade (row-based OCR text and total reconciliation) and is kept only as a record of the previous version.

## 1. Architecture A vs Architecture B

| Metric | A (sequential) | B (optimized) | B vs A |
|---|---|---|---|
| Wall time for 600 invoices | 677.9 ± 7.4 s | 119.4 ± 4.7 s | 5.7× faster |
| Throughput | 53.1 ± 0.6 /min | 301.7 ± 11.8 /min | 5.7× |
| Service time per invoice | 1.119 s | 0.718 s | 1.6× lower |
| of which OCR | 0.588 s | 0.576 s | same |
| of which embedding filter | 0.017 s | 0.008 s | 2× lower |
| of which LLM | 0.514 s | 0.134 s | 3.8× lower |
| Latency of one invoice (A: p95) | 1.79 s | – | B is built for throughput |
| GPU memory used by the pipeline (PyTorch peak) | 1067 MB | 1996 MB | 1.9× higher |
| Peak RAM, all processes of the run | 1438 MB | 2012 MB | +40 % |
| Cost per 100 invoices (€0.35 per GPU-hour) | €0.0110 | €0.0019 | 5.8× lower |
| Same 600 invoices again (warm cache) | – | 0.59 ± 0.03 s | ≈ 1 ms per invoice |
| F1, all / real / synthetic | 0.941 / 0.892 / 0.989 | 0.941 / 0.894 / 0.987 | equal |

Reading the table:

- Throughput improves more (5.7×) than compute per invoice (1.6×). The difference is overlap: in B, three OCR workers and the GPU work on different invoices at the same time. A can only do one thing at a time.
- OCR costs the same in both; B does not make OCR faster, it runs three at once.
- The LLM stage is 3.8× cheaper per invoice in B because 24 prompts (8 invoices × 3 fields) go through the model in one call.
- B keeps two models resident and processes larger batches, so it uses more GPU memory and RAM. Both stay well inside the 4 GB GPU and 16 GB RAM.
- B's latency in this benchmark includes queueing, because all 600 invoices are submitted at once; throughput is the metric B is built to improve.

## 2. Where B's speed comes from

Each variant changes one thing in B.

| Configuration | Wall time | Throughput | Effect |
|---|---|---|---|
| B (3 OCR workers, batch 8, cache, routing) | 119.4 ± 4.7 s | 301.7 /min | reference |
| B-1worker (OCR not parallel) | 338.4 ± 5.1 s | 106.4 /min | parallel OCR is worth 2.8× |
| B-nobatch (batch size 1) | 248.8 ± 2.5 s | 144.7 /min | GPU batching is worth 2.1× |
| B-6workers | 102.9 ± 10.3 s | 352.1 /min | 16 % faster again, but higher variance and 2.6 GB RAM – the 12 CPU threads start to saturate |
| B-nocache (first pass) | 126.0 ± 2.6 s | 285.9 /min | the cache costs nothing measurable on a first pass |

Parallel OCR and GPU batching are the two optimizations that matter. Caching only pays off when invoices are processed again (0.59 s instead of ~120 s for all 600).

Differences of a few percent between configurations that should behave the same (B, B-nocache, B-allfull, B-allfast, B-tuned: 119–126 s) are run-to-run variation.

## 3. Model precision and quantization

The same pipeline with every invoice on one model, in four precisions:

| Precision | Wall time | LLM per invoice | Pipeline GPU peak | F1 |
|---|---|---|---|---|
| fp32 (B-allfull) | 126.1 ± 2.6 s | 0.115 s | 1645 MB | 0.941 |
| fp16 (B-fp16) | 123.5 ± 0.1 s | 0.079 s | 1049 MB | 0.941 |
| 4-bit NF4 (B-allfast) | 125.8 ± 0.8 s | 0.115 s | 1102 MB | 0.941 |
| 8-bit LLM.int8 (B-8bit) | 167.8 ± 1.6 s | 0.258 s | 1176 MB | 0.9385 ± 0.0003 |

- **4-bit saves memory, not time**: one third less GPU memory than fp32 at the same speed and accuracy.
- **fp16 is the best option on this GPU**: fastest LLM step, least memory, same accuracy. For a 248 M-parameter model, bitsandbytes quantization brings no advantage over half precision; its de-quantization cost cancels the saved memory traffic.
- **8-bit is the slowest** (2.2× slower LLM step than fp32) and slightly non-deterministic: its F1 varies between runs, while every other precision gives identical answers in all three runs.

## 4. Routing

- B as designed routes by token count and OCR confidence with the initial thresholds (300 tokens, confidence 85): 505 invoices to the 4-bit model, 95 to fp32.
- `evaluation/routing_analysis.py` compared the all-4-bit and all-fp32 runs invoice by invoice and searched a grid of thresholds for the largest 4-bit share that keeps field accuracy within 0.5 percentage points of all-fp32. Result: **525 tokens, confidence 70**, which sends 99.8 % of invoices to the 4-bit model.
- The two models disagree on only 10 invoices (5 each way); accuracy per token-count bin is equal within 0.5 percentage points everywhere.
- Validated with three fresh runs (B-tuned): 126.1 ± 3.7 s, F1 0.941, 599 of 600 invoices on the 4-bit route.

The router works; with two precisions of the same small model it has little to choose between. It becomes useful when the two routes differ more – a small model for simple invoices and a larger one for complex ones.

## 5. Accuracy

| Version | Arch. | F1 all | F1 real | F1 synthetic | Real: number / date / total | Real invoices fully correct |
|---|---|---|---|---|---|---|
| Before the accuracy upgrade | A | 0.847 | 0.704 | 0.989 | 0.93 / 0.86 / 0.32 | 24 % |
| | B | 0.850 | 0.712 | 0.987 | 0.93 / 0.86 / 0.34 | 26 % |
| Current code | A | 0.941 | 0.892 | 0.989 | 0.94 / 0.92 / 0.82 | 72 % |
| | B | 0.941 | 0.894 | 0.987 | 0.94 / 0.92 / 0.83 | 72 % |

Per field over all 600 invoices (B): invoice number 0.967, date 0.960, total 0.896. Precision equals recall because the model almost never answers `null`, so every miss is also a wrong answer.

**What the upgrade changed.** Error analysis of the previous version showed that on real invoices only about a third of totals were right, and that in 154 of 199 wrong totals the correct amount was already in the text the model received. Two causes:

1. Tesseract emits tables column by column, so `Total` and `$ 327,08` ended up 20–40 lines apart. Fix: rebuild the OCR text row by row from the word positions (`OCR_LAYOUT = "rows"`).
2. The model took the first amount after "Total" – the net, not the amount due. Fix: arithmetic reconciliation – if amounts on a total line satisfy subtotal + tax = total, use that total (`RECONCILE_TOTALS = True`).

Together: F1 0.847 → 0.941. Of the 146 totals that changed from wrong to right, 121 are on labels written by the original author, so the gain does not depend on the label corrections below.

**Ground-truth corrections.** On invoices 316–600 the original labels mixed net and gross totals. 26 labels that held the net total were changed to the gross (the amount due), each verified by net + VAT = gross on the OCR output. Originals are in `ground_truth_backup_20261018/`, the list in `results/relabel_totals_log_pass1.csv` (25) plus invoice 417.

**Remaining errors** on real invoices are mostly invoice numbers and dates taken from a nearby field, and day/month confusions. A larger instruction-tuned model would address these; Flan-T5-base was kept so that both architectures fit the 4 GB GPU in fp32.

## 6. Human review

Each invoice is flagged for review when a field is missing or unreadable, the total does not appear on a total line, or the mean OCR confidence is below 80.

| | A | B |
|---|---|---|
| Accepted without review (straight-through) | 92.0 % | 92.5 % |
| Fully correct among accepted invoices | 90.9 % | 90.5 % |
| Fully correct if every invoice were accepted | 84.2 % | 84.2 % |
| Flagged invoices that really were wrong | 45 of 48 (94 %) | 42 of 45 (93 %) |
| Real invoices: accepted / fully correct among accepted | 87.3 % / 81.3 % | 89.0 % / 80.5 % |

Fewer than 1 in 12 invoices go to a person, and almost all of those really needed checking. The flag cannot detect a wrong invoice number or date that looks valid.

## 7. Figures

All in `outputs/2026-10-18_23-52-41_full_benchmark/figures/`:

| File | Content |
|---|---|
| `throughput.png`, `wall_time.png` | throughput and wall time per configuration |
| `service_time.png`, `stage_breakdown.png` | compute per invoice, split into OCR / embedding / LLM |
| `memory.png` | peak RAM and pipeline GPU memory |
| `cost.png` | cost per 100 invoices |
| `f1.png`, `f1_by_source.png` | F1 overall and per subset |
| `precision_tradeoff.png` | LLM time and F1 per precision |
| `routing_scatter.png` | token count vs OCR confidence per invoice, coloured by route |

## 8. Measurement incidents

33 runs were discarded and repeated; all are kept in `discarded/` of the benchmark folder with the reason in the folder name.

- **Overlapping demo runs**: two `demo.py` runs were started during the benchmark. They competed for CPU and GPU, and – because all runs then shared one cache file – one of them reset B's cache mid-run. Since then every output folder has its own cache file.
- **A runaway background process** (`cmd.exe`, not part of the project) used up to 8 of the 12 CPU threads for hours. It was detected through OCR time per invoice rising from ~0.6 s to ~0.95 s, and all runs made while it was active were repeated after it was stopped.
- **Earlier**: Windows Modern Standby stalled a run for over two hours; the runner now keeps the system awake.

Lesson for the thesis: wall-clock benchmarks on a laptop need control of power management and background load, and per-stage timings (OCR time per invoice is a sensitive indicator) should be checked for every run.

## 9. Limitations

- Flan-T5-base is small; it limits accuracy on real invoices. The thesis measures architecture, not model quality, and both architectures share the same errors.
- The synthetic half of the dataset is easy (one layout, clean rendering); results are reported per subset for that reason.
- One machine; the main benchmark submits all invoices at once, so B's per-invoice latency there includes queueing. Latency with invoices arriving one at a time is in section 10.
- Quantization should be re-evaluated with a model large enough to be memory-bandwidth-bound (≥ 1 B parameters).

## 10. Additional experiments

Methods in `docs/EVALUATION.md` section 7; full discussion in `docs/Architecture_and_Benchmark_Report.pdf` sections 6.11, 6.12 and 7.

**Latency with invoices arriving one at a time** (`outputs/2026-10-20_01-01-33_latency_60docs`, 60 invoices, Poisson arrivals, 3 runs; median latency in seconds):

| Arrivals per minute | A | B as benchmarked | B with dynamic batching |
|---|---|---|---|
| 30 | 1.70 ± 0.13 | 7.94 ± 1.57 | 0.95 ± 0.03 |
| 48 | 3.87 ± 0.79 | 5.31 ± 0.93 | 1.05 ± 0.04 |
| 120 | 22.76 ± 1.64 | 2.98 ± 0.22 | 1.16 ± 0.18 |
| 240 | 29.94 ± 2.03 | 2.79 ± 0.02 | 1.78 ± 0.29 |

B as benchmarked waits for a batch of 8, so at light load it is slower than A; A's queue grows without limit above about 53 invoices per minute. Dynamic batching (the GPU takes every invoice already read when it is free) is fastest at every rate. F1 is identical in all runs.

**OCR engines** (`outputs/2026-10-20_01-52-39_ocr_comparison`, all 600 invoices, same downstream as A):

| Engine | OCR s per real invoice | GPU memory for OCR | F1 all | F1 real |
|---|---|---|---|---|
| Tesseract 5.4 (CPU) | 0.89 | none | 0.941 | 0.892 |
| EasyOCR 1.7.2 (GPU) | 8.00 | 3883 MB peak | 0.934 | 0.871 |

**Vision-language model** (`outputs/2026-10-20_23-54-32_vlm_baseline`, Qwen2-VL-2B in 4-bit NF4, at most 1024 visual tokens, image only):

| System | Wall time per invoice | Pipeline GPU peak | F1 all | F1 real | Fully correct of 600 |
|---|---|---|---|---|---|
| A | 1.13 s | 1067 MB | 0.941 | 0.892 | 505 |
| B | 0.20 s | 1996 MB | 0.941 | 0.894 | 505 |
| VLM | 3.90 s | 1745 MB | 0.966 | 0.930 | 541 |

The VLM is more accurate but about 19× slower per invoice than B. Estimated from the stored results: sending only the 45 invoices that B flags for review to the VLM would give 541 fully correct invoices for about 0.29 s of extra GPU time per invoice. Resolution probe (20 invoices, one setting per process, `outputs/2026-10-21_00-35-48_vlm_resolution_probe_20docs`): 512 tokens 1568 MB, 1024 tokens 1745 MB, 2048 tokens 2329 MB; accuracy did not improve beyond 1024.
