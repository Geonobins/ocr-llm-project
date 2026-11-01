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
| F1, all / real / synthetic (strict matching) | 0.939 / 0.890 / 0.989 | 0.939 / 0.892 / 0.986 | equal |
| Invoices with all three fields correct | 504 of 600 | 503 of 600 | one invoice apart |

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
| fp32 (B-allfull) | 126.1 ± 2.6 s | 0.115 s | 1645 MB | 0.9394 |
| fp16 (B-fp16) | 123.5 ± 0.1 s | 0.079 s | 1049 MB | 0.9394 |
| 4-bit NF4 (B-allfast) | 125.8 ± 0.8 s | 0.115 s | 1102 MB | 0.9389 |
| 8-bit LLM.int8 (B-8bit) | 167.8 ± 1.6 s | 0.258 s | 1176 MB | 0.9374 ± 0.0003 |

- **4-bit saves memory, not time**: one third less GPU memory than fp32 at the same speed and practically the same accuracy (one invoice apart: the 4-bit model answers `1494-131` for `INV-1494-131`).
- **fp16 is the best option on this GPU**: fastest LLM step, least memory, same accuracy. For a 248 M-parameter model, bitsandbytes quantization brings no advantage over half precision; its de-quantization cost cancels the saved memory traffic.
- **8-bit is the slowest** (2.2× slower LLM step than fp32) and slightly non-deterministic: its F1 varies between runs, while every other precision gives identical answers in all three runs.

## 4. Routing

- B as designed routes by token count and OCR confidence with the initial thresholds (300 tokens, confidence 85): 505 invoices to the 4-bit model, 95 to fp32.
- `evaluation/routing_analysis.py` compared the all-4-bit and all-fp32 runs invoice by invoice and searched a grid of thresholds for the largest 4-bit share that keeps field accuracy within 0.5 percentage points of all-fp32. Result: **525 tokens, confidence 70**, which sends 99.8 % of invoices to the 4-bit model.
- One model is right and the other wrong on only 11 invoices (5 favour the 4-bit model, 6 the fp32 model); accuracy per token-count bin is equal within 0.5 percentage points everywhere. Re-derived with the strict scorer, the thresholds are unchanged.
- The thresholds were chosen on the same 600 invoices that are used to report accuracy; there is no held-out set. This cannot inflate the reported accuracy, because sending every invoice to either model gives practically the same accuracy (0.934 vs 0.935 field accuracy).
- Validated with three fresh runs (B-tuned): 126.1 ± 3.7 s, F1 0.939, 599 of 600 invoices on the 4-bit route.

The router works; with two precisions of the same small model it has little to choose between. It becomes useful when the two routes differ more – a small model for simple invoices and a larger one for complex ones.

## 5. Accuracy

| Version | Scorer | Arch. | F1 all | F1 real | F1 synthetic | Real: number / date / total | Real invoices fully correct |
|---|---|---|---|---|---|---|---|
| Before the accuracy upgrade | original | A | 0.847 | 0.704 | 0.989 | 0.93 / 0.86 / 0.32 | 24 % |
| | original | B | 0.850 | 0.712 | 0.987 | 0.93 / 0.86 / 0.34 | 26 % |
| Current code | original | A | 0.941 | 0.892 | 0.989 | 0.94 / 0.92 / 0.82 | 72 % |
| | original | B | 0.941 | 0.894 | 0.987 | 0.94 / 0.92 / 0.83 | 72 % |
| Current code | strict | A | 0.939 | 0.890 | 0.989 | 0.93 / 0.92 / 0.82 | 71 % |
| | strict | B | 0.939 | 0.892 | 0.986 | 0.93 / 0.92 / 0.83 | 72 % |

The before/after rows use the original scorer for both versions, so the comparison isolates the extraction upgrade. The pre-upgrade archive is not compared under strict matching: its Architecture A dropped the `INV-` prefix of every synthetic invoice number (`8289-16` for `INV-8289-16`), a post-processing difference of that older version that strict matching counts as wrong (its synthetic F1 would fall to 0.66) and that has nothing to do with the upgrade.

Per field over all 600 invoices (B, strict): invoice number 0.962, date 0.960, total 0.896. Precision (0.935) and recall (0.943) are close, because the model almost never answers `null`, so nearly every miss is also a wrong answer.

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
| Fully correct among accepted invoices | 90.8 % | 90.1 % |
| Fully correct if every invoice were accepted | 84.0 % | 83.8 % |
| Flagged invoices that really were wrong | 45 of 48 (94 %) | 42 of 45 (93 %) |
| Wrong invoices caught by the flag | 45 of 96 (47 %) | 42 of 97 (43 %) |
| Real invoices: accepted / fully correct among accepted | 87.3 % / 80.9 % | 89.0 % / 80.2 % |

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
| Tesseract 5.4 (CPU) | 0.89 | none | 0.939 | 0.890 |
| EasyOCR 1.7.2 (GPU) | 8.00 | 3883 MB peak | 0.932 | 0.867 |

**Vision-language model** (`outputs/2026-10-20_23-54-32_vlm_baseline`, Qwen2-VL-2B in 4-bit NF4, at most 1024 visual tokens, image only):

| System | Wall time per invoice | Pipeline GPU peak | F1 all | F1 real | Fully correct of 600 |
|---|---|---|---|---|---|
| A | 1.13 s | 1067 MB | 0.939 | 0.890 | 504 |
| B | 0.20 s | 1996 MB | 0.939 | 0.892 | 503 |
| VLM | 3.90 s | 1745 MB | 0.964 | 0.927 | 538 |

With relaxed matching the VLM reaches 0.965 and 540 (two invoice numbers with one extra digit); A and B do not change. The VLM is more accurate but about 19× slower per invoice than B's batched average and 3.5× slower than A, which like the VLM reads one invoice at a time. Estimated from the stored results: sending only the 45 invoices that B flags for review to the VLM would give 539 fully correct invoices (B is right on 3 of them, the VLM on 39) for about 0.3 s of extra GPU time per invoice. Resolution probe (20 invoices, one setting per process, `outputs/2026-10-21_00-35-48_vlm_resolution_probe_20docs`): 512 tokens 1568 MB, 1024 tokens 1745 MB, 2048 tokens 2329 MB; accuracy did not improve beyond 1024.

## 11. Scorer change (1 Nov 2026)

An external review showed that the original scorer accepted some wrong answers (rules and reasons in `docs/EVALUATION.md` §2 and §7.4). The scorer now reports **strict** matching as the headline metric and **relaxed** matching (fuzzy invoice numbers ≥ 90, day/month swap for ambiguous numeric dates) as a secondary metric. All stored predictions were re-scored; no model was re-run.

| System | F1, original → strict | Fully correct, original → strict | F1 and fully correct, relaxed |
|---|---|---|---|
| A | 0.9406 → 0.9394 | 505 → 504 | 0.9394 / 504 |
| B (and every B variant except B-8bit, B-allfull, B-fp16) | 0.9406 → 0.9389 | 505 → 503 | 0.9389 / 503 |
| B-allfull, B-fp16 | 0.9406 → 0.9394 | 505 → 504 | 0.9394 / 504 |
| B-8bit | 0.9385 → 0.9374 | 501.3 → 500.3 | 0.9374 / 500.3 |
| Tesseract → A downstream | 0.9406 → 0.9394 | 505 → 504 | 0.9394 / 504 |
| EasyOCR → A downstream | 0.9342 → 0.9319 | 496 → 493 | 0.9319 / 493 |
| VLM (Qwen2-VL-2B) | 0.9656 → 0.9639 | 541 → 538 | 0.9651 / 540 |

Only 9 distinct answers changed:

| Invoice | Field | Answer | Ground truth | Original rule that accepted it | Where |
|---|---|---|---|---|---|
| 14 | invoice number | `009` | `GALT-009` | same digits | A, B and all variants, Tesseract |
| 579 | invoice number | `73627 Square` | `73627` | same digits | A, B and all variants, both OCR engines |
| 131 | invoice number | `1494-131` | `INV-1494-131` | same digits | B variants that use the 4-bit model |
| 1 | invoice number | `01234 Richard Sanchez` | `01234` | same digits | EasyOCR |
| 339 | invoice number | `93632 Avenue` | `93632` | same digits | EasyOCR |
| 571, 572 | invoice number | `898724443` | `89872443` | fuzzy ratio ≥ 90 (still accepted by relaxed) | VLM |
| 436 | total | `1,1696` | `$11696` | lenient separator parsing | VLM |
| 495 | date | `21/2022` | `Jun 21, 2022` | fuzzy fallback for unparseable dates | EasyOCR |

The day/month swap, the fuzzy fallback for totals, negative amounts and answers containing several amounts never occurred in any stored run. For A and B relaxed equals strict: fuzzy matching never rescued an answer of either architecture. The comparison of A and B, the ranking of the OCR engines and the advantage of the VLM are unchanged.

**Ground-truth labels to check.** Checking every label against the strict rules found four real invoices whose date label is not a date: 489 (`Invoice Date`), 503 (`Invoice Date`), 510 (`01`) and 534 (`THANK YOU..`); the invoice-number labels of 489 (`#`), 503 (`FLYING COLOURS CORPORATION`) and 534 (`ID`) are doubtful as well. Invoice 489 was compared with its image: it has no invoice number and no date, so both labels should be empty. The VLM answers `null` for both fields on all four invoices and is currently marked wrong. The labels are left unchanged here; they belong to the label audit.
