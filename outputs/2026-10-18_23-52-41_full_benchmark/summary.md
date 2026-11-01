# Experiment summary

Hardware: python=3.12.2, platform=Windows-11-10.0.22621-SP0, cpu=Intel64 Family 6 Model 154 Stepping 3, GenuineIntel, cpu_count=12, ram_total_gb=15.7, torch=2.5.1+cu121, cuda=12.1, gpu=NVIDIA GeForce RTX 3050 Laptop GPU, gpu_vram_gb=4.0

Values are mean ± standard deviation over the runs of each configuration.

## Configurations

| Configuration | Runs | Description |
|---|---|---|
| A | 3 | Baseline sequential OCR -> embeddings -> LLM, fp32, one document at a time |
| B | 3 | Optimized pipeline: 3 OCR workers, batch 8, cache, routing FAST=4bit / FULL=fp32 |
| B-warm | 3 | Optimized pipeline: 3 OCR workers, batch 8, cache, routing FAST=4bit / FULL=fp32 |
| B-nocache | 3 | B without the result cache |
| B-allfull | 3 | B with every document on the FULL fp32 route (no quantized model) |
| B-allfast | 3 | B with every document on the FAST 4-bit route |
| B-fp16 | 3 | B with every document on an fp16 model |
| B-8bit | 3 | B with every document on an 8-bit model |
| B-1worker | 3 | B with a single OCR worker (no OCR parallelism) |
| B-6workers | 3 | B with six OCR workers |
| B-nobatch | 3 | B with batch size 1 (no GPU batching) |
| B-tuned | 3 | B with thresholds from routing_analysis of this run: 525 tokens, confidence 70.0 |

## Performance

| Configuration | Wall time (s) | Throughput (docs/min) | Mean service time (s/doc) | p95 latency (s) | Mean OCR (s) | Mean embedding (s) | Mean LLM (s) |
|---|---|---|---|---|---|---|---|
| A | 677.92 ± 7.41 | 53.11 ± 0.58 | 1.119 ± 0.012 | 1.79 ± 0.03 | 0.588 ± 0.015 | 0.017 ± 0.001 | 0.514 ± 0.007 |
| B | 119.43 ± 4.69 | 301.73 ± 11.81 | 0.718 ± 0.037 | 112.70 ± 4.39 | 0.576 ± 0.025 | 0.008 ± 0.002 | 0.134 ± 0.010 |
| B-warm | 0.59 ± 0.03 | 60757.09 ± 3073.71 | 0.001 ± 0.000 | 0.00 ± 0.00 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| B-nocache | 125.95 ± 2.63 | 285.92 ± 6.00 | 0.763 ± 0.018 | 118.95 ± 2.65 | 0.616 ± 0.013 | 0.008 ± 0.000 | 0.139 ± 0.006 |
| B-allfull | 126.05 ± 2.63 | 285.69 ± 6.03 | 0.740 ± 0.026 | 118.85 ± 2.54 | 0.615 ± 0.013 | 0.010 ± 0.002 | 0.115 ± 0.013 |
| B-allfast | 125.84 ± 0.80 | 286.07 ± 1.82 | 0.739 ± 0.004 | 118.65 ± 0.96 | 0.614 ± 0.005 | 0.010 ± 0.000 | 0.115 ± 0.001 |
| B-fp16 | 123.46 ± 0.07 | 291.59 ± 0.17 | 0.693 ± 0.001 | 116.25 ± 0.13 | 0.604 ± 0.001 | 0.011 ± 0.000 | 0.079 ± 0.000 |
| B-8bit | 167.79 ± 1.61 | 214.57 ± 2.05 | 0.896 ± 0.007 | 160.94 ± 1.46 | 0.630 ± 0.006 | 0.009 ± 0.000 | 0.258 ± 0.003 |
| B-1worker | 338.36 ± 5.12 | 106.41 ± 1.60 | 0.690 ± 0.010 | 319.06 ± 4.71 | 0.558 ± 0.009 | 0.019 ± 0.001 | 0.113 ± 0.001 |
| B-6workers | 102.92 ± 10.29 | 352.11 ± 35.08 | 0.920 ± 0.095 | 98.27 ± 10.25 | 0.762 ± 0.079 | 0.007 ± 0.002 | 0.151 ± 0.015 |
| B-nobatch | 248.81 ± 2.48 | 144.70 ± 1.45 | 0.988 ± 0.017 | 237.32 ± 2.49 | 0.588 ± 0.013 | 0.014 ± 0.000 | 0.385 ± 0.005 |
| B-tuned | 126.06 ± 3.68 | 285.74 ± 8.48 | 0.742 ± 0.022 | 118.77 ± 3.73 | 0.614 ± 0.018 | 0.010 ± 0.000 | 0.118 ± 0.004 |

## Resources and cost

| Configuration | Peak RAM (MB) | Pipeline GPU peak (MB) | Whole-GPU peak (MB) | Cost per 100 docs (EUR) |
|---|---|---|---|---|
| A | 1438 ± 14 | 1067 ± 0 | 1388 ± 0 | 0.0110 ± 0.0001 |
| B | 2012 ± 16 | 1996 ± 15 | 3172 ± 158 | 0.0019 ± 0.0001 |
| B-warm | 1424 ± 2 | 1996 ± 15 | 3172 ± 158 | 0.0000 ± 0.0000 |
| B-nocache | 2025 ± 8 | 2006 ± 0 | 3103 ± 27 | 0.0020 ± 0.0001 |
| B-allfull | 2062 ± 17 | 1645 ± 0 | 3124 ± 0 | 0.0021 ± 0.0001 |
| B-allfast | 1997 ± 37 | 1102 ± 0 | 2222 ± 0 | 0.0020 ± 0.0001 |
| B-fp16 | 2404 ± 78 | 1049 ± 0 | 1996 ± 0 | 0.0020 ± 0.0000 |
| B-8bit | 1984 ± 5 | 1176 ± 0 | 2550 ± 1 | 0.0027 ± 0.0000 |
| B-1worker | 1741 ± 53 | 1979 ± 0 | 3000 ± 0 | 0.0055 ± 0.0001 |
| B-6workers | 2604 ± 12 | 1979 ± 0 | 3290 ± 0 | 0.0017 ± 0.0002 |
| B-nobatch | 2052 ± 11 | 1537 ± 0 | 1880 ± 0 | 0.0040 ± 0.0001 |
| B-tuned | 2079 ± 20 | 2049 ± 0 | 3544 ± 0 | 0.0021 ± 0.0001 |

## Accuracy

Strict matching (headline): invoice numbers equal after removing labels and punctuation, the same calendar date, the same amount to the cent. Relaxed matching adds invoice numbers with a similarity ratio of at least 90 and day/month swaps of ambiguous numeric dates.

| Configuration | F1 (all) | F1 (real) | F1 (synthetic) | Documents fully correct | F1 (all), relaxed | Fully correct, relaxed |
|---|---|---|---|---|---|---|
| A | 0.9394 ± 0.0000 | 0.8895 ± 0.0000 | 0.9889 ± 0.0000 | 0.8400 ± 0.0000 | 0.9394 ± 0.0000 | 0.8400 ± 0.0000 |
| B | 0.9389 ± 0.0000 | 0.8918 ± 0.0000 | 0.9856 ± 0.0000 | 0.8383 ± 0.0000 | 0.9389 ± 0.0000 | 0.8383 ± 0.0000 |
| B-warm | 0.9389 ± 0.0000 | 0.8918 ± 0.0000 | 0.9856 ± 0.0000 | 0.8383 ± 0.0000 | 0.9389 ± 0.0000 | 0.8383 ± 0.0000 |
| B-nocache | 0.9389 ± 0.0000 | 0.8918 ± 0.0000 | 0.9856 ± 0.0000 | 0.8383 ± 0.0000 | 0.9389 ± 0.0000 | 0.8383 ± 0.0000 |
| B-allfull | 0.9394 ± 0.0000 | 0.8895 ± 0.0000 | 0.9889 ± 0.0000 | 0.8400 ± 0.0000 | 0.9394 ± 0.0000 | 0.8400 ± 0.0000 |
| B-allfast | 0.9389 ± 0.0000 | 0.8918 ± 0.0000 | 0.9856 ± 0.0000 | 0.8383 ± 0.0000 | 0.9389 ± 0.0000 | 0.8383 ± 0.0000 |
| B-fp16 | 0.9394 ± 0.0000 | 0.8895 ± 0.0000 | 0.9889 ± 0.0000 | 0.8400 ± 0.0000 | 0.9394 ± 0.0000 | 0.8400 ± 0.0000 |
| B-8bit | 0.9374 ± 0.0003 | 0.8877 ± 0.0006 | 0.9867 ± 0.0000 | 0.8339 ± 0.0010 | 0.9374 ± 0.0003 | 0.8339 ± 0.0010 |
| B-1worker | 0.9389 ± 0.0000 | 0.8918 ± 0.0000 | 0.9856 ± 0.0000 | 0.8383 ± 0.0000 | 0.9389 ± 0.0000 | 0.8383 ± 0.0000 |
| B-6workers | 0.9389 ± 0.0000 | 0.8918 ± 0.0000 | 0.9856 ± 0.0000 | 0.8383 ± 0.0000 | 0.9389 ± 0.0000 | 0.8383 ± 0.0000 |
| B-nobatch | 0.9389 ± 0.0000 | 0.8918 ± 0.0000 | 0.9856 ± 0.0000 | 0.8383 ± 0.0000 | 0.9389 ± 0.0000 | 0.8383 ± 0.0000 |
| B-tuned | 0.9389 ± 0.0000 | 0.8918 ± 0.0000 | 0.9856 ± 0.0000 | 0.8383 ± 0.0000 | 0.9389 ± 0.0000 | 0.8383 ± 0.0000 |

## Per-field F1 (all documents)

| Configuration | invoice_number | date | total |
|---|---|---|---|
| A | 0.9648 ± 0.0000 | 0.9597 ± 0.0000 | 0.8939 ± 0.0000 |
| B | 0.9615 ± 0.0000 | 0.9597 ± 0.0000 | 0.8956 ± 0.0000 |
| B-warm | 0.9615 ± 0.0000 | 0.9597 ± 0.0000 | 0.8956 ± 0.0000 |
| B-nocache | 0.9615 ± 0.0000 | 0.9597 ± 0.0000 | 0.8956 ± 0.0000 |
| B-allfull | 0.9648 ± 0.0000 | 0.9597 ± 0.0000 | 0.8939 ± 0.0000 |
| B-allfast | 0.9615 ± 0.0000 | 0.9597 ± 0.0000 | 0.8956 ± 0.0000 |
| B-fp16 | 0.9648 ± 0.0000 | 0.9597 ± 0.0000 | 0.8939 ± 0.0000 |
| B-8bit | 0.9648 ± 0.0000 | 0.9581 ± 0.0000 | 0.8895 ± 0.0010 |
| B-1worker | 0.9615 ± 0.0000 | 0.9597 ± 0.0000 | 0.8956 ± 0.0000 |
| B-6workers | 0.9615 ± 0.0000 | 0.9597 ± 0.0000 | 0.8956 ± 0.0000 |
| B-nobatch | 0.9615 ± 0.0000 | 0.9597 ± 0.0000 | 0.8956 ± 0.0000 |
| B-tuned | 0.9615 ± 0.0000 | 0.9597 ± 0.0000 | 0.8956 ± 0.0000 |

## Human review

An invoice is flagged for review when a field is missing or unreadable, the total does not appear on a total line, or OCR confidence is below 80.

| Configuration | Straight-through (no review) | Fully correct, auto-accepted | Fully correct, no review step | Wrong invoices caught by the flag |
|---|---|---|---|---|
| A | 92.0% | 90.8% | 84.0% | 46.9% |
| B | 92.5% | 90.1% | 83.8% | 43.3% |
| B-warm | 92.5% | 90.1% | 83.8% | 43.3% |
| B-nocache | 92.5% | 90.1% | 83.8% | 43.3% |
| B-allfull | 92.0% | 90.8% | 84.0% | 46.9% |
| B-allfast | 92.5% | 90.1% | 83.8% | 43.3% |
| B-fp16 | 92.0% | 90.8% | 84.0% | 46.9% |
| B-8bit | 91.6% | 90.5% | 83.4% | 47.8% |
| B-1worker | 92.5% | 90.1% | 83.8% | 43.3% |
| B-6workers | 92.5% | 90.1% | 83.8% | 43.3% |
| B-nobatch | 92.5% | 90.1% | 83.8% | 43.3% |
| B-tuned | 92.5% | 90.1% | 83.8% | 43.3% |

## Routing and cache

| Configuration | FAST docs | FULL docs | Cache hits | Cache misses |
|---|---|---|---|---|
| A | 0 | 0 | 0 | 0 |
| B | 505 | 95 | 0 | 600 |
| B-warm | 0 | 0 | 600 | 0 |
| B-nocache | 505 | 95 | 0 | 0 |
| B-allfull | 0 | 600 | 0 | 600 |
| B-allfast | 600 | 0 | 0 | 600 |
| B-fp16 | 600 | 0 | 0 | 600 |
| B-8bit | 600 | 0 | 0 | 600 |
| B-1worker | 505 | 95 | 0 | 600 |
| B-6workers | 505 | 95 | 0 | 600 |
| B-nobatch | 505 | 95 | 0 | 600 |
| B-tuned | 599 | 1 | 0 | 600 |
