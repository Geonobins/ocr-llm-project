# Vision-language model baseline: Qwen/Qwen2-VL-2B-Instruct

The VLM reads the invoice image directly (no OCR) and answers in JSON; answers go through the same normalisation and scorer as A and B. One run, one invoice at a time. F1 and fully correct use strict matching; the relaxed columns add fuzzy invoice numbers and day/month swaps of ambiguous numeric dates.

| Max visual tokens | Mean visual tokens | Time per invoice mean / median / p95 (s) | Throughput (/min) | GPU peak (MB) | Valid JSON | F1 all | F1 real | F1 synthetic | Real: number / date / total | Fully correct | F1 all, relaxed | Fully correct, relaxed | Out of memory |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 879.5 | 3.8987 / 4.0073 / 4.3566 | 15.39 | 1745 | 0.9967 | 0.9639 | 0.9268 | 1.0 | 0.98 / 0.96 / 0.85 | 0.8967 | 0.9651 | 0.9 | 0 |
