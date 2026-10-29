# Vision-language model baseline: Qwen/Qwen2-VL-2B-Instruct

The VLM reads the invoice image directly (no OCR) and answers in JSON; answers go through the same normalisation and scorer as A and B. One run, one invoice at a time.

| Max visual tokens | Mean visual tokens | Time per invoice mean / median / p95 (s) | Throughput (/min) | GPU peak (MB) | Valid JSON | F1 all | F1 real | F1 synthetic | Real: number / date / total | Fully correct | Out of memory |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 879.5 | 3.8987 / 4.0073 / 4.3566 | 15.39 | 1745 | 0.9967 | 0.9656 | 0.9302 | 1.0 | 0.99 / 0.96 / 0.85 | 0.9017 | 0 |
