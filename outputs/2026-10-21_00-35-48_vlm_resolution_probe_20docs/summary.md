# Vision-language model baseline: Qwen/Qwen2-VL-2B-Instruct

The VLM reads the invoice image directly (no OCR) and answers in JSON; answers go through the same normalisation and scorer as A and B. One run, one invoice at a time.

| Max visual tokens | Mean visual tokens | Time per invoice mean / median / p95 (s) | Throughput (/min) | GPU peak (MB) | Valid JSON | F1 all | F1 real | F1 synthetic | Real: number / date / total | Fully correct | Out of memory |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 512 | 493.9 | 3.2264 / 3.2306 / 3.6708 | 18.59 | 1568 | 1.0 | 0.9474 | 0.9 | 1.0 | 1.00 / 1.00 / 0.70 | 0.85 | 0 |
| 1024 | 874.1 | 3.8122 / 3.8506 / 4.2617 | 15.74 | 1745 | 1.0 | 1.0 | 1.0 | 1.0 | 1.00 / 1.00 / 1.00 | 1.0 | 0 |
| 2048 | 1445.8 | 5.1766 / 5.7151 / 6.8242 | 11.59 | 2329 | 1.0 | 0.9474 | 0.9 | 1.0 | 1.00 / 1.00 / 0.70 | 0.85 | 0 |
