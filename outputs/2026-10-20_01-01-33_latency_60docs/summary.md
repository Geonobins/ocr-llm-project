# Latency with invoices arriving one at a time

Invoices arrive at random times (Poisson process) at the given average rate. Latency = time from an invoice's arrival to its result.
Values are mean ± standard deviation over the runs.

| Variant | Arrival rate (/min) | Runs | Median latency (s) | p95 latency (s) | Max latency (s) | Service time (s) | Mean OCR (s) | Throughput (/min) | F1 |
|---|---|---|---|---|---|---|---|---|---|
| A | 30 | 3 | 1.70 ± 0.13 | 4.19 ± 0.38 | 5.75 ± 0.86 | 1.233 ± 0.064 | 0.605 ± 0.008 | 28.6 ± 6.4 | 0.9444 ± 0.0000 |
| A | 48 | 3 | 3.87 ± 0.79 | 11.77 ± 4.65 | 13.13 ± 5.27 | 1.182 ± 0.029 | 0.591 ± 0.004 | 42.5 ± 6.4 | 0.9444 ± 0.0000 |
| A | 120 | 3 | 22.76 ± 1.64 | 41.46 ± 5.64 | 43.56 ± 6.66 | 1.179 ± 0.022 | 0.589 ± 0.006 | 50.4 ± 0.9 | 0.9444 ± 0.0000 |
| A | 240 | 3 | 29.94 ± 2.03 | 53.88 ± 1.08 | 56.76 ± 1.59 | 1.185 ± 0.046 | 0.595 ± 0.010 | 50.6 ± 2.0 | 0.9444 ± 0.0000 |
| B | 30 | 3 | 7.94 ± 1.57 | 21.40 ± 4.76 | 23.18 ± 5.80 | 0.767 ± 0.021 | 0.593 ± 0.014 | 28.4 ± 6.3 | 0.9444 ± 0.0000 |
| B | 48 | 3 | 5.31 ± 0.93 | 12.46 ± 1.53 | 15.40 ± 2.33 | 0.750 ± 0.021 | 0.581 ± 0.017 | 48.8 ± 13.2 | 0.9444 ± 0.0000 |
| B | 120 | 3 | 2.98 ± 0.22 | 5.92 ± 0.72 | 7.07 ± 0.84 | 0.728 ± 0.023 | 0.572 ± 0.018 | 123.0 ± 25.7 | 0.9444 ± 0.0000 |
| B | 240 | 3 | 2.79 ± 0.02 | 4.52 ± 0.51 | 5.11 ± 0.66 | 0.733 ± 0.031 | 0.579 ± 0.021 | 219.2 ± 31.3 | 0.9444 ± 0.0000 |
| B-greedy | 30 | 3 | 0.95 ± 0.03 | 1.74 ± 0.09 | 1.96 ± 0.05 | 0.981 ± 0.015 | 0.583 ± 0.005 | 28.6 ± 6.4 | 0.9444 ± 0.0000 |
| B-greedy | 48 | 3 | 1.05 ± 0.04 | 1.80 ± 0.12 | 2.21 ± 0.17 | 0.942 ± 0.023 | 0.588 ± 0.011 | 49.1 ± 13.5 | 0.9444 ± 0.0000 |
| B-greedy | 120 | 3 | 1.16 ± 0.18 | 2.18 ± 0.38 | 2.41 ± 0.31 | 0.864 ± 0.009 | 0.570 ± 0.020 | 124.5 ± 26.4 | 0.9444 ± 0.0000 |
| B-greedy | 240 | 3 | 1.78 ± 0.29 | 2.76 ± 0.50 | 3.04 ± 0.53 | 0.822 ± 0.020 | 0.587 ± 0.014 | 225.4 ± 29.9 | 0.9444 ± 0.0000 |

| Variant | Description |
|---|---|
| A | Architecture A: invoices processed one at a time in arrival order |
| B | Architecture B as benchmarked: a batch goes to the GPU once 8 invoices are read |
| B-greedy | Architecture B: when the GPU is free it takes every invoice already read, up to 8 |
