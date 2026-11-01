# OCR engine comparison

Each engine reads all invoices once; its text then goes through the same downstream as Architecture A (row layout, MiniLM line filter, Flan-T5-base fp32, total check) and the same scorer. F1 and fully correct use strict matching; the relaxed columns add fuzzy invoice numbers and day/month swaps of ambiguous numeric dates.

| Engine | OCR time mean / median / p95 (s) | Real / synthetic mean (s) | OCR GPU peak (MB) | Mean confidence | F1 all | F1 real | F1 synthetic | Fully correct | F1 all, relaxed | Fully correct, relaxed |
|---|---|---|---|---|---|---|---|---|---|---|
| tesseract | 0.571 / 0.362 / 1.219 | 0.892 / 0.250 | 0 | 93.0 | 0.9394 | 0.8895 | 0.9889 | 0.8400 | 0.9394 | 0.8400 |
| easyocr | 4.155 / 0.553 / 13.488 | 7.998 / 0.312 | 3883 | 86.6 | 0.9319 | 0.8666 | 0.9967 | 0.8217 | 0.9319 | 0.8217 |

Share of ground-truth values that appear in the OCR text (an upper bound for extraction):

| Engine | Subset | Invoice number | Date | Total |
|---|---|---|---|---|
| tesseract | all | 0.990 | 0.995 | 0.987 |
| tesseract | real | 0.980 | 0.990 | 0.973 |
| tesseract | synthetic | 1.000 | 1.000 | 1.000 |
| easyocr | all | 0.990 | 0.995 | 0.960 |
| easyocr | real | 0.980 | 0.990 | 0.919 |
| easyocr | synthetic | 1.000 | 1.000 | 1.000 |

| Engine | F1 invoice number | F1 date | F1 total |
|---|---|---|---|
| tesseract | 0.9648 | 0.9597 | 0.8939 |
| easyocr | 0.9648 | 0.9614 | 0.8698 |
