# Architecture

This document explains how the two pipelines work, module by module, and why each design decision was made. Where a decision rests on a measurement, the measurement is given. All numbers were taken on the thesis hardware: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM), 12 logical CPU cores, 16 GB RAM, Windows 11, Python 3.12, PyTorch 2.5.1 + CUDA 12.1, Tesseract 5.4.

## 1. The task

Input: an invoice image (PNG, JPG or WebP). Output: a JSON object with three fields.

```json
{"invoice_number": "INV-8289-16", "date": "April 19, 2024", "total": "$45,130"}
```

Both architectures solve exactly this task with exactly the same models. The thesis question is not "which model is better" but "how much latency, throughput, memory and cost can an optimized execution architecture save while keeping accuracy".

## 2. Shared building blocks

### 2.1 OCR (`pipeline/ocr.py`)

Tesseract 5 through `pytesseract.image_to_data`. One call returns every recognised word with its bounding box, block/paragraph/line numbers and a confidence value between 0 and 100. From that single call the module builds:

- the **text**, rebuilt row by row from the word positions (`OCR_LAYOUT = "rows"` in `pipeline/config.py`, function `_assemble_rows`): Tesseract's line segments whose vertical centres differ by less than 0.6 × the median line height form one visual row, joined left to right with ` | `. A label and its value stay on one line even where Tesseract reads a table column by column – `Total | $ 297,35 | $ 29,73 | $ 327,08` instead of `Total` and `$ 327,08` thirty lines apart. The earlier layout (`"blocks"`, Tesseract's own block → paragraph → line order) remains selectable for comparison;
- the **mean word confidence**, used as a routing feature and a quality signal;
- the word and line counts.

The image is converted to grayscale (`convert("L")`) and EXIF-rotated before OCR. `OMP_THREAD_LIMIT=1` is set in every worker so that Tesseract's internal threads do not fight with the process-level parallelism of Architecture B.

`EasyOcrEngine` in the same module wraps EasyOCR (on the GPU) for the engine comparison in `docs/EVALUATION.md` §7.2: its detected text boxes go through the same row-by-row layout (`_rows_from_segments`) and its confidences (0–1) are scaled to 0–100. The pipelines themselves use Tesseract.

Why `image_to_data` instead of `image_to_string`: the earlier version of the code called Tesseract once for text and would have needed a second call for confidence. One call gives both, and OCR is the most expensive CPU stage (0.25 s for a synthetic 800×600 image, about 1 s for a 2481×3509 scanned invoice).

### 2.2 Embedding filter (`pipeline/embeddings.py`)

Flan-T5 accepts at most 512 input tokens. Real invoices OCR to 100–750 tokens, and the total is usually at the bottom, so simple truncation would cut off the field we need most. The embedding filter keeps the prompt within the limit while preserving the relevant lines:

1. Split the OCR text into non-empty lines.
2. If there are at most 18 lines (6 per field), keep everything – nothing can be gained by filtering.
3. Otherwise encode every line with `all-MiniLM-L6-v2` (384-dimensional sentence embeddings, L2-normalised) and compute cosine similarity against three fixed query embeddings: "invoice number", "invoice date", "total amount due".
4. For each query keep the 6 most similar lines **and the line that follows each of them**, because on invoices the label and the value are often on consecutive lines (`Total` / `1574`).
5. Emit the selected lines in their original order.

Architecture A calls `select` once per document. Architecture B calls `select_batch`, which encodes the lines of a whole batch of documents in one forward pass and then splits the result – the same selection, less GPU overhead.

### 2.3 LLM extraction (`pipeline/llm.py`)

Model: `google/flan-t5-base` (248 M parameters, encoder–decoder). It is loaded in one of five precisions:

| Precision | How | Weight memory |
|---|---|---|
| fp32 | plain PyTorch | ~990 MB |
| fp16 / bf16 | `dtype=torch.float16/bfloat16` | ~500 MB |
| 8-bit | `BitsAndBytesConfig(load_in_8bit=True)` (LLM.int8) | ~300 MB |
| 4-bit | `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)` (QLoRA-style NF4) | ~200 MB |

**Prompt format.** Each field is asked as a separate reading-comprehension question:

```
<selected invoice lines>

Question: What is the total amount due?
Answer:
```

This is the format Flan-T5 was instruction-tuned on. The earlier version of the code asked the model to "return STRICT JSON", which can never work with T5: its SentencePiece vocabulary has no `{` or `}` token, so the braces are emitted as `<unk>` and dropped. The JSON regex in the old `clean_output` therefore never matched and every result silently came from a regex fallback – the LLM was never actually used. The question format removes the fallback entirely; what the model answers is what the pipeline outputs.

**Post-processing** (`normalize.clean_prediction`) is deliberately minimal and identical for both architectures: whitespace collapse, an amount token is cut out of the total answer, an "Invoice No:" style prefix is removed from the invoice-number answer, empty answers become `"null"`.

**Sequential vs batched.** `extract_sequential` runs three `generate` calls per document (A). `extract_batch` builds `3 × batch_size` prompts and runs one `generate` call (B). Measured on one document with the three questions, batched generation is about 2.2× faster than three sequential calls at every precision.

**Precision micro-benchmark** (one document, three questions, batched, measured while choosing the design; the full-dataset numbers are in `docs/RESULTS.md` §3):

| Precision | Batched LLM time | Peak VRAM |
|---|---|---|
| fp32 | 222 ms | 971 MB |
| fp16 | 324 ms | 568 MB |
| 8-bit | 986 ms | 518 MB |
| 4-bit NF4 | 404 ms | 450 MB |

The full-dataset benchmark confirms the pattern: on a 248 M-parameter model quantization **does not reduce time** (fp32 126.1 s, 4-bit 125.8 s, 8-bit 167.8 s for 600 invoices), but 4-bit **does reduce memory** (pipeline GPU peak 1102 MB vs 1645 MB for fp32) at the same accuracy. fp16 turned out best on this GPU: 123.5 s, 1049 MB, same F1. 8-bit (LLM.int8 with its outlier decomposition) is the slowest and slightly non-deterministic.

### 2.4 Normalisation (`pipeline/normalize.py`)

Ground-truth values were written by hand in whatever format the invoice used (`FEB112020`, `Aug. 22, 2023`, `11.02.2030`, `188,11`, `$1 650.00`, `#1234`). Comparing raw strings would punish the pipeline for formatting, not for errors, so both scoring and output cleaning go through one canonical form:

- **Invoice number**: strip an `Invoice / No / Number / #` prefix, drop every non-alphanumeric character, upper-case. `INV-8289-16` → `INV828916`, `#1234` → `1234`.
- **Date**: insert spaces at letter/digit boundaries (`FEB112020` → `FEB 11 2020`), split a glued day+year (`112020` → `11 2020`), remove ordinal suffixes, parse with `dateutil`. Day/month order is ambiguous in `07/06/2025`, so two dates are also considered equal when day and month are swapped.
- **Amount**: remove currency symbols and spaces, then decide what `,` and `.` mean: if both occur, the last one is the decimal separator; a lone comma followed by two digits is a decimal (`188,11` → 188.11); a comma between groups of three digits is a thousands separator (`6,270` → 6270); `1.300` is treated as a thousands separator because invoice totals are rarely quoted to three decimals.

### 2.5 Result cache (`pipeline/cache.py`)

A SQLite table `results(key, payload, created)`. The key is the SHA-256 of the **image bytes** plus a *namespace* string that encodes the model name, precisions, routing policy and thresholds. Consequences:

- a renamed or copied file hits the cache (content-addressed);
- a modified image or a changed model/configuration misses it, so stale results cannot leak across experiments;
- the payload stores the prediction, the routing features and the route, so a warm run can still report what happened.

The earlier version keyed the cache on the file name only and serialised the result with `str()` / `eval()`, which is both unsafe and brittle. The new cache stores JSON.

SQLite was chosen over Redis because the thesis targets a single workstation: SQLite needs no server process, survives restarts, and a lookup costs about 1 ms including hashing the image (0.59 s for all 600 invoices). Redis would only pay off when several machines share the cache.

Every output folder gets its own cache file (`outputs/<run>/cache.sqlite`). With one shared file, a demo started during the benchmark once cleared the benchmark's cache mid-run.

### 2.6 Routing (`pipeline/routing.py`)

Features computed per document after OCR:

| Feature | Source |
|---|---|
| `token_count` | Flan-T5 tokenizer length of the full OCR text |
| `confidence` | mean Tesseract word confidence |
| `char_count`, `line_count`, `word_count` | OCR text statistics |

Decision rule:

```
FAST  if token_count <= max_tokens_fast and confidence >= min_confidence_fast
FULL  otherwise
```

FAST documents go to the 4-bit model with `max_new_tokens=12`; FULL documents go to the fp32 model with `max_new_tokens=24`. The thresholds are **not guessed**: `evaluation/routing_analysis.py` takes the runs where every document was forced through FAST and every document through FULL, computes per-document accuracy under each, and sweeps a grid of `(max_tokens_fast, min_confidence_fast)` to find the policy that sends the most documents to FAST without losing more than 0.5 percentage points of field accuracy compared with all-FULL. The chosen thresholds go into `RoutingThresholds` and are re-validated with the `B-tuned` configuration.

`routing="all_fast"` and `routing="all_full"` bypass the rule; they exist for the ablation and the threshold analysis.

Outcome on this dataset (see `docs/RESULTS.md` §4): the 4-bit and fp32 models disagree on only 10 of 600 invoices, so the data-driven thresholds are `max_tokens_fast=525`, `min_confidence_fast=70`, which routes 99.8 % of documents to the 4-bit model. The initial estimate (300 tokens, confidence 85) used in configuration `B` routes 505 of 600. When `--configs all` is run, the runner derives the thresholds from that run's own all-FAST and all-FULL results before running `B-tuned`. The mechanism is kept because it is what makes the per-document comparison possible and because it becomes useful as soon as the two routes differ more than two precisions of the same small model.

### 2.7 Metrics (`pipeline/metrics.py`)

- `DocumentRecord` holds everything measured for one document.
- `ResourceMonitor` samples, four times a second from a background thread, the resident memory of the **whole process tree** (main process plus OCR workers, via `psutil`) and the GPU memory in use on the whole card (via NVML). The runner also records PyTorch's own peak allocation, which is reported as the pipeline's GPU memory because the NVML figure includes other programs. The earlier code measured RSS of the parent only and never looked at the GPU.
- `summarize_run` computes throughput from **wall-clock time** – the time from submitting the first document to receiving the last result – not from the sum of per-document times. The sum of per-document times cannot show the benefit of parallelism; wall time can.
- Cost = wall time × €0.35 per GPU-hour (a typical cloud price for an entry-level GPU instance), reported per document and per 100 documents.

### 2.8 Validation and human review (`pipeline/validate.py`)

**Total reconciliation** (`RECONCILE_TOTALS = True`). Error analysis showed that the model often answers the *net* amount when asked for the total, because the net is the first amount after the word "Total". The validation step collects the amounts on every line that contains *total*, *amount due*, *balance due*, *grand total* or *gross*. If two of them add up to a third (subtotal + tax = total, within 0.01), the third is the total; otherwise the model's answer stays. Percentages are ignored. This is the cross-field check that commercial invoice processors use.

**Review flag.** `review_reasons` returns why an invoice should be checked by a person: a missing field, a date or amount that cannot be parsed, a total that does not appear on any total line, or mean OCR confidence below 80. An invoice with no reasons is accepted automatically. Each record stores `needs_review` and the reasons; every run reports the straight-through rate (share accepted), the accuracy of accepted invoices and how many wrong invoices the flag caught.

Both steps run in A and in B, after the LLM, so they do not affect the A/B comparison.

## 3. Architecture A – baseline sequential pipeline (`pipeline/arch_a.py`)

```
for each document:
    OCR (Tesseract, main process, row text)            ~0.59 s
    embedding filter (MiniLM, one encode call)         ~0.02 s
    LLM: three generate calls, fp32                    ~0.51 s
    normalise answers, total reconciliation, review flag
```

Everything happens in one process, on one document at a time, in full precision, without cache. Per-document latency and service time are the same thing because nothing overlaps. This is the reference the optimized architecture is compared against.

The model and the kernels are warmed up once with a dummy document before timing starts, so the first real document does not pay CUDA initialisation. The same warm-up is applied to B.

## 4. Architecture B – optimized pipeline (`pipeline/arch_b.py`)

```
                    ┌───────────────┐
 documents ───────▶ │ cache lookup  │──hit──▶ record (no compute)
                    └──────┬────────┘
                           │ miss
            ┌──────────────┴──────────────┐
            │ ProcessPoolExecutor (N OCR  │   CPU-bound, N documents at once
            │ workers, Tesseract)         │
            └──────────────┬──────────────┘
          OCR results arrive in completion order (asyncio.as_completed)
                           │
              group into batches of `batch_size`
                           │
            ┌──────────────┴──────────────┐
            │ GPU stage (one thread)      │
            │  1. batched MiniLM encode   │
            │  2. features + routing      │
            │  3. FAST docs → 4-bit LLM   │  one generate call per route
            │     FULL docs → fp32 LLM    │
            │  4. total check, review flag│
            │  5. write cache             │
            └──────────────┬──────────────┘
                           ▼
                        records
```

The pieces and the reason each exists:

**asyncio event loop.** The loop owns the schedule. OCR jobs are submitted with `loop.run_in_executor(process_pool, run_ocr, path)`, GPU batches with `loop.run_in_executor(gpu_thread, process_batch, batch)`. The loop itself never blocks on either, so while the GPU thread is busy with batch *k*, the OCR workers are already producing batch *k+1*. This is the overlap that a plain sequential loop cannot achieve. `asyncio` on its own would **not** help – OCR and inference are CPU/GPU-bound, not I/O-bound – which is why both are pushed into executors, as the supervisor advised.

**Process pool for OCR.** Tesseract is a C++ binary driven through subprocess calls; Python threads would be serialised by the GIL around the call overhead and the image decoding, so processes are used. Each worker initialises Tesseract once (`configure_worker`) and imports only `pytesseract` and `Pillow`, never PyTorch, so the workers stay small (~60 MB each). The ablations `B-1worker` and `B-6workers` measure how far OCR parallelism scales on 12 logical cores.

**One GPU thread.** All GPU work is serialised through a `ThreadPoolExecutor(max_workers=1)`. Running two `generate` calls concurrently on one 4 GB GPU would not make them faster and would risk out-of-memory; serialising them keeps the GPU saturated by *batching* instead of by concurrency.

**Batching.** A batch of 8 documents becomes 24 prompts in one `generate` call and one MiniLM encode call. Padding is applied per batch. `B-nobatch` (batch size 1) isolates the gain.

**Routing.** See §2.6. `B-allfull` and `B-allfast` isolate the gain and cost of the quantized route.

**Cache.** Looked up before any compute. `B` is always measured cold (cache cleared before each run) and then immediately again warm (`B-warm`), so both the first-pass cost and the steady-state benefit are reported. `B-nocache` shows the lookup overhead on a cold pass is negligible.

**Two models resident at once.** fp32 Flan-T5 (~1 GB), 4-bit Flan-T5 (~0.2 GB) and MiniLM (~0.1 GB) fit together in the 4 GB of the RTX 3050 with room for activations. When a routing policy uses only one route, only that model is loaded.

### 4.1 Latency in a batched system

Architecture B reports two per-document times:

- **service time** = OCR duration + the document's share of its batch's embedding time + its share of its route's LLM time. This is the compute actually spent on the document and is directly comparable with A.
- **latency** = time from submission to result. Because the evaluation submits all 600 documents at once, latency includes queueing and grows with position in the queue; the p95 latency of B is therefore close to its wall time. When documents arrive one by one, B as benchmarked waits until 8 invoices are read before the GPU starts, so at low arrival rates an invoice can wait for the next seven; §4.2 describes the streaming mode that measures this and the dynamic-batching variant that removes the wait.

Throughput (documents per minute of wall time) is the headline metric for B, since that is what batching and parallelism improve.

### 4.2 Streaming mode and dynamic batching (`run_stream`)

`ArchitectureA.run_stream` and `ArchitectureB.run_stream` process invoices that arrive over time instead of all at once (used by `evaluation/latency_experiment.py`). Each invoice has an arrival offset; A waits for it and then processes invoices in arrival order. B submits each invoice's OCR job at its arrival time and keeps at most one GPU batch in flight. A batch starts when the GPU is free and one of these holds:

- 8 invoices are read (`batch_size`), or
- no more invoices are coming, or
- the oldest invoice has waited `max_batch_wait_s` since its OCR finished.

`max_batch_wait_s = inf` reproduces B as benchmarked (wait for a full batch). `max_batch_wait_s = 0` is dynamic batching: when the GPU is free it takes every invoice already read. Under light load batches are then small and results come back quickly; under heavy load invoices accumulate while the GPU is busy, so batches grow by themselves. The OCR worker processes are started before the clock starts, so the first invoice does not pay process start-up.

## 5. What changed from the earlier code and why it mattered

| Earlier behaviour | Effect on results | Now |
|---|---|---|
| Prompt asked Flan-T5 for JSON; T5 cannot emit `{}` | LLM output discarded, regex fallback produced every answer | Question-answer prompts, no fallback |
| `accuracy_check.py` scored the wrapper dict instead of `["result"]` | every field empty → near-zero accuracy | scorer receives `record.prediction` |
| `extract_number` kept only the first digit group (`$6,270` → `6`) | totals matched on the first digit | full numeric parsing with separator rules |
| A used 0.7 similarity, B used 0.9 | A and B scored by different rules | one scorer, one threshold (≥ 90) |
| `sorted()` on file names | "first 50" were invoices 1, 10, 100–143 | numeric ordering by invoice index |
| ground truth for invoice 1 named `test_invoice.json` | invoice 1 always skipped | renamed to `test_invoice1.json` |
| Architecture A ran on CPU in fp32, B on GPU | most of B's speed-up was GPU vs CPU | both on the GPU |
| `main_optimized_4bit.py` used `load_in_8bit=True` | no 4-bit experiment existed | precision is a parameter: fp32/fp16/bf16/8bit/4bit |
| throughput = docs ÷ sum of per-doc times | parallelism could never show a gain | throughput from wall time |
| memory sampled in the parent only, after the run | model and worker memory invisible | process-tree RSS + GPU, sampled during the run |
| cache keyed by file name, values via `eval` | stale and unsafe | SHA-256 of bytes + config namespace, JSON |
| routing on character count only | thresholds unjustified | token count + OCR confidence, thresholds from data |
| OCR text in Tesseract's block order | in tables, `Total` and its amount far apart; totals 32 % right on real invoices | text rebuilt row by row from word positions |
| model's total taken as is | net amount returned instead of the amount due | subtotal + tax = total reconciliation |
| no notion of confidence in the output | every answer treated as final | review flag with reasons; straight-through rate reported |
| results written to fixed folders | every run overwrote the previous one | a new `outputs/<date>_<time>_<label>/` folder per run, with log, settings and metrics |
