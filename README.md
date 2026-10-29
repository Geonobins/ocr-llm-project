# OCR → LLM invoice extraction: Architecture A vs Architecture B

Two pipelines that read an invoice image and return three fields (`invoice_number`, `date`, `total`):

- **Architecture A** – baseline sequential pipeline: OCR → embedding filter → LLM, one document at a time, full-precision model.
- **Architecture B** – optimized pipeline: parallel OCR workers, result cache, batched embeddings and LLM calls, complexity-based routing between a 4-bit quantized model and the full-precision model, orchestrated with `asyncio`.

Both architectures use the same OCR engine (Tesseract 5), the same embedding model (`all-MiniLM-L6-v2`), the same LLM (`google/flan-t5-base`), the same prompts and the same post-processing. The only differences are *how* the work is scheduled and *which precision* the model runs in, so any difference in latency, throughput, memory or cost is attributable to the architecture.

Documentation for studying the system:

| File | What it covers |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Data flow, every module, the design decisions and the measurements behind them |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Dataset, ground truth, metric definitions, experiment design, routing-threshold method |
| [docs/RESULTS.md](docs/RESULTS.md) | Measured results on the 600-document dataset with interpretation |
| [docs/STUDY_GUIDE.md](docs/STUDY_GUIDE.md) | Questions a committee may ask and where the answers come from |
| [docs/Architecture_Report.pdf](docs/Architecture_Report.pdf) | **Start here.** A vs B with diagrams, every change made, how to run, results – including the accuracy upgrade |
| [docs/Committee_QA.pdf](docs/Committee_QA.pdf) | Every committee and supervisor question, plus likely defence questions – short answers with the measured proof |
| [docs/Thesis_Outline.pdf](docs/Thesis_Outline.pdf) | Chapter plan for the thesis in the IACS order: what each chapter must answer, facts, figures, reading list |
| [outputs/2026-10-18_23-52-41_full_benchmark/summary.md](outputs/2026-10-18_23-52-41_full_benchmark/summary.md) | Final benchmark: all tables (mean ± std over 3 runs) |
| [outputs/2026-10-18_23-52-41_full_benchmark/figures/](outputs/2026-10-18_23-52-41_full_benchmark/figures/) | Final benchmark: all figures |

The repository holds the final experiment folders cited in the docs (full benchmark, latency, OCR comparison, VLM baseline and resolution probe) without their raw `log.txt`. Older benchmark archives, the legacy scripts and the local virtual environment are not tracked.

## Layout

```
pipeline/            the two architectures and their building blocks
  config.py          paths, model names, prompts, default thresholds, config dataclasses
  dataset.py         loads images + ground truth, numeric ordering, real/synthetic split
  ocr.py             Tesseract wrapper returning text, mean confidence, word/line counts; EasyOCR engine for the comparison
  normalize.py       canonical forms for invoice numbers, dates and amounts
  embeddings.py      MiniLM line selector (single-document and batched)
  llm.py             Flan-T5 loader for fp32/fp16/bf16/8-bit/4-bit, question-answer extraction
  cache.py           SQLite result cache keyed by image hash + configuration
  routing.py         complexity features and FAST/FULL decision
  validate.py        subtotal + tax = total reconciliation and review flags
  metrics.py         per-document records, resource monitor, run summary, cost model
  arch_a.py          Architecture A
  arch_b.py          Architecture B
evaluation/
  scorer.py          field normalization + fuzzy matching → TP/FP/FN/TN → precision, recall, F1
  run_experiments.py runs configurations × repetitions, writes per-document CSV and summary JSON
  report.py          aggregates the runs of an output folder into summary.md, summary.csv and figures
  output.py          new output folder per run, log capture, run_info.json
  routing_analysis.py derives routing thresholds from the all-FAST / all-FULL runs
  rescore.py         re-scores every stored run with the current scorer and ground truth
  relabel_totals.py  normalises the total labels of invoices 316-600 to the gross total (with backup and log)
  latency_experiment.py  latency of A, B and B with dynamic batching when invoices arrive one at a time
  ocr_comparison.py  Tesseract vs EasyOCR: speed, values found in the text, F1 through the same downstream
  vlm_baseline.py    Qwen2-VL-2B (4-bit) reading the invoice images directly, same scorer
invoice_images/      600 invoices (1–15 and 316–600 real, 16–315 synthetic)
ground_truth/        one JSON per invoice
ground_truth_backup_20261018/  original total labels before the gross-total correction
generate_dataset.py  synthetic invoice generator used for invoices 16–315
demo.py              runs A and B side by side on a sample of invoices
tests/               unit tests (pytest)
outputs/             final experiment folders cited in the docs
```

## Setup (Windows, NVIDIA GPU)

1. Install Tesseract 5 (`winget install UB-Mannheim.TesseractOCR`). The path is set in `pipeline/config.py`.
2. Create the environment:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.venv\Scripts\python -m pip install -r requirements.txt
```

3. Check the GPU is visible: `.venv\Scripts\python -c "import torch;print(torch.cuda.get_device_name(0))"`.

Model weights (~1 GB for Flan-T5-base and ~90 MB for MiniLM) download from the Hugging Face Hub. Runs are offline by default; add `--online` the first time on a new machine.

## Running

Every command below creates a **new folder** `outputs/<date>_<time>_<label>/` – nothing is ever overwritten. `outputs/LATEST.txt` holds the path of the newest one.

```powershell
# unit tests
.venv\Scripts\python -m pytest tests -q

# A and B side by side, printed and saved
.venv\Scripts\python demo.py --docs 30
.venv\Scripts\python demo.py --docs 100 --source real

# quick benchmark: 24 invoices, 1 run
.venv\Scripts\python -m evaluation.run_experiments --configs A B --runs 1 --limit 24

# both architectures, all 600 invoices, 3 runs
.venv\Scripts\python -m evaluation.run_experiments --configs A B --runs 3

# everything: A, B and all ablations, 3 runs, routing thresholds derived and validated (≈ 3 h)
.venv\Scripts\python -m evaluation.run_experiments --configs all --runs 3 --name full_benchmark

# give the folder your own label
.venv\Scripts\python -m evaluation.run_experiments --configs A B --runs 1 --name before_meeting

# finish an interrupted run in the same folder
.venv\Scripts\python -m evaluation.run_experiments --configs all --runs 3 --out outputs\<folder> --skip-existing

# latency with invoices arriving one at a time: A, B and B with dynamic batching (about 50 min)
.venv\Scripts\python -m evaluation.latency_experiment --docs 60 --runs 3

# Tesseract vs EasyOCR on all 600 invoices (needs: pip install easyocr)
.venv\Scripts\python -m evaluation.ocr_comparison

# vision-language model reading the images directly (Qwen2-VL-2B, 4-bit)
.venv\Scripts\python -m evaluation.vlm_baseline --tokens 1024

# open the newest results
notepad (Get-Content outputs\LATEST.txt)\summary.md
```

What each output folder contains:

| File | Content |
|---|---|
| `summary.md` | All tables (mean ± std over runs): speed, resources, cost, F1, per-field F1, human review, routing and cache |
| `summary.csv`, `runs.csv` | The same numbers per configuration and per run, for Excel |
| `figures/` | Throughput, service time, stage breakdown, memory, cost, F1, routing charts |
| `runs/<config>/run_<k>/documents.csv` | One row per invoice: timings, route, OCR confidence, tokens, prediction, truth, outcome per field, review flag and reasons |
| `runs/<config>/run_<k>/summary.json` | Metrics, scores per source, human-review metrics, configuration, hardware |
| `run_info.json` | Command, start/finish time, pipeline settings, hardware, library versions |
| `log.txt` | Everything printed during the run |
| `routing_analysis.md`, `routing_thresholds.json` | Only when B-allfast and B-allfull were run: the threshold study and the chosen thresholds |
| `comparison.csv` | Demo only: A's and B's answer for every invoice and field |

Metrics recorded per run: wall time, throughput, mean / p50 / p95 latency, service time, OCR / embedding / LLM time per invoice, peak RAM (process tree), peak GPU memory, model load time, cost per invoice and per 100, cache hits and misses, FAST / FULL routing counts, precision / recall / F1 per field and overall (all, real, synthetic), share of fully correct invoices, straight-through rate, accuracy of auto-accepted invoices, wrong invoices caught by the review flag.

For valid timings keep the laptop plugged in and idle during a run; accuracy is not affected by other use.

## Using the pipelines directly

```python
from pipeline.dataset import load_documents
from pipeline.arch_a import ArchitectureA
from pipeline.arch_b import ArchitectureB
from pipeline.config import ArchBConfig

documents = load_documents(limit=10)
baseline = ArchitectureA()
records_a = baseline.run(documents)

optimized = ArchitectureB(ArchBConfig(ocr_workers=3, batch_size=8))
records_b = optimized.run(documents)
print(records_b[0].prediction, records_b[0].route, records_b[0].service_time)
```
