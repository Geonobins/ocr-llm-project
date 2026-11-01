# Study guide

Questions a supervisor or committee is likely to ask about the solution, with the place in the code or the measurements where the answer comes from. Use it to check that you can explain every part of the system in your own words.

## The problem and the two architectures

**Why is an OCR → LLM pipeline still needed when vision-language models can read documents directly?**
The hardware constraint, cost and control. The thesis GPU has 4 GB in total. The OCR → LLM pipeline needs 1.07 GB (A) to 2.0 GB (B) of GPU memory at peak (benchmark `summary.md`, "Pipeline GPU peak"), keeps the OCR text inspectable, and lets the text stage be cached, validated and routed. A 2-billion-parameter VLM needs more than a gigabyte for its weights even in 4-bit, plus the activations of a high-resolution scan; whether it fits next to everything else and how fast it is must be measured – that is the recommended follow-up experiment, and until then the argument should cite the VLM literature rather than claim numbers. Architecture B spends about 0.7 s of compute per invoice.

**What is the single difference between A and B that matters for accuracy?**
None by design. Same OCR, same line selector, same prompts, same post-processing, same scorer. The only thing that can change an answer is the precision of the model on the FAST route (4-bit instead of fp32). The size of that effect is `B-allfull` F1 minus `B-allfast` F1.

**Why Flan-T5-base and not a larger model?**
It fits the thesis hardware in every precision, including fp32 (≈1 GB), which makes the A/B comparison about the architecture rather than about memory starvation. Its weakness – only 512 input tokens – is exactly what the embedding filter addresses, so the OCR → embeddings → LLM structure in the task description has a real function.

## OCR

**Why Tesseract and not EasyOCR?**
Measured on all 600 invoices with the same downstream and scorer (`evaluation/ocr_comparison.py`, folder `outputs/2026-10-20_01-52-39_ocr_comparison`): Tesseract reaches F1 0.939, EasyOCR 0.932. On the real scans Tesseract needs 0.89 s per invoice on one CPU thread, EasyOCR 8.0 s on the GPU, with a GPU memory peak of 3.9 GB – almost the whole 4 GB card, so it could not share the GPU with the language model. Tesseract also returns the word-level confidence that the routing needs. The gap in accuracy is mainly the total on real invoices (the true total is in Tesseract's text for 97 % of real invoices, in EasyOCR's for 92 %).

**What does OCR confidence mean and how is it used?**
Tesseract's per-word confidence (0–100) averaged over the words of the page (`pipeline/ocr.py`, `_assemble_text`). Low confidence means noisy text; the router sends such documents to the full-precision model (`pipeline/routing.py`).

**Why one OCR call instead of `image_to_string` plus `image_to_data`?**
OCR is the dominant CPU cost (0.25–1.2 s per page). `image_to_data` gives both text and confidence in one pass; the text is rebuilt from the word/line/paragraph structure.

## Embeddings

**What is the role of the embedding step?**
Context selection. Flan-T5 can read 512 tokens; a scanned invoice OCRs to up to ~750 tokens. MiniLM ranks the lines by similarity to "invoice number", "invoice date", "total amount due", keeps the best six per field plus the following line, and preserves order (`pipeline/embeddings.py`). Short documents are passed through unchanged.

**Why "plus the following line"?**
On invoices, the label and the value are frequently on consecutive OCR lines (`Total Due` / `1574`). The similarity search finds the label; the next line carries the value.

**How does B batch embeddings?**
`select_batch` concatenates the lines of all documents in the batch, encodes them in one call with `batch_size=256`, and splits the result by document.

## LLM and quantization

**Why are the three fields asked as three questions?**
Flan-T5 was instruction-tuned on question answering. Asking for JSON fails for a hard technical reason: the T5 SentencePiece vocabulary has no `{` or `}` tokens, so the model cannot emit JSON at all. The earlier code never noticed because a regex fallback produced the answers (`docs/ARCHITECTURE.md` §2.3).

**How is the model quantized?**
With `bitsandbytes` through `transformers.BitsAndBytesConfig` (`pipeline/llm.py`). 8-bit uses LLM.int8 (Dettmers et al., 2022): weights stored in int8, outlier features computed in fp16. 4-bit uses NF4 with double quantization (Dettmers et al., 2023, QLoRA): weights stored in a 4-bit "normal-float" grid, de-quantized to fp16 on the fly for each matrix multiplication.

**Does quantization make inference faster?**
Not for a 248 M-parameter model on this GPU. Full benchmark, every invoice on one model: fp32 126.1 s (LLM 0.115 s per invoice), fp16 123.5 s (0.079 s), 4-bit 125.8 s (0.115 s), 8-bit 167.8 s (0.258 s). The de-quantization kernels cost as much as the memory traffic they save; the model is too small to be memory-bandwidth-bound. What 4-bit buys is memory: 1102 MB pipeline GPU peak against 1645 MB for fp32. fp16 is the best option here – fastest and smallest (1049 MB) at the same F1. Say this plainly in the thesis; the LLM.int8() paper (Dettmers et al., 2022) also reports that its method is slower for small models.

**Then why route anything to the 4-bit model?**
Memory headroom. Two models and the embedder are resident at once on 4 GB; the FAST route also generates fewer tokens (`max_new_tokens=12` vs 24). The routing analysis shows how much accuracy the 4-bit route costs and on which documents, and the thesis should present routing as the mechanism that makes the trade-off selectable per document rather than global.

## Cache

**What is the cache key?**
SHA-256 of the image bytes plus a namespace string containing the model name, both precisions, the routing policy and the thresholds (`pipeline/cache.py`, `ArchBConfig.namespace`). A renamed file hits; a modified image or a changed configuration misses.

**Why SQLite rather than Redis?**
Single workstation, no server to administer, durable across restarts, lookup ≈ 1 ms including hashing the image (0.59 s for all 600 invoices). Redis is the right choice when several machines share a cache; that is future work, not this setup.

**How is the warm-cache benefit measured without hiding the cold cost?**
`B` is always run cold (cache cleared), then the same documents are run again immediately (`B-warm`). Both are reported.

## Routing

**What are the features and the rule?**
Token count of the OCR text (Flan-T5 tokenizer) and mean OCR confidence. FAST if `token_count ≤ max_tokens_fast` and `confidence ≥ min_confidence_fast`, FULL otherwise (`pipeline/routing.py`).

**Where do the thresholds come from?**
From data, not estimation. `evaluation/routing_analysis.py` uses the all-FAST and all-FULL runs to know, for every document, how each model scored and how long it took, then sweeps a grid of thresholds and picks the policy with the largest FAST share whose accuracy stays within 0.5 pp of all-FULL. `routing_analysis.md` in the benchmark folder contains the grid, the accuracy per token bin and per confidence bin, and the chosen pair (525 tokens, confidence 70). `B-tuned` validates it on three fresh runs (F1 0.939).

## Asynchronous execution

**Why does `asyncio` alone not speed up OCR or inference?**
Both are CPU/GPU-bound. An `await` on a blocking function blocks the event loop. The loop therefore hands OCR to a `ProcessPoolExecutor` and GPU work to a one-thread `ThreadPoolExecutor` through `loop.run_in_executor` (`pipeline/arch_b.py`, `run_async`), and only orchestrates: collects OCR results as they complete, forms batches, dispatches them.

**Why processes for OCR and a thread for the GPU?**
Tesseract work does not release the GIL in a useful way, so threads would not scale; processes do. PyTorch releases the GIL inside kernels, and a single thread is enough because the GPU is kept busy by batching; two concurrent `generate` calls on a 4 GB GPU would risk out-of-memory without being faster.

**How do you know the overlap actually happens?**
Wall time of `B` is less than the sum of OCR time and LLM time over the documents, and `B-1worker` and `B-nobatch` show what happens when one side of the overlap is removed: 338.4 s and 248.8 s instead of 119.4 s (benchmark `summary.md`).

## Metrics

**Why is F1 used instead of accuracy?**
The pipeline may answer `null`. Accuracy would count a null on a null truth and a correct value the same way and would not separate "missed" from "wrong". Each field gets TP/FP/FN/TN, a wrong value counts as FP and FN, and precision, recall and F1 are computed per field and micro-averaged (`evaluation/scorer.py`).

**Why strict matching, and where is fuzzy matching?**
The headline score is strict: an invoice number must be the same after removing labels and punctuation, a date the same calendar day with day, month and year present, a total the same amount to the cent. One wrong character in an invoice number is a different invoice, so it counts as wrong. Fuzzy matching (RapidFuzz `ratio ≥ 90`, roughly one substitution in a ten-character string) is reported as a secondary, relaxed score, as the supervisor asked; it also allows a day/month swap when a date is written ambiguously (`07/06/2025`). For A and B both scores are identical (F1 0.939): fuzzy matching never rescued an answer of either architecture; for the VLM it accepts two answers with one extra digit. Amounts are never matched fuzzily, because `6,270` and `6,999` are 80% similar as strings and completely different as amounts.

**Why was the scorer changed after the benchmark?**
An external review (1 Nov 2026) showed that the original scorer accepted wrong answers: equal digit sequences counted as equal invoice numbers (`009` = `GALT-009`), and day/month swaps were allowed for every date. Every stored run was re-scored without re-running a model; 9 distinct answers changed, F1 moved from 0.941 to 0.939, and no conclusion changed (`docs/RESULTS.md` §11).

**Why are latency and service time both reported for B?**
In a batched system the time a document waits in the queue depends on how many documents were submitted with it. Service time is the compute spent on the document and is comparable with A; latency includes queueing and, because the benchmark submits all 600 at once, grows with queue position. Throughput is the metric that batching and parallelism actually improve.

**How is throughput computed?**
Documents ÷ wall-clock time of the run. Summing per-document times instead (as the earlier code did) can never show a gain from parallelism.

**How is memory measured?**
A background thread samples, four times per second, the resident memory of the main process plus all OCR workers (`psutil`) and the GPU memory in use (NVML). Peak values are reported (`pipeline/metrics.py`, `ResourceMonitor`).

**How is cost computed?**
Wall time × €0.35 per GPU-hour, a typical on-demand price for an entry-level cloud GPU. The constant is in `pipeline/config.py`; the thesis should state it and note that the ratio between A and B is independent of the constant.

**Why three runs?**
To report mean ± standard deviation. Generation is greedy, so accuracy is stable across runs; timing varies with the OS scheduler and GPU clocks, and the standard deviation shows how much.

## Honest limitations to state in the thesis

- Flan-T5-base is a small model; on the real invoices it makes field errors that a larger model would not. The thesis measures architecture, not model quality, and both architectures share the errors.
- The synthetic half of the dataset is easy; results are reported per subset for this reason.
- Some numeric dates are ambiguous in the ground truth itself (`07/06/2025`); only the relaxed score tolerates a day/month swap for them, the strict headline score does not.
- The routing thresholds were chosen on the same 600 invoices that report accuracy; there is no held-out set. Both routes are practically equally accurate, so this cannot inflate accuracy.
- Latency of B in the benchmark includes queueing because all documents are submitted at once. With invoices arriving one at a time (`evaluation/latency_experiment.py`), B as benchmarked is slower than A at light load (median 7.9 s vs 1.7 s at 30 per minute) because it waits for a batch of 8; B with dynamic batching answers in about 1–2 s at every load.
- Quantization reduces memory, not latency, at this model size on this GPU.
