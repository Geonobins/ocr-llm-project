import asyncio
import math
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from pipeline.cache import ResultCache
from pipeline.config import CACHE_PATH, MAX_NEW_TOKENS_FAST, MAX_NEW_TOKENS_FULL, OCR_LAYOUT, RECONCILE_TOTALS, ArchBConfig
from pipeline.dataset import Document
from pipeline.embeddings import LineSelector
from pipeline.llm import FieldExtractor, load_tokenizer
from pipeline.metrics import DocumentRecord
from pipeline.ocr import OcrResult, configure_worker, run_ocr
from pipeline.routing import FAST, FULL, compute_features, decide_route
from pipeline.validate import finalize, review_reasons

MAX_NEW_TOKENS = {FAST: MAX_NEW_TOKENS_FAST, FULL: MAX_NEW_TOKENS_FULL}
WARMUP_TEXT = "Invoice Number: INV-0000\nDate: January 1, 2024\nTotal: $1.00"


class ArchitectureB:
    name = "B"

    def __init__(self, config: ArchBConfig | None = None, device: str = "cuda", cache_path=None):
        self.config = config or ArchBConfig()
        self.device = device
        self.tokenizer = load_tokenizer()
        self.selector = LineSelector(device)
        self.extractors = {}
        if self.config.routing != "all_full":
            self.extractors[FAST] = FieldExtractor(self.config.fast_precision, device, self.tokenizer)
        if self.config.routing != "all_fast":
            if self.config.full_precision == self.config.fast_precision and FAST in self.extractors:
                self.extractors[FULL] = self.extractors[FAST]
            else:
                self.extractors[FULL] = FieldExtractor(self.config.full_precision, device, self.tokenizer)
        self.cache = ResultCache(cache_path or CACHE_PATH) if self.config.use_cache else None
        self.namespace = self.config.namespace()
        self.warmup()

    def warmup(self) -> None:
        contexts = self.selector.select_batch([WARMUP_TEXT])
        for extractor in set(self.extractors.values()):
            extractor.extract_batch(contexts, 4)

    def _lookup(self, doc: Document, truth: dict) -> DocumentRecord | None:
        if self.cache is None:
            return None
        started = time.perf_counter()
        key = ResultCache.make_key(doc.image_path, self.namespace)
        payload = self.cache.get(key)
        if payload is None:
            return None
        elapsed = time.perf_counter() - started
        return DocumentRecord(
            name=doc.name,
            index=doc.index,
            source=doc.source,
            prediction=payload["prediction"],
            truth=truth,
            route="CACHE",
            cached=True,
            service_time=elapsed,
            latency=elapsed,
            features=payload.get("features", {}),
        )

    def _store(self, doc: Document, record: DocumentRecord) -> None:
        if self.cache is None or record.error:
            return
        key = ResultCache.make_key(doc.image_path, self.namespace)
        self.cache.put(key, {"prediction": record.prediction, "features": record.features, "route": record.route})

    def _token_counts(self, texts: list[str]) -> list[int]:
        encoded = self.tokenizer(texts, truncation=False)["input_ids"]
        return [len(ids) for ids in encoded]

    def _process_batch(self, batch: list[tuple[Document, OcrResult, dict, float]]) -> list[DocumentRecord]:
        texts = [ocr.text for _, ocr, _, _ in batch]
        mark = time.perf_counter()
        contexts = self.selector.select_batch(texts)
        embed_each = (time.perf_counter() - mark) / len(batch)
        token_counts = self._token_counts(texts)
        features = [
            compute_features(ocr.text, ocr.confidence, ocr.word_count, tokens)
            for (_, ocr, _, _), tokens in zip(batch, token_counts)
        ]
        routes = [decide_route(feature, self.config.thresholds, self.config.routing) for feature in features]
        predictions: list[dict | None] = [None] * len(batch)
        llm_each: list[float] = [0.0] * len(batch)
        for route in (FAST, FULL):
            positions = [i for i, r in enumerate(routes) if r == route]
            if not positions:
                continue
            mark = time.perf_counter()
            outputs = self.extractors[route].extract_batch([contexts[i] for i in positions], MAX_NEW_TOKENS[route])
            elapsed = (time.perf_counter() - mark) / len(positions)
            for position, output in zip(positions, outputs):
                predictions[position] = output
                llm_each[position] = elapsed
        finalized = [finalize(predictions[i] or {}, ocr.text, RECONCILE_TOTALS) for i, (_, ocr, _, _) in enumerate(batch)]
        finished = time.perf_counter()
        records = []
        for i, (doc, ocr, truth, submitted) in enumerate(batch):
            prediction, total_source = finalized[i]
            feature_row = features[i].to_dict()
            feature_row["total_source"] = total_source
            reasons = review_reasons(prediction, ocr.text, ocr.confidence)
            feature_row["needs_review"] = bool(reasons)
            feature_row["review_reasons"] = ";".join(reasons)
            record = DocumentRecord(
                name=doc.name,
                index=doc.index,
                source=doc.source,
                prediction=prediction,
                truth=truth,
                route=routes[i],
                ocr_time=ocr.duration,
                embed_time=embed_each,
                llm_time=llm_each[i],
                service_time=ocr.duration + embed_each + llm_each[i],
                latency=finished - submitted,
                features=feature_row,
            )
            self._store(doc, record)
            records.append(record)
        return records

    async def run_async(self, documents: list[Document], progress=None) -> list[DocumentRecord]:
        loop = asyncio.get_running_loop()
        if self.cache is not None:
            self.cache.hits = 0
            self.cache.misses = 0
        submitted = time.perf_counter()
        records: dict[int, DocumentRecord] = {}
        truths = {doc.index: doc.load_truth() for doc in documents}
        pending: list[Document] = []
        for doc in documents:
            hit = self._lookup(doc, truths[doc.index])
            if hit is not None:
                records[doc.index] = hit
            else:
                pending.append(doc)
        done = len(records)
        if progress and done:
            progress(done, len(documents), None)

        async def ocr_job(doc: Document, pool):
            try:
                result = await loop.run_in_executor(pool, run_ocr, doc.image_path, OCR_LAYOUT)
                return doc, result, None
            except Exception as exc:
                return doc, None, f"{type(exc).__name__}: {exc}"

        gpu_executor = ThreadPoolExecutor(max_workers=1)
        with ProcessPoolExecutor(max_workers=self.config.ocr_workers, initializer=configure_worker) as pool:
            tasks = [asyncio.ensure_future(ocr_job(doc, pool)) for doc in pending]
            batch: list[tuple[Document, OcrResult, dict, float]] = []
            gpu_jobs = []
            for completed in asyncio.as_completed(tasks):
                doc, result, error = await completed
                if error:
                    now = time.perf_counter()
                    records[doc.index] = DocumentRecord(
                        name=doc.name,
                        index=doc.index,
                        source=doc.source,
                        prediction={},
                        truth=truths[doc.index],
                        route="ERROR",
                        service_time=now - submitted,
                        latency=now - submitted,
                        error=error,
                    )
                    continue
                batch.append((doc, result, truths[doc.index], submitted))
                if len(batch) >= self.config.batch_size:
                    gpu_jobs.append(loop.run_in_executor(gpu_executor, self._process_batch, batch))
                    batch = []
            if batch:
                gpu_jobs.append(loop.run_in_executor(gpu_executor, self._process_batch, batch))
            for job in asyncio.as_completed(gpu_jobs):
                for record in await job:
                    records[record.index] = record
                    done += 1
                if progress:
                    progress(done, len(documents), None)
        gpu_executor.shutdown(wait=True)
        return [records[doc.index] for doc in documents]

    def run(self, documents: list[Document], progress=None) -> list[DocumentRecord]:
        return asyncio.run(self.run_async(documents, progress))

    async def run_stream_async(self, documents: list[Document], arrivals: list[float], max_batch_wait_s: float = math.inf) -> list[DocumentRecord]:
        loop = asyncio.get_running_loop()
        truths = {doc.index: doc.load_truth() for doc in documents}
        ready: asyncio.Queue = asyncio.Queue()
        records: dict[int, DocumentRecord] = {}
        gpu_executor = ThreadPoolExecutor(max_workers=1)
        with ProcessPoolExecutor(max_workers=self.config.ocr_workers, initializer=configure_worker) as pool:
            list(pool.map(time.sleep, [0.5] * self.config.ocr_workers))
            start = time.perf_counter()

            async def arrive(doc: Document, offset: float):
                arrived = start + offset
                delay = arrived - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
                try:
                    result = await loop.run_in_executor(pool, run_ocr, doc.image_path, OCR_LAYOUT)
                    await ready.put((doc, result, None, arrived))
                except Exception as exc:
                    await ready.put((doc, None, f"{type(exc).__name__}: {exc}", arrived))

            feeders = [asyncio.ensure_future(arrive(doc, offset)) for doc, offset in zip(documents, arrivals)]
            total = len(documents)
            received = 0
            waiting: list[tuple[tuple, float]] = []
            next_item = None
            batch_job = None
            while received < total or waiting or batch_job is not None:
                if batch_job is None and waiting:
                    oldest = time.perf_counter() - waiting[0][1]
                    if len(waiting) >= self.config.batch_size or received == total or oldest >= max_batch_wait_s:
                        taken = [item for item, _ in waiting[: self.config.batch_size]]
                        waiting = waiting[self.config.batch_size:]
                        batch_job = loop.run_in_executor(gpu_executor, self._process_batch, taken)
                        continue
                watch = []
                if received < total:
                    if next_item is None:
                        next_item = asyncio.ensure_future(ready.get())
                    watch.append(next_item)
                if batch_job is not None:
                    watch.append(batch_job)
                timeout = None
                if batch_job is None and waiting and max_batch_wait_s != math.inf:
                    timeout = max(0.0, max_batch_wait_s - (time.perf_counter() - waiting[0][1]))
                finished, _ = await asyncio.wait(watch, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
                if next_item is not None and next_item in finished:
                    doc, result, error, arrived = next_item.result()
                    next_item = None
                    received += 1
                    if error:
                        now = time.perf_counter()
                        records[doc.index] = DocumentRecord(
                            name=doc.name,
                            index=doc.index,
                            source=doc.source,
                            prediction={},
                            truth=truths[doc.index],
                            route="ERROR",
                            service_time=now - arrived,
                            latency=now - arrived,
                            error=error,
                        )
                    else:
                        waiting.append(((doc, result, truths[doc.index], arrived), time.perf_counter()))
                if batch_job is not None and batch_job in finished:
                    for record in batch_job.result():
                        records[record.index] = record
                    batch_job = None
            await asyncio.gather(*feeders)
        gpu_executor.shutdown(wait=True)
        return [records[doc.index] for doc in documents]

    def run_stream(self, documents: list[Document], arrivals: list[float], max_batch_wait_s: float = math.inf) -> list[DocumentRecord]:
        return asyncio.run(self.run_stream_async(documents, arrivals, max_batch_wait_s))

    def clear_cache(self) -> None:
        if self.cache is not None:
            self.cache.clear()

    def cache_stats(self) -> dict:
        if self.cache is None:
            return {"cache_hits": 0, "cache_misses": 0, "cache_entries": 0}
        return {"cache_hits": self.cache.hits, "cache_misses": self.cache.misses, "cache_entries": self.cache.size()}
