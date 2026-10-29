import time

from pipeline.config import MAX_NEW_TOKENS_FULL, OCR_LAYOUT, RECONCILE_TOTALS, ArchAConfig
from pipeline.dataset import Document
from pipeline.embeddings import LineSelector
from pipeline.llm import FieldExtractor
from pipeline.metrics import DocumentRecord
from pipeline.ocr import configure_worker, run_ocr
from pipeline.routing import compute_features
from pipeline.validate import finalize, review_reasons

WARMUP_TEXT = "Invoice Number: INV-0000\nDate: January 1, 2024\nTotal: $1.00"


class ArchitectureA:
    name = "A"

    def __init__(self, config: ArchAConfig | None = None, device: str = "cuda"):
        self.config = config or ArchAConfig()
        configure_worker()
        self.selector = LineSelector(device)
        self.extractor = FieldExtractor(self.config.precision, device)
        self.warmup()

    def warmup(self) -> None:
        context = self.selector.select(WARMUP_TEXT)
        self.extractor.extract_sequential(context, 4)

    def process(self, doc: Document) -> DocumentRecord:
        truth = doc.load_truth()
        started = time.perf_counter()
        try:
            ocr = run_ocr(doc.image_path, OCR_LAYOUT)
            mark = time.perf_counter()
            context = self.selector.select(ocr.text)
            embed_time = time.perf_counter() - mark
            mark = time.perf_counter()
            prediction = self.extractor.extract_sequential(context, MAX_NEW_TOKENS_FULL)
            llm_time = time.perf_counter() - mark
            prediction, total_source = finalize(prediction, ocr.text, RECONCILE_TOTALS)
            latency = time.perf_counter() - started
            features = compute_features(
                ocr.text, ocr.confidence, ocr.word_count, self.extractor.count_tokens(ocr.text)
            ).to_dict()
            features["total_source"] = total_source
            reasons = review_reasons(prediction, ocr.text, ocr.confidence)
            features["needs_review"] = bool(reasons)
            features["review_reasons"] = ";".join(reasons)
            return DocumentRecord(
                name=doc.name,
                index=doc.index,
                source=doc.source,
                prediction=prediction,
                truth=truth,
                route="SEQUENTIAL",
                ocr_time=ocr.duration,
                embed_time=embed_time,
                llm_time=llm_time,
                service_time=latency,
                latency=latency,
                features=features,
            )
        except Exception as exc:
            latency = time.perf_counter() - started
            return DocumentRecord(
                name=doc.name,
                index=doc.index,
                source=doc.source,
                prediction={},
                truth=truth,
                route="SEQUENTIAL",
                service_time=latency,
                latency=latency,
                error=f"{type(exc).__name__}: {exc}",
            )

    def run_stream(self, documents: list[Document], arrivals: list[float]) -> list[DocumentRecord]:
        records = []
        start = time.perf_counter()
        for doc, offset in zip(documents, arrivals):
            arrived = start + offset
            delay = arrived - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            record = self.process(doc)
            record.latency = time.perf_counter() - arrived
            records.append(record)
        return records

    def run(self, documents: list[Document], progress=None) -> list[DocumentRecord]:
        records = []
        for position, doc in enumerate(documents, start=1):
            record = self.process(doc)
            records.append(record)
            if progress:
                progress(position, len(documents), record)
        return records
