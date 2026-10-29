import statistics
import threading
import time
from dataclasses import dataclass, field, asdict

import psutil

from pipeline.config import GPU_HOURLY_COST_EUR

try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    pynvml = None
    _NVML_HANDLE = None


@dataclass
class DocumentRecord:
    name: str
    index: int
    source: str
    prediction: dict
    truth: dict
    route: str = "SEQUENTIAL"
    cached: bool = False
    ocr_time: float = 0.0
    embed_time: float = 0.0
    llm_time: float = 0.0
    service_time: float = 0.0
    latency: float = 0.0
    features: dict = field(default_factory=dict)
    error: str = ""

    def to_row(self) -> dict:
        row = {
            "name": self.name,
            "index": self.index,
            "source": self.source,
            "route": self.route,
            "cached": self.cached,
            "ocr_time": round(self.ocr_time, 4),
            "embed_time": round(self.embed_time, 4),
            "llm_time": round(self.llm_time, 4),
            "service_time": round(self.service_time, 4),
            "latency": round(self.latency, 4),
            "error": self.error,
        }
        for key, value in self.features.items():
            row[f"feat_{key}"] = value
        for key, value in self.prediction.items():
            row[f"pred_{key}"] = value
        for key, value in self.truth.items():
            row[f"truth_{key}"] = value
        return row


def gpu_memory_used_mb() -> float:
    if _NVML_HANDLE is None:
        return 0.0
    info = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)
    return info.used / (1024 * 1024)


def process_tree_rss_mb() -> float:
    current = psutil.Process()
    total = current.memory_info().rss
    for child in current.children(recursive=True):
        try:
            total += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total / (1024 * 1024)


class ResourceMonitor:
    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self.rss_samples: list[float] = []
        self.gpu_samples: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self.baseline_gpu = gpu_memory_used_mb()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.rss_samples.append(process_tree_rss_mb())
            self.gpu_samples.append(gpu_memory_used_mb())
            self._stop.wait(self.interval)

    def start(self) -> "ResourceMonitor":
        self._thread.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join()
        self.rss_samples.append(process_tree_rss_mb())
        self.gpu_samples.append(gpu_memory_used_mb())
        return {
            "peak_rss_mb": round(max(self.rss_samples), 1),
            "mean_rss_mb": round(statistics.fmean(self.rss_samples), 1),
            "peak_gpu_mb": round(max(self.gpu_samples), 1),
            "baseline_gpu_mb": round(self.baseline_gpu, 1),
        }


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[position]


def summarize_run(records: list[DocumentRecord], wall_time: float, resources: dict, extra: dict | None = None) -> dict:
    processed = [r for r in records if not r.error]
    uncached = [r for r in processed if not r.cached]
    latencies = [r.latency for r in processed]
    service = [r.service_time for r in processed]
    total_hours = wall_time / 3600
    total_cost = total_hours * GPU_HOURLY_COST_EUR
    docs = len(records)
    summary = {
        "documents": docs,
        "processed": len(processed),
        "errors": len(records) - len(processed),
        "cached": sum(1 for r in processed if r.cached),
        "wall_time_s": round(wall_time, 3),
        "throughput_docs_per_min": round(docs / wall_time * 60, 3) if wall_time else 0.0,
        "mean_latency_s": round(statistics.fmean(latencies), 4) if latencies else 0.0,
        "p50_latency_s": round(percentile(latencies, 0.5), 4),
        "p95_latency_s": round(percentile(latencies, 0.95), 4),
        "mean_service_time_s": round(statistics.fmean(service), 4) if service else 0.0,
        "mean_ocr_time_s": round(statistics.fmean([r.ocr_time for r in uncached]), 4) if uncached else 0.0,
        "mean_embed_time_s": round(statistics.fmean([r.embed_time for r in uncached]), 4) if uncached else 0.0,
        "mean_llm_time_s": round(statistics.fmean([r.llm_time for r in uncached]), 4) if uncached else 0.0,
        "route_fast": sum(1 for r in processed if r.route == "FAST"),
        "route_full": sum(1 for r in processed if r.route == "FULL"),
        "cost_total_eur": round(total_cost, 5),
        "cost_per_doc_eur": round(total_cost / docs, 6) if docs else 0.0,
        "cost_per_100_docs_eur": round(total_cost / docs * 100, 4) if docs else 0.0,
    }
    summary.update(resources)
    if extra:
        summary.update(extra)
    return summary


class Stopwatch:
    def __init__(self):
        self.start = time.perf_counter()

    def lap(self) -> float:
        now = time.perf_counter()
        elapsed = now - self.start
        self.start = now
        return elapsed

    @staticmethod
    def now() -> float:
        return time.perf_counter()
