from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_ROOT / "invoice_images"
GROUND_TRUTH_DIR = PROJECT_ROOT / "ground_truth"
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_PATH = RESULTS_DIR / "cache.sqlite"

TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
LLM_NAME = "google/flan-t5-base"
EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

FIELDS = ("invoice_number", "date", "total")

QUESTIONS = {
    "invoice_number": "What is the invoice number?",
    "date": "What is the invoice date?",
    "total": "What is the total amount due?",
}

QUERY_TEXTS = {
    "invoice_number": "invoice number",
    "date": "invoice date",
    "total": "total amount due",
}

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
SYNTHETIC_INDICES = range(16, 316)

MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS_FAST = 12
MAX_NEW_TOKENS_FULL = 24
SELECTOR_TOP_K = 6

GPU_HOURLY_COST_EUR = 0.35

OCR_LAYOUT = "rows"
RECONCILE_TOTALS = True
EXTRACTION_VERSION = f"layout={OCR_LAYOUT}|reconcile={RECONCILE_TOTALS}"


@dataclass(frozen=True)
class RoutingThresholds:
    max_tokens_fast: int = 300
    min_confidence_fast: float = 85.0


@dataclass(frozen=True)
class ArchAConfig:
    precision: str = "fp32"


@dataclass(frozen=True)
class ArchBConfig:
    ocr_workers: int = 3
    batch_size: int = 8
    use_cache: bool = True
    routing: str = "auto"
    fast_precision: str = "4bit"
    full_precision: str = "fp32"
    thresholds: RoutingThresholds = field(default_factory=RoutingThresholds)

    def namespace(self) -> str:
        return f"{LLM_NAME}|{self.fast_precision}|{self.full_precision}|{self.routing}|{self.thresholds}|{EXTRACTION_VERSION}"
