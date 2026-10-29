import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def go_offline() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def new_run_dir(label: str) -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-") or "run"
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    candidate = OUTPUTS_DIR / f"{stamp}_{clean}"
    suffix = 2
    while candidate.exists():
        candidate = OUTPUTS_DIR / f"{stamp}_{clean}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def mark_latest(run_dir: Path) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "LATEST.txt").write_text(str(run_dir) + "\n", encoding="utf-8")


def package_versions() -> dict:
    versions = {}
    for name in ("torch", "transformers", "bitsandbytes", "accelerate", "sentence_transformers", "pytesseract", "rapidfuzz"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = "not installed"
    try:
        import pytesseract

        from pipeline.config import TESSERACT_CMD

        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        versions["tesseract"] = str(pytesseract.get_tesseract_version())
    except Exception:
        versions["tesseract"] = "unknown"
    return versions


def pipeline_settings() -> dict:
    from pipeline import config

    return {
        "llm": config.LLM_NAME,
        "embedder": config.EMBEDDER_NAME,
        "ocr_layout": config.OCR_LAYOUT,
        "reconcile_totals": config.RECONCILE_TOTALS,
        "max_input_tokens": config.MAX_INPUT_TOKENS,
        "max_new_tokens_fast": config.MAX_NEW_TOKENS_FAST,
        "max_new_tokens_full": config.MAX_NEW_TOKENS_FULL,
        "default_routing_thresholds": vars(config.RoutingThresholds()),
        "gpu_hourly_cost_eur": config.GPU_HOURLY_COST_EUR,
    }


def write_run_info(run_dir: Path, info: dict) -> None:
    path = run_dir / "run_info.json"
    existing = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    existing.update(info)
    path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")


class Tee:
    def __init__(self, stream, handle):
        self.stream = stream
        self.handle = handle

    def write(self, data):
        self.stream.write(data)
        self.handle.write(data)
        return len(data)

    def flush(self):
        self.stream.flush()
        self.handle.flush()

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self.stream, name)


def capture_console(run_dir: Path):
    handle = open(run_dir / "log.txt", "a", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, handle)
    sys.stderr = Tee(sys.__stderr__, handle)
    return handle
