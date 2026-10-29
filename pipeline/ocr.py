import os
import time
from dataclasses import dataclass, asdict

import pytesseract
from PIL import Image, ImageOps

from pipeline.config import TESSERACT_CMD


@dataclass
class OcrResult:
    text: str
    confidence: float
    word_count: int
    line_count: int
    duration: float

    def to_dict(self) -> dict:
        return asdict(self)


def configure_worker() -> None:
    os.environ["OMP_THREAD_LIMIT"] = "1"
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def _load_image(image_path) -> Image.Image:
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    return image.convert("L")


def _assemble_text(data: dict) -> tuple[str, float, int, int]:
    lines: dict[tuple, list[str]] = {}
    confidences: list[float] = []
    for word, conf, block, par, line in zip(
        data["text"], data["conf"], data["block_num"], data["par_num"], data["line_num"]
    ):
        token = str(word).strip()
        if not token:
            continue
        key = (block, par, line)
        lines.setdefault(key, []).append(token)
        confidence = float(conf)
        if confidence >= 0:
            confidences.append(confidence)
    ordered = sorted(lines.items())
    output: list[str] = []
    previous_par = None
    for (block, par, _), words in ordered:
        if previous_par is not None and (block, par) != previous_par:
            output.append("")
        output.append(" ".join(words))
        previous_par = (block, par)
    text = "\n".join(output).strip()
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return text, mean_conf, len(confidences), len(lines)


def _segments(data: dict) -> list[dict]:
    segments: dict[tuple, dict] = {}
    for i, word in enumerate(data["text"]):
        token = str(word).strip()
        if not token:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        left, top = data["left"][i], data["top"][i]
        right, bottom = left + data["width"][i], top + data["height"][i]
        segment = segments.get(key)
        if segment is None:
            segments[key] = {"words": [token], "left": left, "top": top, "right": right, "bottom": bottom}
        else:
            segment["words"].append(token)
            segment["left"] = min(segment["left"], left)
            segment["top"] = min(segment["top"], top)
            segment["right"] = max(segment["right"], right)
            segment["bottom"] = max(segment["bottom"], bottom)
    return list(segments.values())


def _assemble_rows(data: dict) -> str:
    return _rows_from_segments(_segments(data))


def _rows_from_segments(segments: list[dict]) -> str:
    if not segments:
        return ""
    heights = sorted(s["bottom"] - s["top"] for s in segments)
    tolerance = 0.6 * heights[len(heights) // 2]
    segments.sort(key=lambda s: (s["top"] + s["bottom"]) / 2)
    rows: list[list[dict]] = []
    centers: list[float] = []
    for segment in segments:
        center = (segment["top"] + segment["bottom"]) / 2
        if rows and abs(center - centers[-1]) <= tolerance:
            rows[-1].append(segment)
            centers[-1] = sum((s["top"] + s["bottom"]) / 2 for s in rows[-1]) / len(rows[-1])
        else:
            rows.append([segment])
            centers.append(center)
    lines = []
    for row in rows:
        row.sort(key=lambda s: s["left"])
        lines.append(" | ".join(" ".join(s["words"]) for s in row))
    return "\n".join(lines)


def run_ocr_data(image_path) -> dict:
    if pytesseract.pytesseract.tesseract_cmd != TESSERACT_CMD:
        configure_worker()
    image = _load_image(image_path)
    return pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)


def build_result(data: dict, duration: float, layout: str = "blocks") -> OcrResult:
    text, confidence, words, lines = _assemble_text(data)
    if layout == "rows":
        text = _assemble_rows(data)
        lines = text.count("\n") + 1 if text else 0
    return OcrResult(text, confidence, words, lines, duration)


def run_ocr(image_path, layout: str = "blocks") -> OcrResult:
    start = time.perf_counter()
    data = run_ocr_data(image_path)
    return build_result(data, time.perf_counter() - start, layout)


class EasyOcrEngine:
    def __init__(self, gpu: bool = True):
        import easyocr

        self.reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    def run(self, image_path, layout: str = "rows") -> OcrResult:
        import numpy

        start = time.perf_counter()
        image = numpy.array(_load_image(image_path))
        segments: list[dict] = []
        confidences: list[float] = []
        for box, token, confidence in self.reader.readtext(image):
            token = str(token).strip()
            if not token:
                continue
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
            segments.append({"words": [token], "left": min(xs), "top": min(ys), "right": max(xs), "bottom": max(ys)})
            confidences.append(float(confidence) * 100)
        if layout == "rows":
            text = _rows_from_segments(segments)
        else:
            text = "\n".join(" ".join(segment["words"]) for segment in segments)
        word_count = sum(len(segment["words"][0].split()) for segment in segments)
        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
        lines = text.count("\n") + 1 if text else 0
        return OcrResult(text, mean_conf, word_count, lines, time.perf_counter() - start)
