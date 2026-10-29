import re

from pipeline.normalize import is_null, parse_amount, parse_date

REVIEW_MIN_CONFIDENCE = 80.0

TOTAL_KEYWORD = re.compile(r"\b(total|amount due|balance due|grand total|gross)\b", re.IGNORECASE)
MONEY = re.compile(r"[$€£]?\s?\d{1,3}(?:[ .,]\d{3})*(?:[.,]\d{2})?(?!\d)|[$€£]?\s?\d+(?:[.,]\d{2})?(?!\d)")
TOLERANCE = 0.011


def _amounts(line: str) -> list[tuple[float, str]]:
    found = []
    for cell in line.split("|"):
        for match in MONEY.finditer(cell):
            raw = match.group(0).strip()
            if "%" in cell[match.end() : match.end() + 1]:
                continue
            value = parse_amount(raw)
            if value is not None and value > 0:
                found.append((value, raw))
    return found


def total_line_amounts(text: str) -> list[tuple[float, str]]:
    amounts = []
    for line in text.splitlines():
        if TOTAL_KEYWORD.search(line):
            amounts.extend(_amounts(line))
    return amounts


def review_reasons(prediction: dict, text: str, confidence: float) -> list[str]:
    reasons = []
    for field in ("invoice_number", "date", "total"):
        if is_null(prediction.get(field)):
            reasons.append(f"missing_{field}")
    date = prediction.get("date")
    if not is_null(date) and parse_date(date) is None:
        reasons.append("date_unreadable")
    total_text = prediction.get("total")
    total = parse_amount(total_text)
    if not is_null(total_text) and total is None:
        reasons.append("total_unreadable")
    amounts = total_line_amounts(text)
    if total is not None and amounts and not any(abs(value - total) < TOLERANCE for value, _ in amounts):
        reasons.append("total_not_on_total_line")
    if confidence < REVIEW_MIN_CONFIDENCE:
        reasons.append("low_ocr_confidence")
    return reasons


def finalize(prediction: dict, text: str, reconcile: bool) -> tuple[dict, str]:
    if not reconcile or "total" not in prediction:
        return prediction, "model"
    total, source = reconcile_total(text, prediction["total"])
    return dict(prediction, total=total), source


def reconcile_total(text: str, predicted: str) -> tuple[str, str]:
    amounts = total_line_amounts(text)
    best = None
    for i, (a, _) in enumerate(amounts):
        for j, (b, _) in enumerate(amounts):
            if i == j or b > a:
                continue
            for value, raw in amounts:
                if abs(a + b - value) < TOLERANCE and value > a:
                    if best is None or value > best[0]:
                        best = (value, raw)
    if best is not None:
        if parse_amount(predicted) is not None and abs(parse_amount(predicted) - best[0]) < TOLERANCE:
            return predicted, "model"
        return best[1], "reconciled"
    return predicted, "model"
