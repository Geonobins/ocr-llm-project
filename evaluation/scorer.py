from rapidfuzz import fuzz

from pipeline.config import FIELDS
from pipeline.normalize import (
    clean_text,
    dates_equal,
    digits_only,
    is_null,
    normalize_invoice_number,
    parse_amount,
    parse_date,
)

FUZZY_THRESHOLD = 90.0


def fuzzy_equal(a: str, b: str) -> bool:
    if not a or not b:
        return False
    return fuzz.ratio(a, b) >= FUZZY_THRESHOLD


def match_invoice_number(pred, truth) -> bool:
    a = normalize_invoice_number(pred)
    b = normalize_invoice_number(truth)
    if a == b:
        return True
    da, db = digits_only(a), digits_only(b)
    if da and db and da == db and len(db) >= 3:
        return True
    return fuzzy_equal(a, b)


def match_date(pred, truth) -> bool:
    da, db = parse_date(pred), parse_date(truth)
    if da is not None and db is not None:
        return dates_equal(da, db)
    a = clean_text(pred).lower()
    b = clean_text(truth).lower()
    if a == b:
        return True
    return fuzzy_equal(digits_only(a) or a, digits_only(b) or b)


def match_total(pred, truth) -> bool:
    fa, fb = parse_amount(pred), parse_amount(truth)
    if fa is not None and fb is not None:
        return abs(fa - fb) < 0.005
    return fuzzy_equal(digits_only(pred), digits_only(truth))


MATCHERS = {"invoice_number": match_invoice_number, "date": match_date, "total": match_total}


def field_outcome(field: str, pred, truth) -> str:
    pred_null = is_null(pred)
    truth_null = is_null(truth)
    if truth_null and pred_null:
        return "TN"
    if truth_null and not pred_null:
        return "FP"
    if pred_null:
        return "FN"
    return "TP" if MATCHERS[field](pred, truth) else "FP_FN"


def score_document(prediction: dict, truth: dict) -> dict:
    outcomes = {}
    for field in FIELDS:
        outcomes[field] = field_outcome(field, prediction.get(field), truth.get(field))
    outcomes["correct_fields"] = sum(1 for field in FIELDS if outcomes[field] in ("TP", "TN"))
    outcomes["all_correct"] = outcomes["correct_fields"] == len(FIELDS)
    return outcomes


def _counts(outcomes: list[str]) -> dict:
    counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for outcome in outcomes:
        if outcome == "FP_FN":
            counts["FP"] += 1
            counts["FN"] += 1
        else:
            counts[outcome] += 1
    return counts


def _prf(counts: dict) -> dict:
    tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / (tp + tn + max(fp, fn)) if tp + tn + max(fp, fn) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def aggregate(document_scores: list[dict]) -> dict:
    result = {}
    all_outcomes = []
    for field in FIELDS:
        outcomes = [score[field] for score in document_scores]
        all_outcomes.extend(outcomes)
        result[field] = _prf(_counts(outcomes))
    result["overall"] = _prf(_counts(all_outcomes))
    result["documents"] = len(document_scores)
    result["documents_fully_correct"] = sum(1 for score in document_scores if score["all_correct"])
    result["document_accuracy"] = (
        round(result["documents_fully_correct"] / len(document_scores), 4) if document_scores else 0.0
    )
    return result
