from rapidfuzz import fuzz

from pipeline.config import FIELDS
from pipeline.normalize import (
    clean_text,
    dates_equal,
    is_ambiguous_numeric_date,
    is_null,
    normalize_invoice_number,
    parse_complete_date,
    parse_exact_amount,
)

FUZZY_THRESHOLD = 90.0
STRICT = "strict"
RELAXED = "relaxed"
RELAXED_PREFIX = "relaxed_"


def fuzzy_equal(a: str, b: str) -> bool:
    if not a or not b:
        return False
    return fuzz.ratio(a, b) >= FUZZY_THRESHOLD


def match_invoice_number(pred, truth, mode: str = STRICT) -> bool:
    a = normalize_invoice_number(pred)
    b = normalize_invoice_number(truth)
    if a and a == b:
        return True
    return mode == RELAXED and fuzzy_equal(a, b)


def match_date(pred, truth, mode: str = STRICT) -> bool:
    da, db = parse_complete_date(pred), parse_complete_date(truth)
    if da is None or db is None:
        return da is None and db is None and clean_text(pred).lower() == clean_text(truth).lower()
    if da == db:
        return True
    ambiguous = is_ambiguous_numeric_date(pred) or is_ambiguous_numeric_date(truth)
    return mode == RELAXED and ambiguous and dates_equal(da, db)


def match_total(pred, truth, mode: str = STRICT) -> bool:
    a, b = parse_exact_amount(pred), parse_exact_amount(truth)
    return a is not None and a == b


MATCHERS = {"invoice_number": match_invoice_number, "date": match_date, "total": match_total}


def field_outcome(field: str, pred, truth, mode: str = STRICT) -> str:
    pred_null = is_null(pred)
    truth_null = is_null(truth)
    if truth_null and pred_null:
        return "TN"
    if truth_null and not pred_null:
        return "FP"
    if pred_null:
        return "FN"
    return "TP" if MATCHERS[field](pred, truth, mode) else "FP_FN"


def score_document(prediction: dict, truth: dict, mode: str = STRICT) -> dict:
    outcomes = {}
    for field in FIELDS:
        outcomes[field] = field_outcome(field, prediction.get(field), truth.get(field), mode)
    outcomes["correct_fields"] = sum(1 for field in FIELDS if outcomes[field] in ("TP", "TN"))
    outcomes["all_correct"] = outcomes["correct_fields"] == len(FIELDS)
    return outcomes


def score_both(prediction: dict, truth: dict) -> dict:
    outcomes = score_document(prediction, truth, STRICT)
    relaxed = score_document(prediction, truth, RELAXED)
    outcomes.update({f"{RELAXED_PREFIX}{key}": value for key, value in relaxed.items()})
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


def aggregate(document_scores: list[dict], prefix: str = "") -> dict:
    result = {}
    all_outcomes = []
    for field in FIELDS:
        outcomes = [score[f"{prefix}{field}"] for score in document_scores]
        all_outcomes.extend(outcomes)
        result[field] = _prf(_counts(outcomes))
    result["overall"] = _prf(_counts(all_outcomes))
    result["documents"] = len(document_scores)
    result["documents_fully_correct"] = sum(1 for score in document_scores if score[f"{prefix}all_correct"])
    result["document_accuracy"] = (
        round(result["documents_fully_correct"] / len(document_scores), 4) if document_scores else 0.0
    )
    return result


def aggregate_by_source(document_scores: list[dict], sources: list[str], prefix: str = "") -> dict:
    result = {"all": aggregate(document_scores, prefix)}
    for source in ("real", "synthetic"):
        subset = [score for score, src in zip(document_scores, sources) if src == source]
        if subset:
            result[source] = aggregate(subset, prefix)
    return result


def review_summary(entries: list[dict]) -> dict:
    result = {}
    for source in ("all", "real", "synthetic"):
        subset = [entry for entry in entries if source == "all" or entry["source"] == source]
        if not subset:
            continue
        flagged = [entry for entry in subset if entry["needs_review"]]
        accepted = [entry for entry in subset if not entry["needs_review"]]
        wrong = [entry for entry in subset if not entry["all_correct"]]
        result[source] = {
            "documents": len(subset),
            "straight_through_rate": round(len(accepted) / len(subset), 4),
            "flagged": len(flagged),
            "accuracy_auto_accepted": round(sum(e["all_correct"] for e in accepted) / len(accepted), 4) if accepted else 0.0,
            "accuracy_without_review": round(sum(e["all_correct"] for e in subset) / len(subset), 4),
            "wrong_documents": len(wrong),
            "wrong_caught_by_flag": sum(1 for e in flagged if not e["all_correct"]),
            "flag_precision": round(sum(1 for e in flagged if not e["all_correct"]) / len(flagged), 4) if flagged else 0.0,
        }
    return result
