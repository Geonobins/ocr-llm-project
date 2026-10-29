import pytest

from evaluation.scorer import (
    aggregate,
    field_outcome,
    match_date,
    match_invoice_number,
    match_total,
    score_document,
)


def test_match_total_is_numeric_not_textual():
    assert match_total("6270", "$6,270")
    assert match_total("$1650", "$1 650.00")
    assert match_total("188.11", "188,11")
    assert match_total("1574.0", "1574")
    assert not match_total("6,999", "$6,270")
    assert not match_total("110", "1574")


def test_match_date_is_semantic():
    assert match_date("2020-02-11", "FEB112020")
    assert match_date("11/02/2030", "11.02.2030")
    assert match_date("June 7, 2025", "07/06/2025")
    assert match_date("MM/DD/YYYY", "MM/DD/YYYY")
    assert not match_date("April 19, 2024", "May 2, 2024")
    assert not match_date("22", "Aug. 22, 2023")


def test_match_invoice_number_tolerates_one_ocr_error_only():
    assert match_invoice_number("1234", "#1234")
    assert match_invoice_number("Invoice No: 01234", "01234")
    assert match_invoice_number("INV-8289-16", "INV-8289-16")
    assert match_invoice_number("8289-16", "INV-8289-16")
    assert not match_invoice_number("INV-8289-17", "INV-8289-16")
    assert not match_invoice_number("123 Anywhere St., Any City, 01234", "01234")


@pytest.mark.parametrize(
    "pred, truth, outcome",
    [
        ("null", "null", "TN"),
        ("$5", "null", "FP"),
        ("null", "$5", "FN"),
        ("$5", "$5", "TP"),
        ("$4", "$5", "FP_FN"),
    ],
)
def test_field_outcome(pred, truth, outcome):
    assert field_outcome("total", pred, truth) == outcome


def test_aggregate_precision_recall_f1():
    truth = {"invoice_number": "INV-1", "date": "April 19, 2024", "total": "$100"}
    docs = [
        score_document({"invoice_number": "INV-1", "date": "April 19, 2024", "total": "100"}, truth),
        score_document({"invoice_number": "INV-2", "date": "null", "total": "100"}, truth),
    ]
    result = aggregate(docs)
    overall = result["overall"]
    assert overall["tp"] == 4
    assert overall["fp"] == 1
    assert overall["fn"] == 2
    assert overall["precision"] == pytest.approx(4 / 5, abs=1e-4)
    assert overall["recall"] == pytest.approx(4 / 6, abs=1e-4)
    assert overall["f1"] == pytest.approx(2 * 0.8 * (4 / 6) / (0.8 + 4 / 6), abs=1e-4)
    assert result["documents_fully_correct"] == 1
    assert result["document_accuracy"] == 0.5
    assert result["date"]["recall"] == 0.5
