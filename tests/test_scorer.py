import pytest

from evaluation.scorer import (
    RELAXED,
    RELAXED_PREFIX,
    STRICT,
    aggregate,
    aggregate_by_source,
    field_outcome,
    match_date,
    match_invoice_number,
    match_total,
    review_summary,
    score_both,
    score_document,
)

BOTH = (STRICT, RELAXED)


@pytest.mark.parametrize("mode", BOTH)
def test_match_total_is_numeric_not_textual(mode):
    assert match_total("6270", "$6,270", mode)
    assert match_total("$1650", "$1 650.00", mode)
    assert match_total("188.11", "188,11", mode)
    assert match_total("1574.0", "1574", mode)
    assert match_total("$ 4 958,16", "4958.16", mode)
    assert not match_total("6,999", "$6,270", mode)
    assert not match_total("110", "1574", mode)


@pytest.mark.parametrize("mode", BOTH)
def test_match_total_rejects_wrong_or_malformed_amounts(mode):
    assert not match_total("123.46", "123.45", mode)
    assert not match_total("-123.45", "123.45", mode)
    assert not match_total("(123.45)", "123.45", mode)
    assert match_total("(123.45)", "-123.45", mode)
    assert not match_total("Total 123.45 tax 9.00", "123.45", mode)
    assert not match_total("123.45 garbage", "123.45", mode)
    assert not match_total("1,1696", "$11696", mode)
    assert not match_total("0", "123.45", mode)


@pytest.mark.parametrize("mode", BOTH)
def test_match_total_compares_numbers_only(mode):
    assert match_total("USD 123.45", "EUR 123.45", mode)


@pytest.mark.parametrize("mode", BOTH)
def test_match_date_is_semantic(mode):
    assert match_date("2020-02-11", "FEB112020", mode)
    assert match_date("11/02/2030", "11.02.2030", mode)
    assert match_date("April 19,2024", "April 19, 2024", mode)
    assert match_date("MM/DD/YYYY", "MM/DD/YYYY", mode)
    assert not match_date("April 19, 2024", "May 2, 2024", mode)
    assert not match_date("22", "Aug. 22, 2023", mode)


@pytest.mark.parametrize("mode", BOTH)
def test_match_date_rejects_swapped_unambiguous_dates(mode):
    assert not match_date("2024-04-05", "2024-05-04", mode)
    assert not match_date("April 5, 2024", "May 4, 2024", mode)


@pytest.mark.parametrize("mode", BOTH)
def test_match_date_rejects_incomplete_dates(mode):
    assert not match_date("21/2022", "Jun 21, 2022", mode)
    assert not match_date("June 2022", "June 1, 2022", mode)
    assert not match_date("April 19", "April 19, 2024", mode)


def test_day_month_swap_only_in_relaxed_and_only_for_ambiguous_numeric_dates():
    assert not match_date("June 7, 2025", "07/06/2025", STRICT)
    assert match_date("June 7, 2025", "07/06/2025", RELAXED)
    assert match_date("07/06/2025", "06/07/2025", RELAXED)
    assert match_date("13/04/2024", "04/13/2024", STRICT)


@pytest.mark.parametrize("mode", BOTH)
def test_match_invoice_number_ignores_labels_and_punctuation(mode):
    assert match_invoice_number("1234", "#1234", mode)
    assert match_invoice_number("Invoice No: 01234", "01234", mode)
    assert match_invoice_number("INV-8289-16", "INV 8289 16", mode)
    assert match_invoice_number("inv-8289-16", "INV-8289-16", mode)


@pytest.mark.parametrize("mode", BOTH)
def test_match_invoice_number_rejects_changed_letters_or_extra_text(mode):
    assert not match_invoice_number("AB1234", "CD1234", mode)
    assert not match_invoice_number("009", "GALT-009", mode)
    assert not match_invoice_number("8289-16", "INV-8289-16", mode)
    assert not match_invoice_number("73627 Square", "73627", mode)
    assert not match_invoice_number("123 Anywhere St., Any City, 01234", "01234", mode)
    assert not match_invoice_number("INV-8289-17", "INV-8289-16", mode)


def test_fuzzy_invoice_number_only_in_relaxed():
    assert not match_invoice_number("1234567890", "1234567891", STRICT)
    assert match_invoice_number("1234567890", "1234567891", RELAXED)
    assert not match_invoice_number("898724443", "89872443", STRICT)
    assert match_invoice_number("898724443", "89872443", RELAXED)
    assert not match_invoice_number("1234567800", "1234567891", RELAXED)


@pytest.mark.parametrize("mode", BOTH)
@pytest.mark.parametrize(
    "pred, truth, outcome",
    [
        ("null", "null", "TN"),
        ("", None, "TN"),
        ("$5", "null", "FP"),
        ("$5", None, "FP"),
        ("null", "$5", "FN"),
        ("", "$5", "FN"),
        (None, "$5", "FN"),
        ("$5", "$5", "TP"),
        ("$4", "$5", "FP_FN"),
    ],
)
def test_field_outcome(pred, truth, outcome, mode):
    assert field_outcome("total", pred, truth, mode) == outcome


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


def test_score_both_keeps_strict_and_relaxed_apart():
    truth = {"invoice_number": "1234567891", "date": "07/06/2025", "total": "$100"}
    score = score_both({"invoice_number": "1234567890", "date": "June 7, 2025", "total": "100"}, truth)
    assert score["invoice_number"] == "FP_FN"
    assert score["date"] == "FP_FN"
    assert score["all_correct"] is False
    assert score[f"{RELAXED_PREFIX}invoice_number"] == "TP"
    assert score[f"{RELAXED_PREFIX}date"] == "TP"
    assert score[f"{RELAXED_PREFIX}all_correct"] is True
    assert aggregate([score])["documents_fully_correct"] == 0
    assert aggregate([score], RELAXED_PREFIX)["documents_fully_correct"] == 1


def test_aggregate_by_source_splits_real_and_synthetic():
    truth = {"invoice_number": "1", "date": "April 19, 2024", "total": "$1"}
    right = score_both({"invoice_number": "1", "date": "April 19, 2024", "total": "1"}, truth)
    wrong = score_both({"invoice_number": "2", "date": "April 19, 2024", "total": "1"}, truth)
    result = aggregate_by_source([right, wrong, right], ["real", "real", "synthetic"])
    assert result["all"]["documents_fully_correct"] == 2
    assert result["real"]["documents_fully_correct"] == 1
    assert result["synthetic"]["documents_fully_correct"] == 1


def test_review_summary_counts_flagged_and_caught_errors():
    entries = [
        {"source": "real", "needs_review": True, "all_correct": False},
        {"source": "real", "needs_review": True, "all_correct": True},
        {"source": "real", "needs_review": False, "all_correct": False},
        {"source": "synthetic", "needs_review": False, "all_correct": True},
    ]
    result = review_summary(entries)["all"]
    assert result["straight_through_rate"] == 0.5
    assert result["flagged"] == 2
    assert result["wrong_documents"] == 2
    assert result["wrong_caught_by_flag"] == 1
    assert result["flag_precision"] == 0.5
    assert result["accuracy_auto_accepted"] == 0.5
    assert result["accuracy_without_review"] == 0.5
