from pipeline.ocr import _assemble_rows
from pipeline.validate import finalize, reconcile_total, review_reasons


def words(*entries):
    data = {key: [] for key in ("text", "block_num", "par_num", "line_num", "left", "top", "width", "height")}
    for text, block, line, left, top in entries:
        data["text"].append(text)
        data["block_num"].append(block)
        data["par_num"].append(1)
        data["line_num"].append(line)
        data["left"].append(left)
        data["top"].append(top)
        data["width"].append(40)
        data["height"].append(20)
    return data


def test_rows_join_columns_that_tesseract_splits_into_blocks():
    data = words(
        ("Total", 1, 1, 10, 500),
        ("$", 2, 1, 300, 502),
        ("297,35", 2, 1, 320, 502),
        ("$", 3, 1, 600, 498),
        ("327,08", 3, 1, 620, 498),
        ("Invoice", 4, 1, 10, 100),
    )
    assert _assemble_rows(data).splitlines() == ["Invoice", "Total | $ 297,35 | $ 327,08"]


def test_reconcile_prefers_amount_that_equals_net_plus_tax():
    text = "SUMMARY\nTotal | $ 4507,42 | $ 450,74 | $ 4 958,16"
    assert reconcile_total(text, "4507,42") == ("$ 4 958,16", "reconciled")


def test_reconcile_keeps_model_answer_when_it_already_reconciles():
    text = "Total | $ 297,35 $ 29,73 | $ 327,08"
    assert reconcile_total(text, "327,08") == ("327,08", "model")


def test_reconcile_keeps_model_answer_without_arithmetic_evidence():
    text = "Subtotal $4711\nTotal $4711"
    assert reconcile_total(text, "4711") == ("4711", "model")
    assert reconcile_total("Invoice 123\nThank you", "50.00") == ("50.00", "model")


def test_reconcile_ignores_percentages():
    text = "Total | 10% | $ 100,00 | $ 10,00 | $ 110,00"
    assert reconcile_total(text, "100,00") == ("$ 110,00", "reconciled")


def test_finalize_respects_switch():
    prediction = {"invoice_number": "1", "date": "x", "total": "4507,42"}
    text = "Total | $ 4507,42 | $ 450,74 | $ 4 958,16"
    assert finalize(prediction, text, False) == (prediction, "model")
    assert finalize(prediction, text, True)[0]["total"] == "$ 4 958,16"


def test_review_accepts_consistent_invoice():
    prediction = {"invoice_number": "INV-1", "date": "April 19, 2024", "total": "$ 327,08"}
    text = "Invoice INV-1\nTotal | $ 297,35 | $ 29,73 | $ 327,08"
    assert review_reasons(prediction, text, 92.0) == []


def test_review_flags_total_missing_from_total_line():
    prediction = {"invoice_number": "15001300", "date": "02/18/2014", "total": "15001300"}
    text = "Invoice no: 15001300\nTotal | $ 297,35 | $ 29,73 | $ 327,08"
    assert review_reasons(prediction, text, 92.0) == ["total_not_on_total_line"]


def test_review_flags_missing_unreadable_and_low_confidence():
    prediction = {"invoice_number": "null", "date": "MM/DD/YYYY", "total": "abc"}
    reasons = review_reasons(prediction, "no totals here", 60.0)
    assert reasons == ["missing_invoice_number", "date_unreadable", "total_unreadable", "low_ocr_confidence"]
