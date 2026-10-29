from evaluation.ocr_comparison import field_found
from evaluation.vlm_baseline import parse_answer
from pipeline.ocr import _rows_from_segments


def test_rows_from_detection_boxes_join_label_and_value():
    segments = [
        {"words": ["Total"], "left": 10, "top": 500, "right": 60, "bottom": 520},
        {"words": ["Invoice"], "left": 10, "top": 100, "right": 80, "bottom": 120},
        {"words": ["$327,08"], "left": 600, "top": 498, "right": 680, "bottom": 518},
    ]
    assert _rows_from_segments(segments).splitlines() == ["Invoice", "Total | $327,08"]


def test_field_found_uses_canonical_forms():
    text = "Invoice No: INV-8289-16\nDate | April 19, 2024\nTotal | $ 45,130.00"
    assert field_found("invoice_number", "INV-8289-16", text) is True
    assert field_found("date", "April 19, 2024", text) is True
    assert field_found("total", "$45,130", text) is True
    assert field_found("total", "$45,131", text) is False
    assert field_found("total", None, text) is None


def test_parse_answer_reads_json_and_falls_back():
    values, valid = parse_answer('```json\n{"invoice_number": "A-1", "date": "1 May 2024", "total": null}\n```')
    assert valid and values == {"invoice_number": "A-1", "date": "1 May 2024", "total": None}
    values, valid = parse_answer('invoice_number: "A-1", date: "1 May 2024", total: "$5"')
    assert not valid and values["invoice_number"] == "A-1" and values["total"] == "$5"


def test_field_found_ignores_box_separators_in_dates():
    assert field_found("date", "June 11, 2024", "Date: June 11 | 2024") is True
    assert field_found("date", "June 11, 2024", "Date: June 12 | 2024") is False
