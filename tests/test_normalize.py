from datetime import date

import pytest

from decimal import Decimal

from pipeline.normalize import (
    clean_prediction,
    dates_equal,
    is_ambiguous_numeric_date,
    is_null,
    normalize_invoice_number,
    parse_amount,
    parse_complete_date,
    parse_date,
    parse_exact_amount,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("$6,270", 6270.0),
        ("$65.06", 65.06),
        ("188,11", 188.11),
        ("1574", 1574.0),
        ("$1 650.00", 1650.0),
        ("$1,300.00", 1300.0),
        ("$1600.00", 1600.0),
        ("€ 12.345,67", 12345.67),
        ("USD 45,130", 45130.0),
        ("Total: 2,874.38", 2874.38),
        ("null", None),
        ("", None),
        ("no amount here", None),
    ],
)
def test_parse_amount(raw, expected):
    assert parse_amount(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("$6,270", Decimal("6270")),
        ("$65.06", Decimal("65.06")),
        ("188,11", Decimal("188.11")),
        ("1574.0", Decimal("1574")),
        ("$ 4 958,16", Decimal("4958.16")),
        ("€ 12.345,67", Decimal("12345.67")),
        ("USD 45,130", Decimal("45130")),
        ("0.00", Decimal("0")),
        ("-123.45", Decimal("-123.45")),
        ("$-123.45", Decimal("-123.45")),
        ("(123.45)", Decimal("-123.45")),
        ("($ 1,234.50)", Decimal("-1234.50")),
        ("Total 123.45 tax 9.00", None),
        ("123.45 9.00", None),
        ("Total: 2,874.38", None),
        ("1,1696", None),
        ("12a", None),
        ("null", None),
        ("", None),
    ],
)
def test_parse_exact_amount(raw, expected):
    assert parse_exact_amount(raw) == expected


def test_parse_exact_amount_is_exact_not_float():
    assert parse_exact_amount("0.30") == parse_exact_amount("0.3")
    assert parse_exact_amount("1234.56") != parse_exact_amount("1234.57")


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("FEB 11 2020", date(2020, 2, 11)),
        ("FEB112020", date(2020, 2, 11)),
        ("FEB272021", date(2021, 2, 27)),
        ("Aug. 22, 2023", date(2023, 8, 22)),
        ("April 19, 2024", date(2024, 4, 19)),
        ("MARCH3,2020", date(2020, 3, 3)),
        ("18 APRIL, 2027", date(2027, 4, 18)),
        ("01/01/18", date(2018, 1, 1)),
        ("MM/DD/YYYY", None),
        ("null", None),
    ],
)
def test_parse_date(raw, expected):
    assert parse_date(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("April 19, 2024", date(2024, 4, 19)),
        ("April 19,2024", date(2024, 4, 19)),
        ("FEB112020", date(2020, 2, 11)),
        ("2024-04-05", date(2024, 4, 5)),
        ("13/04/2024", date(2024, 4, 13)),
        ("June 2022", None),
        ("21/2022", None),
        ("April 19", None),
        ("MM/DD/YYYY", None),
        ("null", None),
    ],
)
def test_parse_complete_date_needs_day_month_and_year(raw, expected):
    assert parse_complete_date(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("07/06/2025", True),
        ("11.02.2030", True),
        ("01-01-18", True),
        ("13/04/2024", False),
        ("2024-04-05", False),
        ("June 7, 2025", False),
        ("null", False),
    ],
)
def test_is_ambiguous_numeric_date(raw, expected):
    assert is_ambiguous_numeric_date(raw) is expected


def test_dates_equal_tolerates_day_month_swap():
    assert dates_equal(parse_date("07/06/2025"), parse_date("June 7, 2025"))
    assert dates_equal(parse_date("11.02.2030"), parse_date("11/02/2030"))
    assert not dates_equal(parse_date("April 19, 2024"), parse_date("May 2, 2024"))
    assert not dates_equal(None, parse_date("May 2, 2024"))


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("INV-8289-16", "INV828916"),
        ("#1234", "1234"),
        ("Invoice No: 01234", "01234"),
        ("invoice number INV-ZFR-0001", "INVZFR0001"),
        ("1009-01", "100901"),
        ("null", ""),
    ],
)
def test_normalize_invoice_number(raw, expected):
    assert normalize_invoice_number(raw) == expected


@pytest.mark.parametrize("value", [None, "", "null", "None", "N/A", "not found", "  "])
def test_is_null(value):
    assert is_null(value)


def test_clean_prediction_total_extracts_amount():
    assert clean_prediction("total", "The total is $1,902.05 due") == "1,902.05"
    assert clean_prediction("total", "unknown") == "null"
    assert clean_prediction("total", "") == "null"


def test_clean_prediction_invoice_strips_prefix():
    assert clean_prediction("invoice_number", "Invoice Number: INV-8289-16") == "INV-8289-16"
    assert clean_prediction("invoice_number", "  ") == "null"


def test_clean_prediction_date_collapses_whitespace():
    assert clean_prediction("date", "April   19,\n2024") == "April 19, 2024"


def test_date_with_comma_and_no_space_keeps_its_year():
    from datetime import date

    from pipeline.normalize import parse_date

    assert parse_date("April 19,2024") == date(2024, 4, 19)
    assert parse_date("Mar 22,2023") == date(2023, 3, 22)
    assert parse_date("April 19, 2024") == date(2024, 4, 19)
