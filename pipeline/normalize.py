import re
from datetime import date

from dateutil import parser as date_parser

NULL_TOKENS = {"", "null", "none", "n/a", "na", "not found", "unknown", "-", "nan"}
INVOICE_PREFIX = re.compile(r"^(invoice|bill|receipt)?\s*(no|number|num|nr|id)?\s*[:.#]*\s*", re.IGNORECASE)
CURRENCY = re.compile(r"[€$£¥]|usd|eur|gbp|rs|inr", re.IGNORECASE)
LETTER_DIGIT_BOUNDARY = re.compile(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")
AMOUNT_TOKEN = re.compile(r"[€$£]?\s*\d[\d\s.,]*\d|\d")


def is_null(value) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in NULL_TOKENS


def clean_text(value) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_invoice_number(value) -> str:
    if is_null(value):
        return ""
    text = clean_text(value)
    text = INVOICE_PREFIX.sub("", text)
    return re.sub(r"[^A-Za-z0-9]", "", text).upper()


def digits_only(value) -> str:
    return re.sub(r"\D", "", str(value or ""))


def parse_date(value) -> date | None:
    if is_null(value):
        return None
    text = clean_text(value)
    text = re.sub(r",(?=\d)", ", ", text)
    text = LETTER_DIGIT_BOUNDARY.sub(" ", text)
    text = re.sub(r"\b(\d{1,2})(\d{4})\b", r"\1 \2", text)
    text = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", text, flags=re.IGNORECASE)
    for day_first in (False, True):
        try:
            parsed = date_parser.parse(text, dayfirst=day_first, fuzzy=True)
            return parsed.date()
        except (ValueError, OverflowError, TypeError):
            continue
    return None


def dates_equal(a: date | None, b: date | None) -> bool:
    if a is None or b is None:
        return False
    if a == b:
        return True
    if a.year == b.year and a.month == b.day and a.day == b.month:
        return True
    return False


def parse_amount(value) -> float | None:
    if is_null(value):
        return None
    text = CURRENCY.sub("", str(value))
    match = AMOUNT_TOKEN.search(text)
    if not match:
        return None
    token = re.sub(r"\s+", "", match.group(0))
    token = re.sub(r"[^\d.,-]", "", token)
    if not token:
        return None
    if "," in token and "." in token:
        if token.rfind(",") > token.rfind("."):
            token = token.replace(".", "").replace(",", ".")
        else:
            token = token.replace(",", "")
    elif "," in token:
        if re.fullmatch(r"-?\d{1,3}(,\d{3})+", token):
            token = token.replace(",", "")
        elif re.search(r",\d{1,2}$", token):
            token = token.replace(",", ".")
        else:
            token = token.replace(",", "")
    elif "." in token:
        if re.fullmatch(r"-?\d{1,3}(\.\d{3})+", token):
            token = token.replace(".", "")
    try:
        return round(float(token), 2)
    except ValueError:
        return None


def clean_prediction(field: str, answer: str) -> str:
    text = clean_text(answer)
    if is_null(text):
        return "null"
    if field == "total":
        match = AMOUNT_TOKEN.search(CURRENCY.sub("", text))
        return match.group(0).strip() if match else "null"
    if field == "invoice_number":
        stripped = INVOICE_PREFIX.sub("", text).strip(" :#.-")
        return stripped or "null"
    return text
