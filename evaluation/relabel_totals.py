import argparse
import csv
import json
import re
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import GROUND_TRUTH_DIR
from pipeline.dataset import load_documents
from pipeline.normalize import parse_amount
from pipeline.ocr import run_ocr

AMOUNT = re.compile(r"(?<![\d,])\$?\s?(?:\d{1,3}(?: \d{3})+|\d+),\d{2}\b")
TOLERANCE = 0.011


def amounts_in(text: str) -> list[tuple[float, str]]:
    found = []
    for match in AMOUNT.finditer(text):
        raw = match.group(0).strip()
        value = parse_amount(raw)
        if value is not None:
            found.append((value, raw))
    return found


def find_totals(text: str):
    if "gross" not in text.lower():
        return None
    tail = amounts_in(text)[-24:]
    values = [v for v, _ in tail]
    if not values:
        return None
    gross = values[-1]
    for i, net in enumerate(values):
        for vat in values[i + 1 :]:
            if abs(net + vat - gross) < TOLERANCE and net > vat:
                return net, vat, gross
    return None


def gross_label(text: str, gross: float) -> str:
    for value, raw in reversed(amounts_in(text)):
        if abs(value - gross) < TOLERANCE and raw.startswith("$"):
            return raw.replace("$", "$ ").replace("  ", " ")
    for value, raw in reversed(amounts_in(text)):
        if abs(value - gross) < TOLERANCE:
            return f"$ {raw}"
    return f"$ {gross:,.2f}".replace(",", " ").replace(".", ",")


def main() -> None:
    parser = argparse.ArgumentParser(description="Relabel the total field of invoices 316-600 to the gross total")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--start", type=int, default=316)
    parser.add_argument("--end", type=int, default=600)
    args = parser.parse_args()

    documents = load_documents(indices=range(args.start, args.end + 1))
    rows = []
    for doc in documents:
        truth = doc.load_truth()
        old = truth.get("total", "")
        old_value = parse_amount(old)
        ocr = run_ocr(doc.image_path)
        totals = find_totals(ocr.text)
        row = {"index": doc.index, "old": old, "new": old, "net": "", "vat": "", "gross": "", "action": ""}
        if totals is None:
            row["action"] = "no summary block found, kept"
        else:
            net, vat, gross = totals
            row.update({"net": net, "vat": vat, "gross": gross})
            if old_value is not None and abs(old_value - gross) < TOLERANCE:
                row["action"] = "already gross, kept"
            elif old_value is not None and abs(old_value - net) < TOLERANCE:
                row["new"] = gross_label(ocr.text, gross)
                row["action"] = "net -> gross"
            else:
                row["action"] = "label matches neither net nor gross, kept"
        rows.append(row)
        print(f"{doc.index}: {row['action']}  old={old!r} new={row['new']!r}", flush=True)

    log_path = GROUND_TRUTH_DIR.parent / "results" / "relabel_totals_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    changed = [row for row in rows if row["action"] == "net -> gross"]
    summary = {}
    for row in rows:
        summary[row["action"]] = summary.get(row["action"], 0) + 1
    print("\nSummary:", summary)
    print("Log:", log_path)

    if not args.apply:
        print("Dry run; pass --apply to write the changes.")
        return

    backup = GROUND_TRUTH_DIR.parent / f"ground_truth_backup_{time.strftime('%Y%m%d')}"
    backup.mkdir(exist_ok=True)
    for row in changed:
        path = GROUND_TRUTH_DIR / f"test_invoice{row['index']}.json"
        shutil.copy2(path, backup / path.name)
        with open(path, "r", encoding="utf-8") as handle:
            truth = json.load(handle)
        truth["total"] = row["new"]
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(truth, handle, indent=2)
    print(f"Updated {len(changed)} files; originals copied to {backup}")


if __name__ == "__main__":
    main()
