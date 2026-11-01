import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.scorer import RELAXED_PREFIX, aggregate_by_source, review_summary, score_both
from pipeline.config import FIELDS, GROUND_TRUTH_DIR, RESULTS_DIR

CHANGE_COLUMNS = ["file", "invoice", "field", "prediction", "truth", "before", "strict", "relaxed"]


def keep_lenient(payload: dict, *keys: str) -> None:
    for key in keys:
        if key in payload and f"{key}_lenient_scorer" not in payload:
            payload[f"{key}_lenient_scorer"] = payload[key]


def rescore_run(run_dir: Path, changes: list) -> None:
    summary_path = run_dir / "summary.json"
    result = rescore_documents(run_dir / "documents.csv", changes)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    keep_lenient(payload, "scores", "review")
    payload["scores"] = result["scores"]
    payload["scores_relaxed"] = result["scores_relaxed"]
    if result["review"]:
        payload["review"] = result["review"]
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"{run_dir.parent.name}/{run_dir.name}: F1 strict={result['scores']['all']['overall']['f1']} "
        f"relaxed={result['scores_relaxed']['all']['overall']['f1']}"
    )


def load_truth(row: dict) -> dict:
    truth_path = GROUND_TRUTH_DIR / f"{row['name']}.json"
    if truth_path.exists():
        with open(truth_path, "r", encoding="utf-8") as handle:
            truth = json.load(handle)
        for field in FIELDS:
            row[f"truth_{field}"] = truth.get(field, "")
        return truth
    return {field: row.get(f"truth_{field}", "") for field in FIELDS}


def rescore_documents(documents_path: Path, changes: list | None = None) -> dict:
    with open(documents_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames)
    scores, sources, entries = [], [], []
    for row in rows:
        prediction = {field: row.get(f"pred_{field}", "") for field in FIELDS}
        truth = load_truth(row)
        score = score_both(prediction, truth)
        for field in FIELDS:
            before = row.get(f"score_{field}", "")
            after = (score[field], score[f"{RELAXED_PREFIX}{field}"])
            if changes is not None and before and (before != after[0] or before != after[1]):
                changes.append({
                    "file": documents_path.as_posix(),
                    "invoice": row["name"],
                    "field": field,
                    "prediction": prediction[field],
                    "truth": truth.get(field, ""),
                    "before": before,
                    "strict": after[0],
                    "relaxed": after[1],
                })
        for key, value in score.items():
            column = f"score_{key}"
            row[column] = value
            if column not in fieldnames:
                fieldnames.append(column)
        scores.append(score)
        sources.append(row["source"])
        if "feat_needs_review" in row and not row.get("error"):
            entries.append({"source": row["source"], "needs_review": row["feat_needs_review"] == "True", "all_correct": score["all_correct"]})
    with open(documents_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "scores": aggregate_by_source(scores, sources),
        "scores_relaxed": aggregate_by_source(scores, sources, RELAXED_PREFIX),
        "review": review_summary(entries) if entries else None,
    }


def write_changes(out_dir: Path, changes: list) -> None:
    if not changes:
        return
    with open(out_dir / "rescore_changes.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CHANGE_COLUMNS)
        writer.writeheader()
        writer.writerows(changes)
    print(f"{len(changes)} changed field outcomes written to {out_dir / 'rescore_changes.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute strict and relaxed scores of every run with the current scorer")
    parser.add_argument("--results", default=str(RESULTS_DIR))
    args = parser.parse_args()
    results = Path(args.results)
    changes = []
    for run_dir in sorted(results.glob("runs/*/run_*")):
        if (run_dir / "documents.csv").exists():
            rescore_run(run_dir, changes)
    write_changes(results, changes)


if __name__ == "__main__":
    main()
