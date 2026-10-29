import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.scorer import aggregate, score_document
from pipeline.config import FIELDS, GROUND_TRUTH_DIR, RESULTS_DIR


def rescore_run(run_dir: Path) -> None:
    summary_path = run_dir / "summary.json"
    aggregated = rescore_documents(run_dir / "documents.csv")
    with open(summary_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["scores"] = aggregated
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"{run_dir.parent.name}/{run_dir.name}: F1={aggregated['all']['overall']['f1']}")


def rescore_documents(documents_path: Path) -> dict:
    with open(documents_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    scores = []
    for row in rows:
        prediction = {field: row.get(f"pred_{field}", "") for field in FIELDS}
        truth_path = GROUND_TRUTH_DIR / f"{row['name']}.json"
        if truth_path.exists():
            with open(truth_path, "r", encoding="utf-8") as handle:
                truth = json.load(handle)
            for field in FIELDS:
                row[f"truth_{field}"] = truth.get(field, "")
        else:
            truth = {field: row.get(f"truth_{field}", "") for field in FIELDS}
        score = score_document(prediction, truth)
        scores.append((row["source"], score))
        for key, value in score.items():
            row[f"score_{key}"] = value
    with open(documents_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    aggregated = {"all": aggregate([score for _, score in scores])}
    for source in ("real", "synthetic"):
        subset = [score for src, score in scores if src == source]
        if subset:
            aggregated[source] = aggregate(subset)
    return aggregated


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute scores of every run with the current scorer")
    parser.add_argument("--results", default=str(RESULTS_DIR))
    args = parser.parse_args()
    for run_dir in sorted(Path(args.results).glob("runs/*/run_*")):
        if (run_dir / "documents.csv").exists():
            rescore_run(run_dir)


if __name__ == "__main__":
    main()
