import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import FIELDS, RESULTS_DIR

TOKEN_GRID = list(range(50, 801, 25))
CONFIDENCE_GRID = [70.0, 75.0, 80.0, 82.5, 85.0, 87.5, 90.0, 92.5, 95.0]


def load_config_documents(results_dir: Path, name: str) -> pd.DataFrame:
    frames = []
    for run_dir in sorted((results_dir / "runs" / name).glob("run_*")):
        frame = pd.read_csv(run_dir / "documents.csv")
        frame["run"] = run_dir.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No runs found for configuration {name} under {results_dir}")
    data = pd.concat(frames, ignore_index=True)
    grouped = data.groupby("name").agg(
        source=("source", "first"),
        token_count=("feat_token_count", "mean"),
        confidence=("feat_confidence", "mean"),
        correct=("score_correct_fields", "mean"),
        llm=("llm_time", "mean"),
    )
    return grouped


def evaluate_policy(merged: pd.DataFrame, max_tokens: int, min_confidence: float) -> dict:
    fast = (merged["token_count"] <= max_tokens) & (merged["confidence"] >= min_confidence)
    correct = merged["correct_fast"].where(fast, merged["correct_full"])
    llm_time = merged["llm_fast"].where(fast, merged["llm_full"])
    return {
        "max_tokens_fast": max_tokens,
        "min_confidence_fast": min_confidence,
        "fast_share": round(fast.mean(), 4),
        "field_accuracy": round(correct.sum() / (len(merged) * len(FIELDS)), 4),
        "mean_llm_time_s": round(llm_time.mean(), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive routing thresholds from the all-fast / all-full runs")
    parser.add_argument("--results", default=str(RESULTS_DIR))
    parser.add_argument("--fast", default="B-allfast")
    parser.add_argument("--full", default="B-allfull")
    parser.add_argument("--tolerance", type=float, default=0.005)
    args = parser.parse_args()
    report = analyze(Path(args.results), args.fast, args.full, args.tolerance)
    print(report["markdown"])


def analyze(results_dir: Path, fast_name: str = "B-allfast", full_name: str = "B-allfull", tolerance: float = 0.005) -> dict:
    fast = load_config_documents(results_dir, fast_name)
    full = load_config_documents(results_dir, full_name)
    merged = fast.join(full, lsuffix="_fast", rsuffix="_full")
    merged = merged.rename(columns={
        "source_fast": "source",
        "token_count_fast": "token_count",
        "confidence_fast": "confidence",
    })

    rows = [evaluate_policy(merged, t, c) for t in TOKEN_GRID for c in CONFIDENCE_GRID]
    grid = pd.DataFrame(rows)
    all_full = evaluate_policy(merged, -1, 101.0)
    all_fast = evaluate_policy(merged, 10**9, 0.0)
    floor = all_full["field_accuracy"] - tolerance
    admissible = grid[grid["field_accuracy"] >= floor]
    chosen = admissible.sort_values(["fast_share", "field_accuracy"], ascending=[False, False]).iloc[0]

    per_route = merged.assign(
        fast_better=(merged["correct_fast"] > merged["correct_full"]),
        full_better=(merged["correct_full"] > merged["correct_fast"]),
    )
    bins = pd.cut(merged["token_count"], bins=[0, 100, 200, 300, 400, 600, 10**6])
    by_bin = merged.groupby(bins, observed=True).agg(
        documents=("correct_fast", "size"),
        acc_fast=("correct_fast", lambda s: round(s.mean() / len(FIELDS), 4)),
        acc_full=("correct_full", lambda s: round(s.mean() / len(FIELDS), 4)),
        llm_fast=("llm_fast", "mean"),
        llm_full=("llm_full", "mean"),
    )
    conf_bins = pd.cut(merged["confidence"], bins=[0, 80, 85, 90, 95, 100])
    by_conf = merged.groupby(conf_bins, observed=True).agg(
        documents=("correct_fast", "size"),
        acc_fast=("correct_fast", lambda s: round(s.mean() / len(FIELDS), 4)),
        acc_full=("correct_full", lambda s: round(s.mean() / len(FIELDS), 4)),
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(results_dir / "routing_grid.csv", index=False)
    merged.to_csv(results_dir / "routing_documents.csv")

    lines = [
        "# Routing threshold analysis",
        "",
        f"Source runs: `{fast_name}` (every document on the FAST 4-bit route) and `{full_name}` (every document on the FULL fp32 route).",
        f"Documents: {len(merged)}. Field accuracy is averaged over all runs of each configuration.",
        "",
        "## Reference policies",
        "",
        "| Policy | FAST share | Field accuracy | Mean LLM time (s) |",
        "|---|---|---|---|",
        f"| all FULL | {all_full['fast_share']:.2f} | {all_full['field_accuracy']:.4f} | {all_full['mean_llm_time_s']:.4f} |",
        f"| all FAST | {all_fast['fast_share']:.2f} | {all_fast['field_accuracy']:.4f} | {all_fast['mean_llm_time_s']:.4f} |",
        "",
        "## Accuracy by token count bin",
        "",
        by_bin.to_markdown(),
        "",
        "## Accuracy by OCR confidence bin",
        "",
        by_conf.to_markdown(),
        "",
        f"Documents where FAST beats FULL: {int(per_route['fast_better'].sum())}; FULL beats FAST: {int(per_route['full_better'].sum())}.",
        "",
        "## Chosen thresholds",
        "",
        f"Admissible policies keep field accuracy within {tolerance:.3f} of the all-FULL policy ({floor:.4f}).",
        f"Among them the policy with the largest FAST share is `max_tokens_fast={int(chosen['max_tokens_fast'])}`, "
        f"`min_confidence_fast={chosen['min_confidence_fast']}` with FAST share {chosen['fast_share']:.2f}, "
        f"field accuracy {chosen['field_accuracy']:.4f} and mean LLM time {chosen['mean_llm_time_s']:.4f}s.",
        "",
        "## Top 15 admissible policies",
        "",
        admissible.sort_values(["fast_share", "field_accuracy"], ascending=[False, False]).head(15).to_markdown(index=False),
        "",
    ]
    markdown = "\n".join(lines)
    (results_dir / "routing_analysis.md").write_text(markdown, encoding="utf-8")
    thresholds = {
        "max_tokens_fast": int(chosen["max_tokens_fast"]),
        "min_confidence_fast": float(chosen["min_confidence_fast"]),
        "fast_share": float(chosen["fast_share"]),
        "field_accuracy": float(chosen["field_accuracy"]),
        "all_full_field_accuracy": float(all_full["field_accuracy"]),
        "tolerance": tolerance,
    }
    (results_dir / "routing_thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    return {"thresholds": thresholds, "markdown": markdown}


if __name__ == "__main__":
    main()
