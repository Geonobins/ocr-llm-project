import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import FIELDS, RESULTS_DIR

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
COLORS = {"A": "#2a78d6", "B": "#eb6834", "ocr": "#2a78d6", "embed": "#1baf7a", "llm": "#eda100", "real": "#2a78d6", "synthetic": "#eb6834", "FAST": "#1baf7a", "FULL": "#4a3aa7", "CACHE": "#eda100"}

METRIC_COLUMNS = [
    ("wall_time_s", "Wall time (s)"),
    ("throughput_docs_per_min", "Throughput (docs/min)"),
    ("mean_service_time_s", "Mean service time (s/doc)"),
    ("p95_latency_s", "p95 latency (s)"),
    ("mean_ocr_time_s", "Mean OCR (s)"),
    ("mean_embed_time_s", "Mean embedding (s)"),
    ("mean_llm_time_s", "Mean LLM (s)"),
    ("peak_rss_mb", "Peak RAM (MB)"),
    ("torch_peak_allocated_mb", "Pipeline GPU peak (MB)"),
    ("peak_gpu_mb", "Whole-GPU peak (MB)"),
    ("cost_per_100_docs_eur", "Cost per 100 docs (EUR)"),
    ("f1_all", "F1 (all)"),
    ("f1_real", "F1 (real)"),
    ("f1_synthetic", "F1 (synthetic)"),
    ("document_accuracy", "Documents fully correct"),
    ("f1_all_relaxed", "F1 (all), relaxed"),
    ("document_accuracy_relaxed", "Fully correct, relaxed"),
]


def load_runs(results_dir: Path) -> pd.DataFrame:
    rows = []
    for summary_path in sorted((results_dir / "runs").glob("*/run_*/summary.json")):
        with open(summary_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        row = {"config": payload["config_name"], "architecture": payload["architecture"], "run": payload["run"], "description": payload["description"]}
        row.update(payload["metrics"])
        scores = payload["scores"]
        row["f1_all"] = scores["all"]["overall"]["f1"]
        row["precision_all"] = scores["all"]["overall"]["precision"]
        row["recall_all"] = scores["all"]["overall"]["recall"]
        row["document_accuracy"] = scores["all"]["document_accuracy"]
        for source in ("real", "synthetic"):
            row[f"f1_{source}"] = scores.get(source, {}).get("overall", {}).get("f1", float("nan"))
        for field in FIELDS:
            row[f"f1_{field}"] = scores["all"][field]["f1"]
            row[f"precision_{field}"] = scores["all"][field]["precision"]
            row[f"recall_{field}"] = scores["all"][field]["recall"]
        relaxed = payload.get("scores_relaxed")
        if relaxed:
            row["f1_all_relaxed"] = relaxed["all"]["overall"]["f1"]
            row["document_accuracy_relaxed"] = relaxed["all"]["document_accuracy"]
        review = payload.get("review", {}).get("all")
        if review:
            row["straight_through_rate"] = review["straight_through_rate"]
            row["accuracy_auto_accepted"] = review["accuracy_auto_accepted"]
            row["accuracy_without_review"] = review["accuracy_without_review"]
            row["wrong_caught_share"] = review["wrong_caught_by_flag"] / review["wrong_documents"] if review["wrong_documents"] else 0.0
        row["hardware"] = json.dumps(payload["hardware"])
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No run summaries found under {results_dir / 'runs'}")
    return pd.DataFrame(rows)


def aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    numeric = runs.select_dtypes("number").columns.drop("run")
    grouped = runs.groupby("config")[numeric].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    grouped["runs"] = runs.groupby("config")["run"].count()
    grouped["architecture"] = runs.groupby("config")["architecture"].first()
    grouped["description"] = runs.groupby("config")["description"].first()
    order = [name for name in CONFIG_ORDER if name in grouped.index] + [name for name in grouped.index if name not in CONFIG_ORDER]
    return grouped.loc[order]


CONFIG_ORDER = ["A", "B", "B-warm", "B-nocache", "B-allfull", "B-allfast", "B-fp16", "B-8bit", "B-1worker", "B-6workers", "B-nobatch", "B-tuned"]


def fmt(mean, std, digits=2) -> str:
    if pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def markdown_tables(summary: pd.DataFrame, runs: pd.DataFrame) -> str:
    lines = ["# Experiment summary", ""]
    hardware = json.loads(runs.iloc[0]["hardware"])
    lines.append("Hardware: " + ", ".join(f"{k}={v}" for k, v in hardware.items()))
    lines.append("")
    lines.append("Values are mean ± standard deviation over the runs of each configuration.")
    lines.append("")
    lines.append("## Configurations")
    lines.append("")
    lines.append("| Configuration | Runs | Description |")
    lines.append("|---|---|---|")
    for name, row in summary.iterrows():
        lines.append(f"| {name} | {int(row['runs'])} | {row['description']} |")
    lines.append("")
    lines.append("## Performance")
    lines.append("")
    perf = ["wall_time_s", "throughput_docs_per_min", "mean_service_time_s", "p95_latency_s", "mean_ocr_time_s", "mean_embed_time_s", "mean_llm_time_s"]
    labels = dict(METRIC_COLUMNS)
    lines.append("| Configuration | " + " | ".join(labels[m] for m in perf) + " |")
    lines.append("|---|" + "---|" * len(perf))
    for name, row in summary.iterrows():
        cells = [fmt(row[f"{m}_mean"], row[f"{m}_std"], 3 if "time" in m and "wall" not in m else 2) for m in perf]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Resources and cost")
    lines.append("")
    res = ["peak_rss_mb", "torch_peak_allocated_mb", "peak_gpu_mb", "cost_per_100_docs_eur"]
    lines.append("| Configuration | " + " | ".join(labels[m] for m in res) + " |")
    lines.append("|---|" + "---|" * len(res))
    for name, row in summary.iterrows():
        cells = [fmt(row[f"{m}_mean"], row[f"{m}_std"], 4 if "cost" in m else 0) for m in res]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Accuracy")
    lines.append("")
    acc = ["f1_all", "f1_real", "f1_synthetic", "document_accuracy"]
    if "f1_all_relaxed_mean" in summary.columns:
        acc += ["f1_all_relaxed", "document_accuracy_relaxed"]
        lines.append(
            "Strict matching (headline): invoice numbers equal after removing labels and punctuation, the same calendar date, "
            "the same amount to the cent. Relaxed matching adds invoice numbers with a similarity ratio of at least 90 and "
            "day/month swaps of ambiguous numeric dates."
        )
        lines.append("")
    lines.append("| Configuration | " + " | ".join(labels[m] for m in acc) + " |")
    lines.append("|---|" + "---|" * len(acc))
    for name, row in summary.iterrows():
        cells = [fmt(row[f"{m}_mean"], row[f"{m}_std"], 4) for m in acc]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Per-field F1 (all documents)")
    lines.append("")
    lines.append("| Configuration | " + " | ".join(FIELDS) + " |")
    lines.append("|---|" + "---|" * len(FIELDS))
    for name, row in summary.iterrows():
        cells = [fmt(row[f"f1_{f}_mean"], row[f"f1_{f}_std"], 4) for f in FIELDS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    if "straight_through_rate_mean" in summary.columns:
        lines.append("## Human review")
        lines.append("")
        lines.append("An invoice is flagged for review when a field is missing or unreadable, the total does not appear on a total line, or OCR confidence is below 80.")
        lines.append("")
        lines.append("| Configuration | Straight-through (no review) | Fully correct, auto-accepted | Fully correct, no review step | Wrong invoices caught by the flag |")
        lines.append("|---|---|---|---|---|")
        for name, row in summary.iterrows():
            if pd.isna(row.get("straight_through_rate_mean")):
                continue
            lines.append(
                f"| {name} | {row['straight_through_rate_mean']:.1%} | {row['accuracy_auto_accepted_mean']:.1%} | "
                f"{row['accuracy_without_review_mean']:.1%} | {row['wrong_caught_share_mean']:.1%} |"
            )
        lines.append("")
    if "route_fast_mean" in summary.columns:
        lines.append("## Routing and cache")
        lines.append("")
        lines.append("| Configuration | FAST docs | FULL docs | Cache hits | Cache misses |")
        lines.append("|---|---|---|---|---|")
        for name, row in summary.iterrows():
            lines.append(
                f"| {name} | {row.get('route_fast_mean', 0):.0f} | {row.get('route_full_mean', 0):.0f} | "
                f"{row.get('cache_hits_mean', 0) if not pd.isna(row.get('cache_hits_mean', float('nan'))) else 0:.0f} | "
                f"{row.get('cache_misses_mean', 0) if not pd.isna(row.get('cache_misses_mean', float('nan'))) else 0:.0f} |"
            )
        lines.append("")
    return "\n".join(lines)


def style_axis(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=TEXT, pad=10)
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=9)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def new_figure(width=7.5, height=4.2):
    figure, ax = plt.subplots(figsize=(width, height), dpi=150)
    figure.patch.set_facecolor(SURFACE)
    return figure, ax


def bar_chart(summary: pd.DataFrame, metric: str, title: str, xlabel: str, path: Path, digits: int = 1, exclude=("B-warm",)) -> None:
    summary = summary[~summary.index.isin(exclude)]
    names = list(summary.index)
    means = summary[f"{metric}_mean"].values
    stds = summary[f"{metric}_std"].fillna(0).values
    colors = [COLORS[summary.loc[name, "architecture"]] for name in names]
    figure, ax = new_figure(height=0.45 * len(names) + 1.4)
    positions = range(len(names))[::-1]
    ax.barh(list(positions), means, xerr=stds, color=colors, height=0.55, error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 1, "capsize": 2})
    ax.set_yticks(list(positions))
    ax.set_yticklabels(names)
    for position, mean in zip(positions, means):
        ax.text(mean, position, f"  {mean:.{digits}f}", va="center", ha="left", fontsize=8.5, color=TEXT)
    ax.set_xlim(0, max(means + stds) * 1.18 if len(means) else 1)
    style_axis(ax, title, xlabel)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def stage_breakdown(summary: pd.DataFrame, path: Path) -> None:
    subset = summary[summary.index.isin(["A", "B", "B-allfull", "B-allfast", "B-fp16", "B-8bit", "B-1worker", "B-nobatch"])]
    names = list(subset.index)
    figure, ax = new_figure(height=0.5 * len(names) + 1.4)
    positions = list(range(len(names)))[::-1]
    left = [0.0] * len(names)
    for stage, label in (("ocr", "OCR"), ("embed", "Embedding"), ("llm", "LLM")):
        values = subset[f"mean_{stage}_time_s_mean"].values
        ax.barh(positions, values, left=left, color=COLORS[stage], height=0.55, label=label, edgecolor=SURFACE, linewidth=1.5)
        left = [a + b for a, b in zip(left, values)]
    for position, total in zip(positions, left):
        ax.text(total, position, f"  {total:.2f}s", va="center", ha="left", fontsize=8.5, color=TEXT)
    ax.set_yticks(positions)
    ax.set_yticklabels(names)
    ax.set_xlim(0, max(left) * 1.18 if left else 1)
    style_axis(ax, "Per-document service time by stage (uncached documents)", "seconds per document")
    handles, labels = ax.get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, fontsize=8.5, loc="lower center", ncol=3)
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    figure.savefig(path)
    plt.close(figure)


def f1_by_source(summary: pd.DataFrame, path: Path) -> None:
    names = list(summary.index)
    figure, ax = new_figure(height=0.5 * len(names) + 1.4)
    positions = list(range(len(names)))[::-1]
    offset = 0.18
    for shift, source in ((offset, "real"), (-offset, "synthetic")):
        values = summary[f"f1_{source}_mean"].values
        ax.barh([p + shift for p in positions], values, color=COLORS[source], height=0.32, label=source)
        for position, value in zip(positions, values):
            if not pd.isna(value):
                ax.text(value, position + shift, f"  {value:.3f}", va="center", ha="left", fontsize=8, color=TEXT)
    ax.set_yticks(positions)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 1.15)
    style_axis(ax, "F1 by document source", "F1")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def memory_panels(summary: pd.DataFrame, path: Path) -> None:
    names = list(summary.index)
    figure, axes = plt.subplots(1, 2, figsize=(10, 0.45 * len(names) + 1.4), dpi=150)
    figure.patch.set_facecolor(SURFACE)
    positions = list(range(len(names)))[::-1]
    for ax, metric, title in zip(axes, ("peak_rss_mb", "torch_peak_allocated_mb"), ("Peak RAM of process tree (MB)", "Peak GPU memory of the pipeline (MB)")):
        values = summary[f"{metric}_mean"].values
        colors = [COLORS[summary.loc[name, "architecture"]] for name in names]
        ax.barh(positions, values, color=colors, height=0.55)
        for position, value in zip(positions, values):
            ax.text(value, position, f"  {value:.0f}", va="center", ha="left", fontsize=8.5, color=TEXT)
        ax.set_yticks(positions)
        ax.set_yticklabels(names)
        ax.set_xlim(0, max(values) * 1.2 if len(values) else 1)
        style_axis(ax, title, "MB")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def routing_scatter(results_dir: Path, path: Path) -> None:
    run_dir = results_dir / "runs" / "B" / "run_1"
    if not (run_dir / "documents.csv").exists():
        return
    docs = pd.read_csv(run_dir / "documents.csv")
    docs = docs[docs["route"].isin(["FAST", "FULL"])]
    figure, ax = new_figure(height=4.6)
    for route in ("FAST", "FULL"):
        subset = docs[docs["route"] == route]
        ax.scatter(subset["feat_token_count"], subset["feat_confidence"], s=18, color=COLORS[route], label=f"{route} ({len(subset)})", alpha=0.8, edgecolors=SURFACE, linewidths=0.6)
    style_axis(ax, "Routing decision per document", "OCR text length (tokens)", "Mean OCR confidence")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8.5)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def precision_tradeoff(summary: pd.DataFrame, path: Path) -> None:
    mapping = {"B-allfull": "fp32", "B-fp16": "fp16", "B-8bit": "8-bit", "B-allfast": "4-bit (nf4)"}
    subset = summary[summary.index.isin(mapping)]
    if subset.empty:
        return
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=150)
    figure.patch.set_facecolor(SURFACE)
    names = [mapping[name] for name in subset.index]
    positions = list(range(len(names)))[::-1]
    for ax, metric, title, digits in zip(axes, ("mean_llm_time_s", "f1_all"), ("Mean LLM time per document (s)", "F1 (all documents)"), (3, 4)):
        values = subset[f"{metric}_mean"].values
        ax.barh(positions, values, color=COLORS["B"], height=0.55)
        for position, value in zip(positions, values):
            ax.text(value, position, f"  {value:.{digits}f}", va="center", ha="left", fontsize=8.5, color=TEXT)
        ax.set_yticks(positions)
        ax.set_yticklabels(names)
        ax.set_xlim(0, max(values) * 1.2 if len(values) else 1)
        style_axis(ax, title)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate experiment runs into tables and figures")
    parser.add_argument("--results", default=str(RESULTS_DIR))
    args = parser.parse_args()
    print(generate(Path(args.results)))


def generate(results_dir: Path) -> str:
    figures = results_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    runs = load_runs(results_dir)
    summary = aggregate(runs)
    runs.drop(columns=["hardware"]).to_csv(results_dir / "runs.csv", index=False)
    summary.to_csv(results_dir / "summary.csv")
    (results_dir / "summary.md").write_text(markdown_tables(summary, runs), encoding="utf-8")

    bar_chart(summary, "throughput_docs_per_min", "Throughput", "documents per minute", figures / "throughput.png")
    bar_chart(summary, "mean_service_time_s", "Mean service time per document", "seconds", figures / "service_time.png", 2)
    bar_chart(summary, "wall_time_s", "Wall time for the whole dataset", "seconds", figures / "wall_time.png", 0)
    bar_chart(summary, "cost_per_100_docs_eur", "Cost per 100 documents", "EUR", figures / "cost.png", 4, exclude=())
    bar_chart(summary, "f1_all", "F1 over all documents", "F1", figures / "f1.png", 3, exclude=())
    stage_breakdown(summary, figures / "stage_breakdown.png")
    f1_by_source(summary, figures / "f1_by_source.png")
    memory_panels(summary, figures / "memory.png")
    routing_scatter(results_dir, figures / "routing_scatter.png")
    precision_tradeoff(summary, figures / "precision_tradeoff.png")
    return (results_dir / "summary.md").read_text(encoding="utf-8")


if __name__ == "__main__":
    main()
