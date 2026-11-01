import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RATES_PER_MIN = (30, 48, 120, 240)
VARIANTS = {
    "A": {"arch": "A", "wait": None, "description": "Architecture A: invoices processed one at a time in arrival order"},
    "B": {"arch": "B", "wait": math.inf, "description": "Architecture B as benchmarked: a batch goes to the GPU once 8 invoices are read"},
    "B-greedy": {"arch": "B", "wait": 0.0, "description": "Architecture B: when the GPU is free it takes every invoice already read, up to 8"},
}
VARIANT_COLORS = {"A": "#2a78d6", "B": "#eb6834", "B-greedy": "#1baf7a"}


def spread_indices(count: int, first: int = 1, last: int = 600) -> list[int]:
    if count <= 1:
        return [first]
    return sorted({round(first + i * (last - first) / (count - 1)) for i in range(count)})


def poisson_arrivals(count: int, rate_per_min: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    times, clock = [], 0.0
    for _ in range(count):
        times.append(clock)
        clock += rng.expovariate(rate_per_min / 60.0)
    return times


def percentile(values, q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * q / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(records, arrivals, rate: float) -> dict:
    ok = [record for record in records if not record.error]
    latencies = [record.latency for record in ok]
    finished = max(arrival + record.latency for arrival, record in zip(arrivals, records))
    return {
        "rate_per_min": rate,
        "documents": len(records),
        "errors": len(records) - len(ok),
        "mean_latency_s": round(statistics.fmean(latencies), 4),
        "p50_latency_s": round(percentile(latencies, 50), 4),
        "p95_latency_s": round(percentile(latencies, 95), 4),
        "max_latency_s": round(max(latencies), 4),
        "mean_service_time_s": round(statistics.fmean(record.service_time for record in ok), 4),
        "mean_ocr_time_s": round(statistics.fmean(record.ocr_time for record in ok), 4),
        "mean_llm_time_s": round(statistics.fmean(record.llm_time for record in ok), 4),
        "arrival_span_s": round(arrivals[-1], 2),
        "finished_after_s": round(finished, 2),
        "throughput_docs_per_min": round(len(records) / finished * 60, 2),
    }


def build(arch: str):
    from pipeline.arch_a import ArchitectureA
    from pipeline.arch_b import ArchitectureB
    from pipeline.config import ArchAConfig, ArchBConfig

    if arch == "A":
        return ArchitectureA(ArchAConfig(precision="fp32"))
    return ArchitectureB(ArchBConfig(use_cache=False))


def run_variant(architecture, variant: dict, documents, arrivals):
    if variant["arch"] == "A":
        return architecture.run_stream(documents, arrivals)
    return architecture.run_stream(documents, arrivals, variant["wait"])


def save(out_dir: Path, name: str, rate: float, run: int, seed: int, records, arrivals, summary: dict) -> dict:
    from evaluation.run_experiments import hardware_info, score_records

    scores, aggregated = score_records(records)
    run_dir = out_dir / "runs" / name / f"rate_{rate:g}" / f"run_{run}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for record, score, arrival in zip(records, scores, arrivals):
        row = {"arrival_s": round(arrival, 4), **record.to_row()}
        row.update({f"score_{key}": value for key, value in score.items()})
        rows.append(row)
    fieldnames = []
    for row in rows:
        fieldnames += [key for key in row if key not in fieldnames]
    with open(run_dir / "documents.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "variant": name,
        "description": VARIANTS[name]["description"],
        "rate_per_min": rate,
        "run": run,
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": hardware_info(),
        "metrics": summary,
        "f1_all": aggregated["all"]["overall"]["f1"],
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_runs(out_dir: Path):
    import pandas as pd

    rows = []
    for path in sorted((out_dir / "runs").glob("*/rate_*/run_*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"variant": payload["variant"], "run": payload["run"], **payload["metrics"], "f1_all": payload["f1_all"]})
    return pd.DataFrame(rows)


def report(out_dir: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from evaluation.report import GRID, SURFACE, TEXT, TEXT_SECONDARY

    runs = load_runs(out_dir)
    metrics = ["mean_latency_s", "p50_latency_s", "p95_latency_s", "max_latency_s", "mean_service_time_s", "mean_ocr_time_s",
               "throughput_docs_per_min", "f1_all"]
    grouped = runs.groupby(["variant", "rate_per_min"])[metrics].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    grouped["runs"] = runs.groupby(["variant", "rate_per_min"])["run"].count()
    grouped = grouped.reset_index()
    order = {name: position for position, name in enumerate(VARIANTS)}
    grouped = grouped.sort_values(by=["variant", "rate_per_min"], key=lambda column: column.map(order) if column.name == "variant" else column)
    grouped.to_csv(out_dir / "summary.csv", index=False)
    runs.to_csv(out_dir / "runs.csv", index=False)

    def cell(row, metric, digits):
        std = row[f"{metric}_std"]
        mean = row[f"{metric}_mean"]
        return f"{mean:.{digits}f}" if math.isnan(std) else f"{mean:.{digits}f} ± {std:.{digits}f}"

    lines = ["# Latency with invoices arriving one at a time", "",
             "Invoices arrive at random times (Poisson process) at the given average rate. Latency = time from an invoice's arrival to its result.",
             "Values are mean ± standard deviation over the runs.", "",
             "| Variant | Arrival rate (/min) | Runs | Median latency (s) | p95 latency (s) | Max latency (s) | Service time (s) | Mean OCR (s) | Throughput (/min) | F1 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, row in grouped.iterrows():
        lines.append(f"| {row['variant']} | {row['rate_per_min']:g} | {int(row['runs'])} | {cell(row, 'p50_latency_s', 2)} | {cell(row, 'p95_latency_s', 2)} | "
                     f"{cell(row, 'max_latency_s', 2)} | {cell(row, 'mean_service_time_s', 3)} | {cell(row, 'mean_ocr_time_s', 3)} | "
                     f"{cell(row, 'throughput_docs_per_min', 1)} | {cell(row, 'f1_all', 4)} |")
    lines += ["", "| Variant | Description |", "|---|---|"] + [f"| {name} | {variant['description']} |" for name, variant in VARIANTS.items()]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    figures = out_dir / "figures"
    figures.mkdir(exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150, sharey=True)
    figure.patch.set_facecolor(SURFACE)
    for ax, metric, title in ((axes[0], "p50_latency_s", "Median latency"), (axes[1], "p95_latency_s", "p95 latency")):
        ax.set_facecolor(SURFACE)
        for name in VARIANTS:
            part = grouped[grouped["variant"] == name]
            if part.empty:
                continue
            ax.errorbar(part["rate_per_min"], part[f"{metric}_mean"], yerr=part[f"{metric}_std"].fillna(0), color=VARIANT_COLORS[name],
                        marker="o", markersize=5, linewidth=2, capsize=3, label=name)
        ax.set_yscale("log")
        ax.set_title(title, loc="left", fontsize=11, color=TEXT)
        ax.set_xlabel("invoices arriving per minute", color=TEXT_SECONDARY, fontsize=9)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        ax.grid(color=GRID, linewidth=0.8)
    axes[0].set_ylabel("seconds (log scale)", color=TEXT_SECONDARY, fontsize=9)
    axes[1].legend(frameon=False, fontsize=9)
    figure.tight_layout()
    figure.savefig(figures / "latency_vs_rate.png", facecolor=SURFACE)
    plt.close(figure)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Latency of A and B when invoices arrive one at a time")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--rates", nargs="+", type=float, default=list(RATES_PER_MIN))
    parser.add_argument("--docs", type=int, default=60)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--rescore", action="store_true")
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()

    from evaluation.output import capture_console, go_offline, new_run_dir, package_versions, pipeline_settings, write_run_info

    if args.rescore:
        from evaluation.rescore import keep_lenient, rescore_documents, write_changes

        changes = []
        for path in sorted(Path(args.out).glob("runs/*/rate_*/run_*/summary.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            result = rescore_documents(path.parent / "documents.csv", changes)
            keep_lenient(payload, "f1_all")
            payload["f1_all"] = result["scores"]["all"]["overall"]["f1"]
            payload["f1_all_relaxed"] = result["scores_relaxed"]["all"]["overall"]["f1"]
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        write_changes(Path(args.out), changes)
    if args.report_only or args.rescore:
        print(report(Path(args.out)))
        return
    if not args.online:
        go_offline()
    unknown = [name for name in args.variants if name not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}. Available: {list(VARIANTS)}")
    out_dir = Path(args.out) if args.out else new_run_dir(f"latency_{args.docs}docs")
    out_dir.mkdir(parents=True, exist_ok=True)
    capture_console(out_dir)

    import gc

    import torch

    from evaluation.run_experiments import hardware_info, keep_system_awake
    from pipeline.dataset import load_documents

    keep_system_awake()
    documents = load_documents(indices=spread_indices(args.docs))
    started = time.time()
    write_run_info(out_dir, {
        "command": " ".join(sys.argv),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": "latency with Poisson arrivals",
        "variants": {name: VARIANTS[name]["description"] for name in args.variants},
        "rates_per_min": args.rates,
        "runs": args.runs,
        "documents": [doc.index for doc in documents],
        "documents_real": sum(doc.source == "real" for doc in documents),
        "documents_synthetic": sum(doc.source == "synthetic" for doc in documents),
        "pipeline": pipeline_settings(),
        "hardware": hardware_info(),
        "packages": package_versions(),
    })
    print(f"Output folder: {out_dir}")
    print(f"Documents: {len(documents)}; rates per minute: {args.rates}; runs: {args.runs}; variants: {args.variants}")

    built_arch, architecture = None, None
    for name in args.variants:
        variant = VARIANTS[name]
        if variant["arch"] != built_arch:
            architecture = None
            gc.collect()
            torch.cuda.empty_cache()
            print(f"\n=== Loading architecture {variant['arch']}", flush=True)
            architecture = build(variant["arch"])
            built_arch = variant["arch"]
        print(f"\n=== {name}: {variant['description']}", flush=True)
        for run in range(1, args.runs + 1):
            for rate in args.rates:
                if args.skip_existing and (out_dir / "runs" / name / f"rate_{rate:g}" / f"run_{run}" / "summary.json").exists():
                    continue
                seed = 1000 * run + int(rate)
                order = list(documents)
                random.Random(seed).shuffle(order)
                arrivals = poisson_arrivals(len(order), rate, seed)
                records = run_variant(architecture, variant, order, arrivals)
                summary = summarize(records, arrivals, rate)
                payload = save(out_dir, name, rate, run, seed, records, arrivals, summary)
                print(f"  {name} rate {rate:g}/min run {run}: median {summary['p50_latency_s']:.2f}s p95 {summary['p95_latency_s']:.2f}s "
                      f"max {summary['max_latency_s']:.2f}s OCR {summary['mean_ocr_time_s']:.3f}s F1 {payload['f1_all']}", flush=True)

    print("\n=== Report")
    print(report(out_dir))
    write_run_info(out_dir, {"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "duration_minutes": round((time.time() - started) / 60, 1)})
    print(f"\nAll results are in: {out_dir}")


if __name__ == "__main__":
    main()
