import argparse
import csv
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def tuned_thresholds(out_dir: Path | None):
    from pipeline.config import RoutingThresholds

    if out_dir is not None and (out_dir / "routing_thresholds.json").exists():
        data = json.loads((out_dir / "routing_thresholds.json").read_text(encoding="utf-8"))
        return RoutingThresholds(max_tokens_fast=data["max_tokens_fast"], min_confidence_fast=data["min_confidence_fast"]), "routing_analysis of this run"
    return RoutingThresholds(max_tokens_fast=475, min_confidence_fast=70.0), "earlier routing_analysis"


def configurations(out_dir: Path | None = None) -> dict:
    from pipeline.config import ArchAConfig, ArchBConfig

    tuned, origin = tuned_thresholds(out_dir)
    return {
        "A": {"arch": "A", "config": ArchAConfig(precision="fp32"), "description": "Baseline sequential OCR -> embeddings -> LLM, fp32, one document at a time"},
        "B": {"arch": "B", "config": ArchBConfig(), "record_warm": True, "description": "Optimized pipeline: 3 OCR workers, batch 8, cache, routing FAST=4bit / FULL=fp32"},
        "B-nocache": {"arch": "B", "config": ArchBConfig(use_cache=False), "description": "B without the result cache"},
        "B-allfull": {"arch": "B", "config": ArchBConfig(routing="all_full"), "description": "B with every document on the FULL fp32 route (no quantized model)"},
        "B-allfast": {"arch": "B", "config": ArchBConfig(routing="all_fast"), "description": "B with every document on the FAST 4-bit route"},
        "B-8bit": {"arch": "B", "config": ArchBConfig(routing="all_fast", fast_precision="8bit"), "description": "B with every document on an 8-bit model"},
        "B-fp16": {"arch": "B", "config": ArchBConfig(routing="all_fast", fast_precision="fp16"), "description": "B with every document on an fp16 model"},
        "B-1worker": {"arch": "B", "config": ArchBConfig(ocr_workers=1), "description": "B with a single OCR worker (no OCR parallelism)"},
        "B-6workers": {"arch": "B", "config": ArchBConfig(ocr_workers=6), "description": "B with six OCR workers"},
        "B-nobatch": {"arch": "B", "config": ArchBConfig(batch_size=1), "description": "B with batch size 1 (no GPU batching)"},
        "B-tuned": {"arch": "B", "config": ArchBConfig(thresholds=tuned), "record_warm": False, "description": f"B with thresholds from {origin}: {tuned.max_tokens_fast} tokens, confidence {tuned.min_confidence_fast}"},
    }


def keep_system_awake() -> None:
    if sys.platform != "win32":
        return
    import ctypes

    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_AWAYMODE_REQUIRED = 0x00000040
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)


def hardware_info() -> dict:
    import psutil
    import torch

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        info["gpu"] = properties.name
        info["gpu_vram_gb"] = round(properties.total_memory / 2**30, 1)
    return info


def config_to_dict(config) -> dict:
    data = asdict(config)
    return data


def write_documents(path: Path, records, scores) -> None:
    rows = []
    for record, score in zip(records, scores):
        row = record.to_row()
        for key, value in score.items():
            row[f"score_{key}"] = value
        rows.append(row)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def score_records(records) -> tuple[list[dict], dict]:
    from evaluation.scorer import aggregate_by_source, score_both

    scores = [score_both(record.prediction, record.truth) for record in records]
    return scores, aggregate_by_source(scores, [record.source for record in records])


def relaxed_scores(records, scores) -> dict:
    from evaluation.scorer import RELAXED_PREFIX, aggregate_by_source

    return aggregate_by_source(scores, [record.source for record in records], RELAXED_PREFIX)


def review_summary(records, scores) -> dict:
    from evaluation.scorer import review_summary as summarize

    entries = [
        {"source": record.source, "needs_review": bool(record.features.get("needs_review")), "all_correct": score["all_correct"]}
        for record, score in zip(records, scores)
        if not record.error
    ]
    return summarize(entries)


def build_architecture(entry: dict, out_dir: Path | None = None):
    import torch
    from pipeline.arch_a import ArchitectureA
    from pipeline.arch_b import ArchitectureB

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    if entry["arch"] == "A":
        architecture = ArchitectureA(entry["config"])
    else:
        architecture = ArchitectureB(entry["config"], cache_path=out_dir / "cache.sqlite" if out_dir else None)
    load_time = time.perf_counter() - started
    torch.cuda.synchronize()
    return architecture, {
        "model_load_time_s": round(load_time, 2),
        "torch_allocated_after_load_mb": round(torch.cuda.memory_allocated() / 2**20, 1),
    }


def execute(architecture, documents, label: str):
    from pipeline.metrics import ResourceMonitor

    total = len(documents)
    last_print = time.perf_counter()

    def progress(done, _total, _record):
        nonlocal last_print
        now = time.perf_counter()
        if now - last_print >= 15 or done == total:
            print(f"    [{label}] {done}/{total} documents, {now - started:.0f}s elapsed", flush=True)
            last_print = now

    monitor = ResourceMonitor().start()
    started = time.perf_counter()
    records = architecture.run(documents, progress)
    wall = time.perf_counter() - started
    resources = monitor.stop()
    return records, wall, resources


def save_run(out_dir: Path, name: str, run: int, entry: dict, records, wall, resources, load_info, extra) -> dict:
    import torch
    from pipeline.metrics import summarize_run

    scores, aggregated = score_records(records)
    run_dir = out_dir / "runs" / name / f"run_{run}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_documents(run_dir / "documents.csv", records, scores)
    summary = summarize_run(records, wall, resources, extra)
    summary.update(load_info)
    summary["torch_peak_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    payload = {
        "config_name": name,
        "architecture": entry["arch"],
        "description": entry["description"],
        "config": config_to_dict(entry["config"]),
        "run": run,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": hardware_info(),
        "metrics": summary,
        "scores": aggregated,
        "scores_relaxed": relaxed_scores(records, scores),
        "review": review_summary(records, scores),
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(
        f"  {name} run {run}: wall={summary['wall_time_s']}s throughput={summary['throughput_docs_per_min']} docs/min "
        f"F1={aggregated['all']['overall']['f1']} straight-through={payload['review']['all']['straight_through_rate']:.1%} "
        f"peakRSS={summary['peak_rss_mb']}MB peakGPU={summary['peak_gpu_mb']}MB",
        flush=True,
    )
    return payload


def ordered_names(requested: list[str], available: dict) -> list[str]:
    names = list(available) if requested == ["all"] else list(requested)
    unknown = [name for name in names if name not in available]
    if unknown:
        raise SystemExit(f"Unknown configurations: {unknown}. Available: {list(available)} or all")
    if "B-tuned" in names:
        names = [name for name in names if name != "B-tuned"] + ["B-tuned"]
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Architecture A / B experiments into a new output folder")
    parser.add_argument("--configs", nargs="+", default=["A", "B"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--source", choices=["real", "synthetic"], default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()

    from evaluation.output import capture_console, go_offline, mark_latest, new_run_dir, package_versions, pipeline_settings, write_run_info

    if not args.online:
        go_offline()

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        label = args.name or "_".join(["all"] if args.configs == ["all"] else args.configs)
        if args.limit:
            label += f"_{args.limit}docs"
        if args.source:
            label += f"_{args.source}"
        out_dir = new_run_dir(label)
    capture_console(out_dir)

    import gc
    import torch
    from pipeline.dataset import load_documents

    keep_system_awake()
    documents = load_documents(limit=args.limit, source=args.source)
    names = ordered_names(args.configs, configurations(out_dir))
    started = time.time()
    write_run_info(out_dir, {
        "command": " ".join(sys.argv),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "configurations": names,
        "runs_per_configuration": args.runs,
        "documents": len(documents),
        "documents_real": sum(d.source == "real" for d in documents),
        "documents_synthetic": sum(d.source == "synthetic" for d in documents),
        "pipeline": pipeline_settings(),
        "hardware": hardware_info(),
        "packages": package_versions(),
    })

    print(f"Output folder: {out_dir}")
    print(f"Documents: {len(documents)} ({sum(d.source == 'real' for d in documents)} real, "
          f"{sum(d.source == 'synthetic' for d in documents)} synthetic)")
    print(f"Configurations: {', '.join(names)}  x {args.runs} runs")

    for name in names:
        if name == "B-tuned" and (out_dir / "runs" / "B-allfast").exists() and (out_dir / "runs" / "B-allfull").exists():
            from evaluation.routing_analysis import analyze

            chosen = analyze(out_dir)["thresholds"]
            print(f"\nRouting thresholds from this run's data: {chosen['max_tokens_fast']} tokens, confidence {chosen['min_confidence_fast']}")
        entry = configurations(out_dir)[name]
        runs_needed = [
            run for run in range(1, args.runs + 1)
            if not (args.skip_existing and (out_dir / "runs" / name / f"run_{run}" / "summary.json").exists())
        ]
        if not runs_needed:
            print(f"\n=== {name}: all {args.runs} runs present, skipping")
            continue
        print(f"\n=== {name}: {entry['description']}")
        architecture, load_info = build_architecture(entry, out_dir)
        for run in runs_needed:
            if entry["arch"] == "B":
                architecture.clear_cache()
            records, wall, resources = execute(architecture, documents, f"{name} run {run}")
            extra = architecture.cache_stats() if entry["arch"] == "B" else {}
            save_run(out_dir, name, run, entry, records, wall, resources, load_info, extra)
            if entry.get("record_warm"):
                records, wall, resources = execute(architecture, documents, f"{name}-warm run {run}")
                extra = architecture.cache_stats()
                save_run(out_dir, f"{name}-warm", run, entry, records, wall, resources, load_info, extra)
        del architecture
        gc.collect()
        torch.cuda.empty_cache()

    print("\n=== Report")
    try:
        from evaluation.report import generate

        print(generate(out_dir))
    except Exception as exc:
        print(f"Report could not be generated: {type(exc).__name__}: {exc}")
    write_run_info(out_dir, {
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_minutes": round((time.time() - started) / 60, 1),
    })
    mark_latest(out_dir)
    print(f"\nAll results are in: {out_dir}")


if __name__ == "__main__":
    main()
