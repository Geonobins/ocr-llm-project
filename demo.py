import argparse
import csv
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def line(label, payload):
    m = payload["metrics"]
    print(
        f"{label:<22} wall {m['wall_time_s']:7.2f} s   throughput {m['throughput_docs_per_min']:8.1f} docs/min   "
        f"OCR/doc {m['mean_ocr_time_s']:.2f} s   LLM/doc {m['mean_llm_time_s']:.2f} s   "
        f"F1 {payload['scores']['all']['overall']['f1']:.3f}   straight-through {payload['review']['all']['straight_through_rate']:.0%}"
    )


def main():
    parser = argparse.ArgumentParser(description="Side-by-side demonstration of Architecture A and B, saved to a new output folder")
    parser.add_argument("--docs", type=int, default=30)
    parser.add_argument("--source", choices=["real", "synthetic"], default=None)
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()

    from evaluation.output import capture_console, go_offline, mark_latest, new_run_dir, package_versions, pipeline_settings, write_run_info

    if not args.online:
        go_offline()
    label = f"demo_{args.docs}docs" + (f"_{args.source}" if args.source else "")
    out_dir = new_run_dir(label)
    capture_console(out_dir)

    import torch
    from evaluation.run_experiments import build_architecture, configurations, execute, hardware_info, save_run
    from pipeline.dataset import load_documents

    documents = load_documents(limit=args.docs, source=args.source)
    write_run_info(out_dir, {
        "command": " ".join(sys.argv),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "documents": len(documents),
        "pipeline": pipeline_settings(),
        "hardware": hardware_info(),
        "packages": package_versions(),
    })
    print(f"\nOutput folder: {out_dir}")
    print(f"{len(documents)} documents: {documents[0].name} ... {documents[-1].name}\n")
    entries = configurations(out_dir)

    print("Architecture A: one document at a time, OCR -> filter -> 3 LLM calls, fp32")
    a, load_a = build_architecture(entries["A"], out_dir)
    records_a, wall, resources = execute(a, documents, "A")
    payload_a = save_run(out_dir, "A", 1, entries["A"], records_a, wall, resources, load_a, {})
    line("A (sequential)", payload_a)
    del a
    torch.cuda.empty_cache()

    print("\nArchitecture B: 3 OCR workers + batched GPU stage + routing + cache")
    b, load_b = build_architecture(entries["B"], out_dir)
    b.clear_cache()
    records_b, wall, resources = execute(b, documents, "B")
    payload_b = save_run(out_dir, "B", 1, entries["B"], records_b, wall, resources, load_b, b.cache_stats())
    line("B (cold cache)", payload_b)
    routes = {}
    for r in records_b:
        routes[r.route] = routes.get(r.route, 0) + 1
    print(f"{'':<22} routes {routes}")
    records_w, wall, resources = execute(b, documents, "B-warm")
    payload_w = save_run(out_dir, "B-warm", 1, entries["B"], records_w, wall, resources, load_b, b.cache_stats())
    line("B (warm cache)", payload_w)

    same = sum(1 for x, y in zip(records_a, records_b) if x.prediction == y.prediction)
    by_route = {}
    for x, y in zip(records_a, records_b):
        hit, total = by_route.get(y.route, (0, 0))
        by_route[y.route] = (hit + (x.prediction == y.prediction), total + 1)
    detail = ", ".join(f"{route} {hit}/{total}" for route, (hit, total) in sorted(by_route.items()))
    wall_a = payload_a["metrics"]["wall_time_s"]
    print(f"\nIdentical answers A vs B: {same}/{len(documents)}  by B route: {detail}")
    print("FULL uses the same fp32 model as A; FAST uses the 4-bit model with a shorter answer budget")
    print(f"Speed-up B over A on this batch: {wall_a / payload_b['metrics']['wall_time_s']:.1f}x cold, "
          f"{wall_a / max(payload_w['metrics']['wall_time_s'], 1e-6):.0f}x warm\n")

    with open(out_dir / "comparison.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["document", "field", "truth", "A", "B", "B route", "B needs review", "identical"])
        for x, y in zip(records_a, records_b):
            for field in ("invoice_number", "date", "total"):
                writer.writerow([x.name, field, x.truth.get(field, ""), x.prediction.get(field, ""), y.prediction.get(field, ""),
                                 y.route, y.features.get("needs_review", ""), x.prediction.get(field) == y.prediction.get(field)])

    print(f"{'document':<18}{'A answer (total)':<20}{'B answer (total)':<20}{'truth':<16}{'B route'}")
    for x, y in zip(records_a[:12], records_b[:12]):
        print(f"{x.name:<18}{x.prediction.get('total', ''):<20}{y.prediction.get('total', ''):<20}{str(x.truth.get('total', '')):<16}{y.route}")

    from evaluation.report import generate

    generate(out_dir)
    write_run_info(out_dir, {"finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    mark_latest(out_dir)
    print(f"\nAll results are in: {out_dir}")


if __name__ == "__main__":
    main()
