import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ENGINES = ("tesseract", "easyocr")


def field_found(field: str, truth, text: str) -> bool | None:
    from rapidfuzz import fuzz

    from pipeline.normalize import clean_text, is_null, normalize_invoice_number, parse_amount
    from pipeline.validate import _amounts

    if is_null(truth):
        return None
    if field == "invoice_number":
        key = normalize_invoice_number(truth)
        return bool(key) and key in re.sub(r"[^A-Za-z0-9]", "", text).upper()
    if field == "total":
        value = parse_amount(truth)
        if value is None:
            return False
        return any(abs(found - value) < 0.005 for line in text.splitlines() for found, _ in _amounts(line))
    target = clean_text(truth).lower()
    squeezed = re.sub(r"[^a-z0-9]", "", target)
    return fuzz.partial_ratio(target, text.lower()) >= 90 or (bool(squeezed) and squeezed in re.sub(r"[^a-z0-9]", "", text.lower()))


def percentile(values, q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low, high = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def read_all(engine: str, documents, out_dir: Path) -> dict:
    import torch

    from pipeline.config import OCR_LAYOUT
    from pipeline.ocr import EasyOcrEngine, configure_worker, run_ocr

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    if engine == "tesseract":
        configure_worker()
        reader = lambda path: run_ocr(path, OCR_LAYOUT)
    else:
        easy = EasyOcrEngine(gpu=True)
        reader = lambda path: easy.run(path, OCR_LAYOUT)
    load_time = time.perf_counter() - started
    reader(documents[0].image_path)
    results = {}
    with open(out_dir / f"ocr_{engine}.jsonl", "w", encoding="utf-8") as handle:
        for position, doc in enumerate(documents, start=1):
            result = reader(doc.image_path)
            results[doc.index] = result
            handle.write(json.dumps({"index": doc.index, "source": doc.source, **result.to_dict()}) + "\n")
            if position % 100 == 0 or position == len(documents):
                print(f"    [{engine}] {position}/{len(documents)} read", flush=True)
    peak = torch.cuda.max_memory_allocated() / 2**20
    if engine == "easyocr":
        del easy
        torch.cuda.empty_cache()
    return {"results": results, "load_time_s": round(load_time, 2), "gpu_peak_mb": round(peak, 1)}


def extract_all(engine: str, documents, results: dict, selector, extractor):
    from pipeline.config import FIELDS, MAX_NEW_TOKENS_FULL, RECONCILE_TOTALS
    from pipeline.metrics import DocumentRecord
    from pipeline.validate import finalize

    records = []
    for doc in documents:
        ocr = results[doc.index]
        truth = doc.load_truth()
        mark = time.perf_counter()
        context = selector.select(ocr.text)
        embed_time = time.perf_counter() - mark
        mark = time.perf_counter()
        prediction = extractor.extract_sequential(context, MAX_NEW_TOKENS_FULL)
        llm_time = time.perf_counter() - mark
        prediction, total_source = finalize(prediction, ocr.text, RECONCILE_TOTALS)
        features = {"ocr_confidence": round(ocr.confidence, 2), "ocr_words": ocr.word_count, "total_source": total_source}
        for field in FIELDS:
            features[f"in_ocr_{field}"] = field_found(field, truth.get(field), ocr.text)
        records.append(DocumentRecord(
            name=doc.name, index=doc.index, source=doc.source, prediction=prediction, truth=truth, route=engine,
            ocr_time=ocr.duration, embed_time=embed_time, llm_time=llm_time,
            service_time=ocr.duration + embed_time + llm_time, latency=ocr.duration + embed_time + llm_time, features=features,
        ))
    return records


def summarize(engine: str, documents, records, read_info: dict, aggregated: dict) -> dict:
    from pipeline.config import FIELDS

    durations = [record.ocr_time for record in records]
    summary = {
        "engine": engine,
        "documents": len(records),
        "load_time_s": read_info["load_time_s"],
        "ocr_gpu_peak_mb": read_info["gpu_peak_mb"],
        "mean_ocr_time_s": round(statistics.fmean(durations), 4),
        "median_ocr_time_s": round(percentile(durations, 50), 4),
        "p95_ocr_time_s": round(percentile(durations, 95), 4),
        "mean_confidence": round(statistics.fmean(record.features["ocr_confidence"] for record in records), 2),
        "f1": {source: aggregated[source]["overall"]["f1"] for source in ("all", "real", "synthetic") if source in aggregated},
        "f1_fields": {field: aggregated["all"][field]["f1"] for field in FIELDS},
        "document_accuracy": aggregated["all"]["document_accuracy"],
        "found_in_ocr_text": {},
    }
    for source in ("all", "real", "synthetic"):
        subset = [record for record in records if source == "all" or record.source == source]
        if not subset:
            continue
        summary[f"mean_ocr_time_s_{source}"] = round(statistics.fmean(record.ocr_time for record in subset), 4)
        found = {}
        for field in FIELDS:
            flags = [record.features[f"in_ocr_{field}"] for record in subset if record.features[f"in_ocr_{field}"] is not None]
            found[field] = round(sum(flags) / len(flags), 4) if flags else None
        summary["found_in_ocr_text"][source] = found
    return summary


def write_markdown(out_dir: Path, summaries: list[dict]) -> str:
    lines = ["# OCR engine comparison", "",
             "Each engine reads all invoices once; its text then goes through the same downstream as Architecture A "
             "(row layout, MiniLM line filter, Flan-T5-base fp32, total check) and the same scorer. "
             "F1 and fully correct use strict matching; the relaxed columns add fuzzy invoice numbers and day/month swaps of ambiguous numeric dates.", "",
             "| Engine | OCR time mean / median / p95 (s) | Real / synthetic mean (s) | OCR GPU peak (MB) | Mean confidence | F1 all | F1 real | F1 synthetic | Fully correct | F1 all, relaxed | Fully correct, relaxed |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        lines.append(f"| {s['engine']} | {s['mean_ocr_time_s']:.3f} / {s['median_ocr_time_s']:.3f} / {s['p95_ocr_time_s']:.3f} | "
                     f"{s.get('mean_ocr_time_s_real', float('nan')):.3f} / {s.get('mean_ocr_time_s_synthetic', float('nan')):.3f} | {s['ocr_gpu_peak_mb']:.0f} | {s['mean_confidence']:.1f} | "
                     f"{s['f1']['all']:.4f} | {s['f1'].get('real', float('nan')):.4f} | {s['f1'].get('synthetic', float('nan')):.4f} | {s['document_accuracy']:.4f} | "
                     f"{s.get('f1_relaxed', float('nan')):.4f} | {s.get('document_accuracy_relaxed', float('nan')):.4f} |")
    lines += ["", "Share of ground-truth values that appear in the OCR text (an upper bound for extraction):", "",
              "| Engine | Subset | Invoice number | Date | Total |", "|---|---|---|---|---|"]
    for s in summaries:
        for source, found in s["found_in_ocr_text"].items():
            lines.append(f"| {s['engine']} | {source} | {found['invoice_number']:.3f} | {found['date']:.3f} | {found['total']:.3f} |")
    lines += ["", "| Engine | F1 invoice number | F1 date | F1 total |", "|---|---|---|---|"]
    for s in summaries:
        lines.append(f"| {s['engine']} | {s['f1_fields']['invoice_number']:.4f} | {s['f1_fields']['date']:.4f} | {s['f1_fields']['total']:.4f} |")
    text = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(text, encoding="utf-8")
    return text


def apply_scores(summary: dict, aggregated: dict, relaxed: dict) -> dict:
    from pipeline.config import FIELDS

    summary["f1"] = {source: aggregated[source]["overall"]["f1"] for source in ("all", "real", "synthetic") if source in aggregated}
    summary["f1_fields"] = {field: aggregated["all"][field]["f1"] for field in FIELDS}
    summary["document_accuracy"] = aggregated["all"]["document_accuracy"]
    summary["f1_relaxed"] = relaxed["all"]["overall"]["f1"]
    summary["document_accuracy_relaxed"] = relaxed["all"]["document_accuracy"]
    return summary


def refresh_found(out_dir: Path, engine: str) -> dict:
    import csv

    from pipeline.config import FIELDS

    with open(out_dir / f"ocr_{engine}.jsonl", encoding="utf-8") as handle:
        texts = {item["index"]: item["text"] for item in map(json.loads, handle)}
    documents = out_dir / engine / "documents.csv"
    with open(documents, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows, fieldnames = list(reader), reader.fieldnames
    for row in rows:
        for field in FIELDS:
            found = field_found(field, row.get(f"truth_{field}") or None, texts[int(row["index"])])
            row[f"feat_in_ocr_{field}"] = "" if found is None else str(found)
    with open(documents, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    result = {}
    for source in ("all", "real", "synthetic"):
        subset = [row for row in rows if source == "all" or row["source"] == source]
        if not subset:
            continue
        result[source] = {}
        for field in FIELDS:
            flags = [row[f"feat_in_ocr_{field}"] == "True" for row in subset if row[f"feat_in_ocr_{field}"] != ""]
            result[source][field] = round(sum(flags) / len(flags), 4) if flags else None
    return result


def rescore(out_dir: Path) -> str:
    from evaluation.rescore import keep_lenient, rescore_documents, write_changes

    summaries, changes = [], []
    for engine in ENGINES:
        path = out_dir / engine / "summary.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = rescore_documents(out_dir / engine / "documents.csv", changes)
        keep_lenient(payload, "scores")
        payload["scores"] = result["scores"]
        payload["scores_relaxed"] = result["scores_relaxed"]
        payload["summary"] = apply_scores(payload["summary"], result["scores"], result["scores_relaxed"])
        payload["summary"]["found_in_ocr_text"] = refresh_found(out_dir, engine)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        summaries.append(payload["summary"])
    write_changes(out_dir, changes)
    return write_markdown(out_dir, summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Tesseract and EasyOCR on speed and on extraction accuracy")
    parser.add_argument("--engines", nargs="+", default=list(ENGINES))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--rescore", action="store_true")
    args = parser.parse_args()

    from evaluation.output import capture_console, go_offline, new_run_dir, package_versions, pipeline_settings, write_run_info

    if args.rescore:
        print(rescore(Path(args.out)))
        return
    if not args.online:
        go_offline()
    out_dir = Path(args.out) if args.out else new_run_dir("ocr_comparison" + (f"_{args.limit}docs" if args.limit else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    capture_console(out_dir)

    import gc

    import torch

    from evaluation.run_experiments import hardware_info, keep_system_awake, relaxed_scores, score_records
    from pipeline.dataset import load_documents
    from pipeline.embeddings import LineSelector
    from pipeline.llm import FieldExtractor

    keep_system_awake()
    documents = load_documents(limit=args.limit)
    started = time.time()
    versions = package_versions()
    try:
        import easyocr

        versions["easyocr"] = easyocr.__version__
    except Exception:
        versions["easyocr"] = "not installed"
    write_run_info(out_dir, {
        "command": " ".join(sys.argv), "started": time.strftime("%Y-%m-%d %H:%M:%S"), "experiment": "OCR engine comparison",
        "engines": args.engines, "documents": len(documents), "pipeline": pipeline_settings(), "hardware": hardware_info(), "packages": versions,
    })
    print(f"Output folder: {out_dir}\nDocuments: {len(documents)}; engines: {args.engines}")

    read = {}
    for engine in args.engines:
        print(f"\n=== Reading with {engine}", flush=True)
        read[engine] = read_all(engine, documents, out_dir)
        gc.collect()
        torch.cuda.empty_cache()

    print("\n=== Extracting with the Architecture A downstream (fp32)", flush=True)
    selector = LineSelector("cuda")
    extractor = FieldExtractor("fp32", "cuda")
    summaries = []
    for engine in args.engines:
        records = extract_all(engine, documents, read[engine]["results"], selector, extractor)
        scores, aggregated = score_records(records)
        engine_dir = out_dir / engine
        engine_dir.mkdir(exist_ok=True)
        from evaluation.run_experiments import write_documents

        write_documents(engine_dir / "documents.csv", records, scores)
        relaxed = relaxed_scores(records, scores)
        summary = apply_scores(summarize(engine, documents, records, read[engine], aggregated), aggregated, relaxed)
        payload = {"summary": summary, "scores": aggregated, "scores_relaxed": relaxed}
        (engine_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        summaries.append(summary)
        print(f"  {engine}: OCR {summary['mean_ocr_time_s']:.3f}s/invoice, F1 {summary['f1']['all']}, real {summary['f1'].get('real')}", flush=True)

    print("\n" + write_markdown(out_dir, summaries))
    write_run_info(out_dir, {"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "duration_minutes": round((time.time() - started) / 60, 1)})
    print(f"All results are in: {out_dir}")


if __name__ == "__main__":
    main()
