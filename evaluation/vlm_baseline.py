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

VLM_NAME = "Qwen/Qwen2-VL-2B-Instruct"
PATCH = 28
PROMPT = (
    "Read this invoice and extract three fields: the invoice number, the invoice date and the total amount due. "
    'Answer with JSON only, exactly in this form: {"invoice_number": "...", "date": "...", "total": "..."}. '
    "Copy each value as it is written on the invoice. Use null for a field that is not on the invoice."
)


def parse_answer(answer: str) -> tuple[dict, bool]:
    from pipeline.config import FIELDS

    match = re.search(r"\{.*\}", answer, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return {field: data.get(field) for field in FIELDS}, True
        except json.JSONDecodeError:
            pass
    values = {}
    for field in FIELDS:
        found = re.search(rf'"?{field}"?\s*:\s*"?([^",}}\n]*)', answer)
        values[field] = found.group(1).strip() if found else None
    return values, False


class VlmExtractor:
    def __init__(self, max_visual_tokens: int, device: str = "cuda"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        self.device = device
        self.max_visual_tokens = max_visual_tokens
        pixels = max_visual_tokens * PATCH * PATCH
        self.processor = AutoProcessor.from_pretrained(VLM_NAME, min_pixels=64 * PATCH * PATCH, max_pixels=pixels)
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(VLM_NAME, quantization_config=config, device_map=device).eval()

    def extract(self, image_path) -> tuple[dict, str, bool, int]:
        import torch
        from PIL import Image, ImageOps

        from pipeline.normalize import clean_prediction

        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt").to(self.device)
        visual_tokens = int(inputs["image_grid_thw"].prod().item() // 4) if "image_grid_thw" in inputs else 0
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)
        answer = self.processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        raw, valid_json = parse_answer(answer)
        prediction = {field: clean_prediction(field, "" if value is None else str(value)) for field, value in raw.items()}
        return prediction, answer, valid_json, visual_tokens


def percentile(values, q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low, high = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def run_setting(max_visual_tokens: int, documents, out_dir: Path) -> dict:
    import torch

    from evaluation.run_experiments import relaxed_scores, score_records, write_documents
    from pipeline.metrics import DocumentRecord, ResourceMonitor

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    extractor = VlmExtractor(max_visual_tokens)
    load_time = time.perf_counter() - started
    extractor.extract(documents[0].image_path)
    setting_dir = out_dir / f"tokens_{max_visual_tokens}"
    setting_dir.mkdir(parents=True, exist_ok=True)
    records, answers = [], []
    monitor = ResourceMonitor().start()
    wall_start = time.perf_counter()
    for position, doc in enumerate(documents, start=1):
        truth = doc.load_truth()
        mark = time.perf_counter()
        try:
            prediction, answer, valid_json, visual_tokens = extractor.extract(doc.image_path)
            error = ""
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            prediction, answer, valid_json, visual_tokens, error = {}, "", False, 0, f"OutOfMemoryError: {exc}"
        elapsed = time.perf_counter() - mark
        records.append(DocumentRecord(
            name=doc.name, index=doc.index, source=doc.source, prediction=prediction, truth=truth, route=f"VLM-{max_visual_tokens}",
            llm_time=elapsed, service_time=elapsed, latency=elapsed, error=error,
            features={"valid_json": valid_json, "visual_tokens": visual_tokens},
        ))
        answers.append({"index": doc.index, "answer": answer})
        if position % 25 == 0 or position == len(documents):
            print(f"    [VLM {max_visual_tokens} tokens] {position}/{len(documents)}, {time.perf_counter() - wall_start:.0f}s", flush=True)
    wall = time.perf_counter() - wall_start
    resources = monitor.stop()
    scores, aggregated = score_records(records)
    write_documents(setting_dir / "documents.csv", records, scores)
    (setting_dir / "answers.jsonl").write_text("\n".join(json.dumps(a, ensure_ascii=False) for a in answers) + "\n", encoding="utf-8")
    ok = [record for record in records if not record.error]
    times = [record.service_time for record in ok]
    summary = {
        "model": VLM_NAME,
        "quantization": "bitsandbytes 4-bit NF4, double quantization, bf16 compute",
        "max_visual_tokens": max_visual_tokens,
        "documents": len(records),
        "errors": len(records) - len(ok),
        "out_of_memory": sum("OutOfMemory" in record.error for record in records),
        "model_load_time_s": round(load_time, 2),
        "wall_time_s": round(wall, 2),
        "throughput_docs_per_min": round(len(records) / wall * 60, 2),
        "mean_time_per_invoice_s": round(statistics.fmean(times), 4) if times else None,
        "median_time_per_invoice_s": round(percentile(times, 50), 4) if times else None,
        "p95_time_per_invoice_s": round(percentile(times, 95), 4) if times else None,
        "mean_visual_tokens": round(statistics.fmean(r.features["visual_tokens"] for r in ok), 1) if ok else None,
        "valid_json_rate": round(sum(r.features["valid_json"] for r in ok) / len(ok), 4) if ok else None,
        "torch_peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "peak_rss_mb": resources.get("peak_rss_mb"),
        "peak_gpu_mb": resources.get("peak_gpu_mb"),
    }
    relaxed = relaxed_scores(records, scores)
    apply_scores(summary, aggregated, relaxed)
    payload = {"summary": summary, "scores": aggregated, "scores_relaxed": relaxed}
    (setting_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  VLM {max_visual_tokens} tokens: {summary['mean_time_per_invoice_s']}s/invoice, F1 {summary['f1'].get('all')}, "
          f"real {summary['f1'].get('real')}, GPU peak {summary['torch_peak_allocated_mb']} MB, OOM {summary['out_of_memory']}", flush=True)
    del extractor
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    return summary


def write_markdown(out_dir: Path, summaries: list[dict]) -> str:
    lines = [f"# Vision-language model baseline: {VLM_NAME}", "",
             "The VLM reads the invoice image directly (no OCR) and answers in JSON; answers go through the same normalisation and scorer as A and B. "
             "One run, one invoice at a time. F1 and fully correct use strict matching; the relaxed columns add fuzzy invoice numbers "
             "and day/month swaps of ambiguous numeric dates.", "",
             "| Max visual tokens | Mean visual tokens | Time per invoice mean / median / p95 (s) | Throughput (/min) | GPU peak (MB) | Valid JSON | F1 all | F1 real | F1 synthetic | Real: number / date / total | Fully correct | F1 all, relaxed | Fully correct, relaxed | Out of memory |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        real = s.get("f1_fields_real") or {}
        lines.append(
            f"| {s['max_visual_tokens']} | {s['mean_visual_tokens']} | {s['mean_time_per_invoice_s']} / {s['median_time_per_invoice_s']} / {s['p95_time_per_invoice_s']} | "
            f"{s['throughput_docs_per_min']} | {s['torch_peak_allocated_mb']:.0f} | {s['valid_json_rate']} | {s['f1'].get('all')} | {s['f1'].get('real')} | "
            f"{s['f1'].get('synthetic')} | {real.get('invoice_number', float('nan')):.2f} / {real.get('date', float('nan')):.2f} / {real.get('total', float('nan')):.2f} | "
            f"{s['document_accuracy']} | {s.get('f1_relaxed')} | {s.get('document_accuracy_relaxed')} | {s['out_of_memory']} |")
    text = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(text, encoding="utf-8")
    return text


def apply_scores(summary: dict, aggregated: dict, relaxed: dict) -> dict:
    fields = ("invoice_number", "date", "total")
    summary["f1"] = {source: aggregated[source]["overall"]["f1"] for source in ("all", "real", "synthetic") if source in aggregated}
    summary["f1_fields"] = {field: aggregated["all"][field]["f1"] for field in fields}
    summary["f1_fields_real"] = {field: aggregated["real"][field]["f1"] for field in fields} if "real" in aggregated else {}
    summary["document_accuracy"] = aggregated["all"]["document_accuracy"]
    summary["f1_relaxed"] = relaxed["all"]["overall"]["f1"]
    summary["document_accuracy_relaxed"] = relaxed["all"]["document_accuracy"]
    return summary


def rescore(out_dir: Path) -> str:
    from evaluation.rescore import keep_lenient, rescore_documents, write_changes

    summaries = json.loads((out_dir / "summaries.json").read_text(encoding="utf-8"))
    changes = []
    for summary in summaries:
        setting_dir = out_dir / f"tokens_{summary['max_visual_tokens']}"
        result = rescore_documents(setting_dir / "documents.csv", changes)
        payload = json.loads((setting_dir / "summary.json").read_text(encoding="utf-8"))
        keep_lenient(payload, "scores")
        payload["summary"] = apply_scores(summary, result["scores"], result["scores_relaxed"])
        payload["scores"] = result["scores"]
        payload["scores_relaxed"] = result["scores_relaxed"]
        (setting_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_changes(out_dir, changes)
    (out_dir / "summaries.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return write_markdown(out_dir, sorted(summaries, key=lambda s: s["max_visual_tokens"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Vision-language model reading invoices directly, on the same data and scorer")
    parser.add_argument("--tokens", nargs="+", type=int, default=[1024])
    parser.add_argument("--docs", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--rescore", action="store_true")
    args = parser.parse_args()

    from evaluation.output import capture_console, go_offline, new_run_dir, package_versions, write_run_info

    if args.rescore:
        print(rescore(Path(args.out)))
        return
    if not args.online:
        go_offline()
    label = "vlm_baseline" + (f"_{args.docs}docs" if args.docs else "")
    out_dir = Path(args.out) if args.out else new_run_dir(label)
    out_dir.mkdir(parents=True, exist_ok=True)
    capture_console(out_dir)

    from evaluation.latency_experiment import spread_indices
    from evaluation.run_experiments import hardware_info, keep_system_awake
    from pipeline.dataset import load_documents

    keep_system_awake()
    documents = load_documents(indices=spread_indices(args.docs)) if args.docs else load_documents()
    started = time.time()
    write_run_info(out_dir, {
        "command": " ".join(sys.argv), "started": time.strftime("%Y-%m-%d %H:%M:%S"), "experiment": "VLM baseline",
        "model": VLM_NAME, "prompt": PROMPT, "max_visual_tokens": args.tokens, "documents": len(documents),
        "hardware": hardware_info(), "packages": package_versions(),
    })
    print(f"Output folder: {out_dir}\nDocuments: {len(documents)}; visual-token limits: {args.tokens}")
    summaries = []
    existing = out_dir / "summaries.json"
    if existing.exists():
        summaries = json.loads(existing.read_text(encoding="utf-8"))
    for tokens in args.tokens:
        print(f"\n=== VLM with at most {tokens} visual tokens", flush=True)
        summaries = [s for s in summaries if s["max_visual_tokens"] != tokens] + [run_setting(tokens, documents, out_dir)]
        existing.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    summaries.sort(key=lambda s: s["max_visual_tokens"])
    print("\n" + write_markdown(out_dir, summaries))
    write_run_info(out_dir, {"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "duration_minutes": round((time.time() - started) / 60, 1)})
    print(f"All results are in: {out_dir}")


if __name__ == "__main__":
    main()
