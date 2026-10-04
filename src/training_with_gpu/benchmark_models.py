"""Benchmark candidate Whisper checkpoints on the project's fixed Qaloon split.

Every model is scored with the same split (qaloon_data.load_splits, seed 42),
the same normalizer and the same jiwer metrics as gpu_evaluation.py, so the
numbers are directly comparable with the published full-model result.
Decoding is blind and greedy: no expected text is ever given to the model.

    python src/training_with_gpu/benchmark_models.py --data-root D:/data/all_reciters \
        --model ours=runs/gpu_base_full --model tarteel=D:/models/whisper-base-ar-quran
"""
import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import torch
from transformers import GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "training"))
sys.path.insert(0, str(ROOT / "src" / "dataset_collection"))
from qaloon_data import load_splits  # noqa: E402
from qaloon_audio2text import normalize_quran_for_asr  # noqa: E402
from gpu_evaluation import _scores  # noqa: E402

# Checkpoints saved without a multilingual generation config get it from their
# OpenAI base, identified by architecture (d_model, encoder layers, decoder layers).
OPENAI_BASES = {
    (384, 4, 4): "openai/whisper-tiny",
    (512, 6, 6): "openai/whisper-base",
    (768, 12, 12): "openai/whisper-small",
    (1024, 24, 24): "openai/whisper-medium",
    (1280, 32, 4): "openai/whisper-large-v3-turbo",
    (1280, 32, 32): "openai/whisper-large-v3",
}
WINDOW_SECONDS = 28  # Same rolling window as src/streaming/server.py.


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--model", action="append", required=True,
                   help="NAME=PATH; append @native to use the checkpoint's own decoder prefix instead of forcing Arabic")
    p.add_argument("--split", choices=("train", "validation", "test", "all"), default="test",
                   help="'all' is for reciters never used in training (unseen-voice evaluation)")
    p.add_argument("--reciters", nargs="+", help="Default: every reciter folder present under --data-root")
    p.add_argument("--skip-surah", type=int, nargs="+", default=[],
                   help="Leave surahs out, e.g. 1 for Waleed, whose al-Fatiha labels are shifted by one ayah")
    p.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--latency-runs", type=int, default=10)
    p.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "benchmark")
    return p.parse_args()


def skeleton(text):
    """Spelling-insensitive form: Uthmani vs standard orthography (بهاذا/بهذا, ءامنوا/امنوا) compares equal."""
    words = []
    for word in text.split():
        word = re.sub(r"[ءاأإآ]", "", word).replace("ى", "ي").replace("ة", "ه")
        words.append(re.sub(r"(.)\1+", r"\1", word) or "ا")
    return " ".join(words)


def load_model(path, dtype):
    processor = WhisperProcessor.from_pretrained(path)
    model = WhisperForConditionalGeneration.from_pretrained(path, torch_dtype=dtype).to("cuda").eval()
    if not hasattr(model.generation_config, "lang_to_id"):
        c = model.config
        base = OPENAI_BASES[(c.d_model, c.encoder_layers, c.decoder_layers)]
        model.generation_config = GenerationConfig.from_pretrained(base)
    model.config.use_cache = True
    return processor, model


def transcribe(model, processor, signals, native, dtype):
    features = processor.feature_extractor(signals, sampling_rate=16000, return_tensors="pt").input_features
    prefix = {} if native else {"language": "arabic", "task": "transcribe"}
    with torch.inference_mode():
        ids = model.generate(input_features=features.to("cuda", dtype=dtype), num_beams=1,
                             do_sample=False, max_new_tokens=200, **prefix)
    return processor.batch_decode(ids, skip_special_tokens=True)


def window_latency(model, processor, rows, native, dtype, runs):
    """Median time to decode one 28 s streaming window, as the live server does."""
    audio = np.concatenate([sf.read(row["path"], dtype="float32")[0] for row in rows[:40]])
    window = audio[: 16000 * WINDOW_SECONDS]
    for _ in range(3):
        transcribe(model, processor, [window], native, dtype)
    timings = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        transcribe(model, processor, [window], native, dtype)
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
    return statistics.median(timings)


def benchmark(name, path, native, rows, args, dtype):
    torch.cuda.reset_peak_memory_stats()
    processor, model = load_model(path, dtype)
    started = time.perf_counter()
    details = []
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset:offset + args.batch_size]
        signals = [sf.read(row["path"], dtype="float32")[0] for row in batch]
        for row, text in zip(batch, transcribe(model, processor, signals, native, dtype)):
            details.append({"surah": row["surah"], "ayah": row["ayah"], "reciter": row["reciter_key"],
                            "reference": normalize_quran_for_asr(row["text_asr_normalized"]),
                            "prediction": normalize_quran_for_asr(text), "raw_prediction": text})
    report = {"model": name, "path": str(path), "prefix": "native" if native else "arabic",
              "dtype": args.dtype, "split": args.split, "overall": _scores(details),
              "spelling_insensitive_wer": jiwer.wer([skeleton(r["reference"]) for r in details],
                                                    [skeleton(r["prediction"]) for r in details]),
              "seconds": round(time.perf_counter() - started, 1)}
    for reciter in sorted({row["reciter"] for row in details}):
        report[reciter] = _scores([row for row in details if row["reciter"] == reciter])
    report["window_latency_ms"] = round(window_latency(model, processor, rows, native, dtype, args.latency_runs))
    report["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    del model
    torch.cuda.empty_cache()
    return report, details


def main():
    args = parse_args()
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    reciters = args.reciters or sorted(
        key for key, folder in __import__("qaloon_data").RECITER_DIRS.items() if (args.data_root / folder).is_dir())
    splits, _ = load_splits(args.data_root, include=reciters)
    rows = [row for part in splits.values() for row in part] if args.split == "all" else splits[args.split]
    rows = [row for row in rows if row["surah"] not in args.skip_surah]
    print(f"{args.split}: {len(rows)} clips from {reciters}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for spec in args.model:
        name, _, path = spec.partition("=")
        native = path.endswith("@native")
        path = path.removesuffix("@native")
        print(f"\n=== {name} ({path})", flush=True)
        report, details = benchmark(name, path, native, rows, args, dtype)
        with (args.output_dir / f"{name}_predictions.jsonl").open("w", encoding="utf-8") as handle:
            handle.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in details)
        (args.output_dir / f"{name}_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        summary.append(report)
        o = report["overall"]
        print(f"WER {o['wer']:.2%}  CER {o['cer']:.2%}  window {report['window_latency_ms']} ms  "
              f"VRAM {report['peak_vram_gb']} GB  ({report['seconds']} s)", flush=True)

    per_reciter = sorted({key for r in summary for key in r if isinstance(r[key], dict) and key != "overall"})
    lines = ["| Model | WER | Spelling-insensitive WER | CER | " + " | ".join(f"{k} WER" for k in per_reciter) + " | 28 s window | VRAM |",
             "|---" * (6 + len(per_reciter)) + "|"]
    for r in sorted(summary, key=lambda r: r["overall"]["wer"]):
        cells = [f"{r[k]['wer']:.1%}" if k in r else "-" for k in per_reciter]
        lines.append(f"| {r['model']} | {r['overall']['wer']:.2%} | {r['spelling_insensitive_wer']:.2%} | {r['overall']['cer']:.2%} | "
                     + " | ".join(cells) + f" | {r['window_latency_ms']} ms | {r['peak_vram_gb']} GB |")
    table = "\n".join(lines)
    (args.output_dir / "summary.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)


if __name__ == "__main__":
    main()
