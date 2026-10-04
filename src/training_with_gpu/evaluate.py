"""Frozen blind/assisted private benchmark CLI; historical helper retained."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from qaloon_data import DATA_ROOT, RECITER_DIRS, load_splits
from gpu_evaluation import evaluate_rows


def legacy_main():
    """Historical evaluator retained for inspection, not the active CLI."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["tiny", "base"], default="base")
    p.add_argument("--adapter", type=Path, help="LoRA directory; omit for a zero-shot baseline")
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--include-reciter", nargs="+", choices=RECITER_DIRS)
    p.add_argument("--exclude-reciter", nargs="+", choices=RECITER_DIRS)
    p.add_argument("--label-field", choices=["text_asr_normalized", "normalized_with_harakat"],
                   default="text_asr_normalized")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=1)
    args = p.parse_args()

    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    splits, skipped = load_splits(args.data_root, args.include_reciter,
                                   args.exclude_reciter, args.seed)
    model_id = f"openai/whisper-{args.model}"
    processor = WhisperProcessor.from_pretrained(args.adapter or model_id,
                                                 language="arabic", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report, details = evaluate_rows(model, processor, splits["test"],
                                    args.batch_size, args.label_field)
    report.update({"model": model_id, "adapter": str(args.adapter) if args.adapter else None,
                   "seed": args.seed, "excluded": skipped})
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for detail in details:
            handle.write(json.dumps(detail, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    """Evaluate frozen manifests and the actual pinned adapter/base, not OpenAI."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.training_with_gpu.benchmark_reciter import main as benchmark
    benchmark()


if __name__ == "__main__":
    main()
