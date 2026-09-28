"""Compare untouched Whisper tiny/base on the same held-out Qaloon ayahs."""

import argparse
import json
from pathlib import Path

from qaloon_data import DATA_ROOT, RECITER_DIRS, evaluate_model, load_splits, score_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("qaloon_zero_shot_results"))
    parser.add_argument("--models", nargs="+", choices=["tiny", "base"], default=["tiny", "base"])
    parser.add_argument("--include-reciter", nargs="+", choices=RECITER_DIRS)
    parser.add_argument("--exclude-reciter", nargs="+", choices=RECITER_DIRS)
    parser.add_argument("--include-bismillah", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    splits, skipped = load_splits(
        args.data_root, args.include_reciter, args.exclude_reciter,
        args.seed, include_bismillah=args.include_bismillah,
    )
    rows = splits[args.split]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"split": args.split, "seed": args.seed, "samples": len(rows),
               "skipped_over_30s": skipped, "models": {}}
    for size in args.models:
        model_id = f"openai/whisper-{size}"
        processor = WhisperProcessor.from_pretrained(model_id, language="arabic", task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained(model_id)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        overall, details = evaluate_model(model, processor, rows, args.batch_size)
        per_reciter = {}
        for reciter in sorted({row["reciter"] for row in details}):
            group = [row for row in details if row["reciter"] == reciter]
            per_reciter[reciter] = {"samples": len(group), **score_predictions(
                [row["reference"] for row in group],
                [row["prediction"] for row in group],
            )}
        summary["models"][size] = {"overall": overall, "per_reciter": per_reciter}
        with (args.output_dir / f"whisper_{size}_{args.split}_predictions.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in details:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(model_id, overall, per_reciter)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    (args.output_dir / "zero_shot_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
