"""Create V2 review manifest; expert must annotate errors, including clean clips."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from qaloon_data import DATA_ROOT, load_splits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    splits, _ = load_splits(args.data_root, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for split, rows in splits.items():
            for row in rows:
                handle.write(json.dumps({
                    "split": split, "surah": row["surah"], "ayah": row["ayah"],
                    "reciter": row["reciter_key"], "audio_path": row["path"],
                    "transcript_with_harakat": row["normalized_with_harakat"],
                    "duration_seconds": row["duration_seconds"],
                    "errors": None,  # An expert must set [] for verified clean recitation.
                    "reviewer": None,
                }, ensure_ascii=False) + "\n")
    print(f"Expert annotation manifest: {args.output}")


if __name__ == "__main__":
    main()
