"""Print actual saved CPU-run metrics and per-reciter score, no new model run."""

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "runs"


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def print_by_reciter(path):
    samples = defaultdict(list)
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        samples[row["reciter"]].append(row)
    for reciter, rows in sorted(samples.items()):
        wer = sum(distance(r["reference"].split(), r["prediction"].split()) for r in rows)
        words = sum(len(r["reference"].split()) for r in rows)
        cer = sum(distance(r["reference"], r["prediction"]) for r in rows)
        chars = sum(len(r["reference"]) for r in rows)
        print(f"  {reciter:10s} samples={len(rows):3d} WER={wer/words:.2%} CER={cer/chars:.2%}")


def main():
    for label, filename in [
        ("Zero-shot tiny, test", ROOT / "zero_shot" / "whisper_tiny_test_predictions.jsonl"),
        ("Zero-shot base, test", ROOT / "zero_shot" / "whisper_base_test_predictions.jsonl"),
        ("Tiny LoRA, validation", ROOT / "tiny_lora" / "validation_predictions.jsonl"),
        ("Tiny LoRA, test", ROOT / "tiny_lora" / "test_predictions.jsonl"),
    ]:
        if filename.exists():
            print(label)
            print_by_reciter(filename)


if __name__ == "__main__":
    main()
