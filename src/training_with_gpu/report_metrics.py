"""Audit saved ASR errors with corpus WER/CER, per-reader scores and cluster CIs.

This recomputes saved predictions, NOT a new unseen-reader evaluation.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import jiwer


def summarize(rows, bootstrap=1000, cluster="ayah", seed=42):
    if not rows:
        raise ValueError("No saved predictions")
    totals = np.zeros(6)
    grouped = defaultdict(lambda: np.zeros(6))
    readers = defaultdict(list)
    for row in rows:
        ref, hyp = row["reference"], row["prediction"]
        word = jiwer.process_words(ref, hyp)
        char = jiwer.process_characters(ref, hyp)
        counts = np.array([word.substitutions, word.deletions, word.insertions,
                           word.hits + word.substitutions + word.deletions,
                           char.substitutions + char.deletions + char.insertions,
                           char.hits + char.substitutions + char.deletions], dtype=float)
        totals += counts
        group = row["reciter"] if cluster == "reader" else (row["surah"], row["ayah"])
        grouped[group] += counts
        readers[row["reciter"]].append(row)
    metrics = {"samples": len(rows), "wer": float(totals[:3].sum() / totals[3]), "cer": float(totals[4] / totals[5]),
               "word_substitutions": int(totals[0]), "word_deletions": int(totals[1]), "word_insertions": int(totals[2]), "reference_words": int(totals[3]),
               "per_reader": {reader: {"samples": len(r), "wer": jiwer.wer([x["reference"] for x in r], [x["prediction"] for x in r]),
                                      "cer": jiwer.cer([x["reference"] for x in r], [x["prediction"] for x in r])} for reader, r in readers.items()}}
    if bootstrap:
        matrix = np.stack(list(grouped.values()))
        rng = np.random.default_rng(seed)
        resampled = matrix[rng.integers(0, len(matrix), size=(bootstrap, len(matrix)))].sum(axis=1)
        wers = resampled[:, :3].sum(axis=1) / np.maximum(1, resampled[:, 3])
        cers = resampled[:, 4] / np.maximum(1, resampled[:, 5])
        metrics["cluster_bootstrap"] = {"unit": cluster, "groups": len(matrix), "replicates": bootstrap, "seed": seed,
            "wer_95_interval": np.quantile(wers, [.025, .975]).tolist(), "cer_95_interval": np.quantile(cers, [.025, .975]).tolist(),
            "warning": "Descriptive interval for this saved split; does not establish unseen-reader generalization. Four expert voices are insufficient for a strong population estimate."}
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--cluster", choices=["ayah", "reader"], default="ayah")
    args = parser.parse_args()
    if not 0 <= args.bootstrap <= 10000:
        parser.error("bootstrap must be 0–10000")
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = {"source": str(args.predictions), "metric_units": "fractions", "fresh_evaluation": False, **summarize(rows, args.bootstrap, args.cluster)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("samples", "wer", "cer", "word_substitutions", "word_deletions", "word_insertions")}, indent=2))
