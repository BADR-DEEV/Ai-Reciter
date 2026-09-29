"""Evaluate expert-labelled tajweed/pronunciation events (not ASR WER)."""

import argparse
import json
from pathlib import Path


def match_events(expected, predicted, min_iou=0.3):
    """One-to-one match, same type, with overlapping time interval."""
    used = set()
    correct = 0
    for event in expected:
        best, best_iou = None, min_iou
        for index, guess in enumerate(predicted):
            if index in used or event["type"] != guess["type"]:
                continue
            overlap = max(0, min(event["end_sec"], guess["end_sec"])
                          - max(event["start_sec"], guess["start_sec"]))
            union = max(event["end_sec"], guess["end_sec"]) - min(event["start_sec"], guess["start_sec"])
            iou = overlap / union if union > 0 else 0
            if iou >= best_iou:
                best, best_iou = index, iou
        if best is not None:
            used.add(best)
            correct += 1
    return correct


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    args = p.parse_args()
    annotated = [json.loads(line) for line in args.annotations.open(encoding="utf-8")]
    predicted = {row["audio_path"]: row for row in (
        json.loads(line) for line in args.predictions.open(encoding="utf-8"))}
    tp = total_expected = total_predicted = false_alarm_clips = clean_clips = 0
    for row in annotated:
        if row["errors"] is None or not row["reviewer"]:
            raise ValueError(f"Unreviewed audio: {row['audio_path']}")
        if row["audio_path"] not in predicted:
            raise ValueError(f"Missing prediction: {row['audio_path']}")
        expected, guesses = row["errors"], predicted[row["audio_path"]]["errors"]
        for event in [*expected, *guesses]:
            if event["type"] not in {"tajweed", "makhraj"} or not 0 <= event["start_sec"] < event["end_sec"] <= row["duration_seconds"]:
                raise ValueError(f"Invalid event in {row['audio_path']}: {event}")
        tp += match_events(expected, guesses)
        total_expected += len(expected)
        total_predicted += len(guesses)
        if not expected:
            clean_clips += 1
            false_alarm_clips += bool(guesses)
    precision = tp / total_predicted if total_predicted else 0.0
    recall = tp / total_expected if total_expected else 0.0
    print(json.dumps({"precision": precision, "recall": recall,
                      "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                      "false_alarm_rate_on_clean_clips": false_alarm_clips / clean_clips if clean_clips else None,
                      "reviewed_clips": len(annotated)}, indent=2))


if __name__ == "__main__":
    main()
