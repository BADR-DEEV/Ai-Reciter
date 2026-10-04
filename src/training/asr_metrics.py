"""ASR diagnostics and clustered uncertainty; never a tajweed correctness score."""
from collections import defaultdict
import numpy as np


def error_counts(reference, prediction):
    import jiwer
    alignment = jiwer.process_words(reference, prediction)
    return {"words": len(reference.split()), "hits": alignment.hits, "substitutions": alignment.substitutions,
            "deletions": alignment.deletions, "insertions": alignment.insertions}


def scores(details):
    import jiwer
    if not details:
        raise ValueError("No predictions to score")
    refs, hyps = [r["reference"] for r in details], [r["prediction"] for r in details]
    counts = [error_counts(r, h) for r, h in zip(refs, hyps)]
    words = sum(c["words"] for c in counts)
    denominator = max(1, words)
    return {"samples": len(details), "reference_words": words, "wer": jiwer.wer(refs, hyps), "cer": jiwer.cer(refs, hyps),
        "exact_ayah_accuracy": sum(r == h for r, h in zip(refs, hyps)) / len(details),
        "word_substitutions": sum(c["substitutions"] for c in counts), "word_deletions": sum(c["deletions"] for c in counts),
        "word_insertions": sum(c["insertions"] for c in counts),
        "substitution_rate": sum(c["substitutions"] for c in counts) / denominator,
        "deletion_rate": sum(c["deletions"] for c in counts) / denominator,
        "insertion_rate": sum(c["insertions"] for c in counts) / denominator,
        "apparent_omission_ayah_rate": sum(c["deletions"] > 0 for c in counts) / len(details),
        "empty_prediction_rate": sum(not h for h in hyps) / len(details),
        "extra_word_ayah_rate": sum(c["insertions"] > 0 for c in counts) / len(details),
        "terminal_word_mismatch_rate": sum(not h.split() or r.split()[-1:] != h.split()[-1:] for r, h in zip(refs, hyps)) / len(details),
        "warning": "Recognition errors on reference audio are NOT proven recitation omissions or tajweed errors."}


def clustered_wer_interval(details, seed=42, repetitions=500, group="surah"):
    clusters = defaultdict(list)
    for row in details:
        clusters[row[group]].append(row)
    totals = []
    for rows in clusters.values():
        counts = [error_counts(r["reference"], r["prediction"]) for r in rows]
        totals.append([sum(c["substitutions"] + c["deletions"] + c["insertions"] for c in counts), sum(c["words"] for c in counts)])
    if len(totals) < 2:
        return {"available": False, "reason": "Need >=2 independent groups", "groups": len(totals)}
    rng, values = np.random.default_rng(seed), np.asarray(totals)
    samples = values[rng.integers(0, len(values), size=(repetitions, len(values)))].sum(axis=1)
    rates = samples[:, 0] / np.maximum(1, samples[:, 1])
    low, high = np.quantile(rates, [.025, .975])
    return {"available": True, "low": float(low), "high": float(high), "groups": len(totals), "group": group,
            "method": "95% percentile cluster bootstrap", "repetitions": repetitions, "seed": seed,
            "warning": "Few held-out recordings give unstable intervals; upstream pretraining overlap is not removed."}


def report_predictions(details, seed=42):
    readers = {reader: scores([r for r in details if r["reciter"] == reader]) for reader in sorted({r["reciter"] for r in details})}
    report = {"overall": scores(details), "per_reciter": readers, "wer_interval": clustered_wer_interval(details, seed),
        "macro_reciter_wer": float(np.mean([v["wer"] for v in readers.values()])),
        "worst_reciter_wer": max(v["wer"] for v in readers.values()),
        "reciter_wer_gap": max(v["wer"] for v in readers.values()) - min(v["wer"] for v in readers.values())}
    for name, subset in (("seen_text", [r for r in details if r.get("text_seen_in_training") is True]),
                         ("unseen_text", [r for r in details if r.get("text_seen_in_training") is False]),
                         ("short_le_5s", [r for r in details if r.get("duration_seconds", 0) <= 5]),
                         ("medium_5_to_15s", [r for r in details if 5 < r.get("duration_seconds", 0) <= 15]),
                         ("long_gt_15s", [r for r in details if r.get("duration_seconds", 0) > 15])):
        if subset:
            report[name] = scores(subset)
    return report


def paired_comparison(baseline, candidate, seed=42, repetitions=500):
    key = lambda r: (r["reciter"], r["surah"], r["ayah"])
    left, right = {key(r): r for r in baseline}, {key(r): r for r in candidate}
    if len(left) != len(baseline) or len(right) != len(candidate) or left.keys() != right.keys():
        raise ValueError("Paired comparison requires identical unique evaluation samples")
    totals = defaultdict(lambda: np.zeros(3))
    for identity in left:
        b, c = left[identity], right[identity]
        if (b["reference"] != c["reference"] or not b.get("audio_sha256")
                or b.get("audio_sha256") != c.get("audio_sha256")):
            raise ValueError("Paired labels/audio differ")
        bc, cc = error_counts(b["reference"], b["prediction"]), error_counts(c["reference"], c["prediction"])
        totals[b["surah"]] += [sum(bc[k] for k in ("substitutions", "deletions", "insertions")),
                              sum(cc[k] for k in ("substitutions", "deletions", "insertions")), bc["words"]]
    values, rng = np.asarray(list(totals.values())), np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(repetitions, len(values)))].sum(axis=1)
    delta = (samples[:, 1] - samples[:, 0]) / np.maximum(1, samples[:, 2])
    result = {"samples": len(candidate), "candidate_minus_baseline_wer": scores(candidate)["wer"] - scores(baseline)["wer"],
        "lower_is_better": True, "same_audio_labels_verified": True}
    if len(values) >= 2:
        result["paired_surah_bootstrap_95_interval"] = [float(x) for x in np.quantile(delta, [.025, .975])]
    return result
