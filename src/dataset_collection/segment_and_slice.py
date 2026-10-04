"""Review-first Qaloon segmentation with explicit omissions and leakage gates.

Full ordered reference evidence + monotonic non-overlapping occurrences replace
greedy terminal-word searches. Word timestamps bound waveform refinement; a
blind per-clip ASR pass rejects extra/missing speech. No acoustic/madd guarantees
are claimed. Outputs remain unapproved candidates, never auto-trusted training.
Existing audio/metadata is never overwritten. Use a new output directory.
"""
import argparse
from difflib import SequenceMatcher
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.dataset_collection.qaloon_audio2text import load_quran, normalize_quran_for_asr, normalize_with_harakat
from src.dataset_collection.quran_geometry import geometry_fields

VERSION = "terminal-protected-coverage-v4"


def word_similarity(left, right):
    if left == right:
        return 1.0
    # ASR sometimes writes final long /a:/ with ا rather than canonical ى
    # (سعا/سعى, قلا/قلى). Comparison only: never rewrite Qaloon labels, fold
    # medial alif (ملك/مالك), or treat a true final ي as this vowel.
    if len(left) >= 3 and len(left) == len(right) and left[:-1] == right[:-1] and {left[-1], right[-1]} == {"ى", "ا"}:
        return 1.0
    if min(len(left), len(right)) <= 3:
        return 0.0
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def prepare_words(words, duration):
    result, rejected = [], []
    for group, word in enumerate(words):
        start, end = word.get("start"), word.get("end")
        if (not isinstance(start, (int, float)) or not isinstance(end, (int, float))
                or not math.isfinite(start) or not math.isfinite(end)
                or start < 0 or end <= start or end > duration + .05
                or (result and start < result[-1]["start"])):
            rejected.append(group)
            continue
        text = word.get("clean_word", word.get("word", word.get("raw_word", "")))
        for token in normalize_quran_for_asr(text).split():
            result.append({"clean_word": token, "start": float(start), "end": min(float(end), duration), "group": group})
    return result, rejected


def edit_alignment(expected, observed, threshold=.84):
    """Global token edit alignment with ordered matched index pairs."""
    m, n = len(expected), len(observed)
    costs = [[0.0] * (n + 1) for _ in range(m + 1)]
    ops = [[""] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        costs[i][0], ops[i][0] = float(i), "delete"
    for j in range(1, n + 1):
        costs[0][j], ops[0][j] = float(j), "insert"
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            score = word_similarity(expected[i - 1], observed[j - 1])
            substitution = (1 - score) / 2 if score >= threshold else 1.0
            choices = [(costs[i - 1][j - 1] + substitution, "pair"),
                       (costs[i - 1][j] + 1, "delete"), (costs[i][j - 1] + 1, "insert")]
            costs[i][j], ops[i][j] = min(choices, key=lambda item: item[0])
    i, j = m, n
    pairs = []
    while i or j:
        operation = ops[i][j]
        if operation == "pair":
            score = word_similarity(expected[i - 1], observed[j - 1])
            if score >= threshold:
                pairs.append((i - 1, j - 1, score))
            i -= 1
            j -= 1
        elif operation == "delete":
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return costs[m][n], pairs


def candidates_for_reference(words, reference, min_coverage=.9, min_precision=.9):
    expected = normalize_quran_for_asr(reference["text_asr"]).split()
    if not expected:
        return []
    m = len(expected)
    heard = [word["clean_word"] for word in words]
    budget = max(1, min(8, math.ceil(m * .15)))
    candidates = []
    for first, token in enumerate(heard):
        if word_similarity(expected[0], token) < .9:
            continue
        if m > 1 and not any(word_similarity(expected[1], t) >= .9 for t in heard[first + 1:first + 3]):
            continue
        for last in range(first + max(1, m - budget) - 1, min(len(words), first + m + budget)):
            if word_similarity(expected[-1], heard[last]) < .9:
                continue
            observed = heard[first:last + 1]
            cost, pairs = edit_alignment(expected, observed)
            coverage, precision = len(pairs) / m, len(pairs) / len(observed)
            if coverage < min_coverage or precision < min_precision:
                continue
            matched = {ref_index: (observed_index, score) for ref_index, observed_index, score in pairs}
            if matched.get(0, (-1, 0))[0] != 0 or matched.get(m - 1, (-1, 0))[0] != len(observed) - 1:
                continue
            if m <= 4 and (len(pairs) != m or len(observed) != m):
                continue
            if m >= 4 and (matched.get(1, (-1, 0))[0] != 1
                    or matched.get(m - 2, (-1, 0))[0] != len(observed) - 2):
                continue
            quality = sum(pair[2] for pair in pairs) / m
            flags = []
            if first and words[first - 1]["end"] > words[first]["start"] + .005:
                flags.append("overlapping-start-timestamps")
            if last + 1 < len(words) and words[last + 1]["start"] < words[last]["end"] - .005:
                flags.append("overlapping-end-timestamps")
            if first and words[first - 1]["group"] == words[first]["group"]:
                flags.append("shared-token-group-at-start")
            if last + 1 < len(words) and words[last + 1]["group"] == words[last]["group"]:
                flags.append("shared-token-group-at-end")
            candidates.append({"data": reference, "ayah": reference["ayah"], "start": words[first]["start"],
                "end": words[last]["end"], "word_start": first, "word_end": last + 1,
                "first_word_end": words[first]["end"], "last_word_start": words[last]["start"],
                "next_word_end": words[last + 1]["end"] if last + 1 < len(words) else None,
                "coverage": coverage, "precision": precision, "quality": quality, "edit_cost": cost,
                "matched_words": len(pairs), "observed_text": " ".join(observed), "flags": flags,
                "speech_wall": words[last + 1]["start"] if last + 1 < len(words) else None,
                "alignment_status": "proposal-needs-review", "usable_for_training": False})
    unique = {}
    for candidate in candidates:
        key = candidate["word_start"]
        rank = (candidate["quality"] + candidate["precision"], -candidate["edit_cost"], -candidate["word_end"])
        prior = unique.get(key)
        if prior is None or rank > prior[0]:
            unique[key] = (rank, candidate)
    return [item[1] for item in sorted(unique.values(), key=lambda item: item[1]["word_start"])]


def align_references(raw_words, references, duration, beam_size=128):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Audio duration must be finite and positive")
    canonical = sorted((r for r in references if r["ayah"] >= 1), key=lambda r: r["ayah"])
    if len({r["ayah"] for r in canonical}) != len(canonical):
        raise ValueError("Duplicate canonical ayah IDs")
    words, invalid_words = prepare_words(raw_words, duration)
    pools = {r["ayah"]: candidates_for_reference(words, r) for r in canonical}
    safe_evidence = [(number, c) for number, pool in pools.items() for c in pool if not c["flags"]]
    for number, pool in pools.items():
        for candidate in pool:
            if any(other_number > number and candidate["word_start"] < other["word_start"]
                   and other["word_end"] <= candidate["word_end"] for other_number, other in safe_evidence):
                candidate["flags"].append("contains-other-reference")
    first_ayah_start = min((c["word_start"] for c in pools.get(1, []) if not c["flags"]), default=len(words))
    preambles = []
    for reference in references:
        if reference["ayah"] < 0:
            for candidate in candidates_for_reference(words, reference, 1.0, 1.0):
                if not candidate["flags"] and candidate["word_start"] < 24 and candidate["word_end"] <= first_ayah_start:
                    preambles.append(candidate)
    preamble_wall = max((c["end"] for c in preambles), default=0.0)
    states = [(0.0, 0, ())]
    for reference in canonical:
        candidates = [c for c in pools[reference["ayah"]] if not c["flags"] and c["start"] >= preamble_wall]
        next_states = list(states)
        for score, end_index, chosen in states:
            for candidate in candidates:
                if candidate["word_start"] < end_index or (chosen and candidate["start"] < chosen[-1]["end"]):
                    continue
                gain = candidate["matched_words"] + 2 + candidate["quality"] - candidate["edit_cost"]
                gap_penalty = min(1.0, (candidate["word_start"] - end_index) * .005)
                next_states.append((score + gain - gap_penalty, candidate["word_end"], chosen + (candidate,)))
        next_states.sort(key=lambda s: (-s[0], s[1], tuple(c["word_start"] for c in s[2])))
        unique = {}
        for state in next_states:
            key = (state[1], tuple(c["ayah"] for c in state[2]))
            unique.setdefault(key, state)
        states = list(unique.values())[:beam_size]
    selected = states[0][2] if states else ()
    bounds = {c["ayah"]: c for c in selected}
    report = {"version": VERSION, "invalid_word_indices": invalid_words,
        "selected_candidates": [{k: c[k] for k in ("ayah", "start", "end", "word_start", "word_end", "coverage", "precision", "observed_text", "speech_wall")} for c in bounds.values()],
        "preambles": [{k: c[k] for k in ("ayah", "start", "end", "observed_text")} for c in preambles],
        "ayahs": [{"ayah": r["ayah"], "status": "matched-candidate" if r["ayah"] in bounds else "needs-review",
                   "candidate_count": len(pools[r["ayah"]]),
                   "reason": None if r["ayah"] in bounds else (
                       "no-full-text-match" if not pools[r["ayah"]] else
                       "unsafe-timestamps-or-fusion" if all(c["flags"] for c in pools[r["ayah"]]) else "monotonic-path-conflict"),
                   "occurrences": [{k: c[k] for k in ("start", "end", "word_start", "word_end", "coverage", "precision", "flags")} for c in pools[r["ayah"]]]}
                  for r in canonical],
        "unmatched_ayahs": [r["ayah"] for r in canonical if r["ayah"] not in bounds],
        "usable_for_training": False}
    return bounds, preamble_wall, report


def validate_clip_transcript(text, reference, later_references=()):
    expected = normalize_quran_for_asr(reference).split()
    observed = normalize_quran_for_asr(text).split()
    if not expected or not observed:
        return {"passed": False, "reason": "empty-clip-transcript", "coverage": 0, "precision": 0}
    _, pairs = edit_alignment(expected, observed)
    coverage, precision = len(pairs) / len(expected), len(pairs) / len(observed)
    contains = []
    for later in later_references:
        following = normalize_quran_for_asr(later["text_asr"]).split()
        if following and len(following) <= len(observed) and " ".join(following) not in " ".join(expected):
            if any(observed[i:i + len(following)] == following for i in range(len(observed) - len(following) + 1)):
                contains.append(later["ayah"])
    anchored = pairs and pairs[0][:2] == (0, 0) and pairs[-1][:2] == (len(expected) - 1, len(observed) - 1)
    passed = bool(anchored and coverage >= .95 and precision >= .95 and not contains)
    return {"passed": passed, "reason": None if passed else "clip-text-incomplete-or-extra-speech",
            "coverage": coverage, "precision": precision, "contains_later_ayahs": contains, "observed_text": " ".join(observed),
            "comparison_only_spelling_variants": [{"reference": expected[i], "observed": observed[j]}
                for i, j, score in pairs if score == 1 and expected[i] != observed[j]]}


def exact_text_check(expected, observed):
    reference = normalize_quran_for_asr(expected).split()
    heard = normalize_quran_for_asr(observed).split()
    operations = []
    for tag, i, j, a, b in SequenceMatcher(None, reference, heard, autojunk=False).get_opcodes():
        operations.append({"operation": tag, "reference_start": i, "reference_end": j,
                           "heard_start": a, "heard_end": b, "expected": reference[i:j], "heard": heard[a:b]})
    matched = sum(o["reference_end"] - o["reference_start"] for o in operations if o["operation"] == "equal")
    return {"passed": bool(reference and reference == heard), "expected": " ".join(reference),
            "observed": " ".join(heard), "expected_words": len(reference), "observed_words": len(heard),
            "exact_matched_words": matched, "word_operations": operations,
            "comparison": "exact-normalized-tokens-no-fuzzy-or-final-alif-folding"}


def frame_energy(audio, sr, frame_seconds=.02, hop_seconds=.005):
    frame, hop = max(1, round(frame_seconds * sr)), max(1, round(hop_seconds * sr))
    if len(audio) < frame:
        return np.array([], dtype=int), np.array([], dtype=float), frame
    positions = np.arange(0, len(audio) - frame + 1, hop)
    cumulative = np.concatenate(([0.0], np.cumsum(np.square(audio, dtype=np.float64))))
    rms = np.sqrt(np.maximum(0, cumulative[positions + frame] - cumulative[positions]) / frame)
    return positions, rms, frame


def true_runs(mask):
    edges = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def propose_pause_boundary(audio, sr, terminal_start, nominal_end, next_start=None,
                           next_word_end=None, search_seconds=1.75):
    """Joint left-END/right-START proposal, never a one-sided tail extension.

    Whisper's next-word onset is an uncertain estimate, not an acoustic wall.
    Search only the two edge words for a sustained low-energy gap. A vowel in
    continuous wasl has no identifiable silence cut: fail closed for review.
    This cannot certify phoneme/madd duration or distinguish every internal waqf.
    """
    duration = len(audio) / sr
    center = nominal_end if next_start is None else (nominal_end + next_start) / 2
    # An internal waqf BEFORE the recognizer's terminal-word end is not an
    # ayah boundary. This caused 79:16's طوى to be assigned to the next clip.
    # Detect the WHOLE pause, even if a late terminal estimate lands inside it.
    # The CUT (not the analysis window) stays protected by the terminal end.
    # Otherwise a genuine pause touching the analysis edge is lost entirely.
    minimum_cut = max(0, nominal_end - .06)
    lower = max(0, terminal_start + .12, center - search_seconds)
    upper = min(duration, center + search_seconds)
    if next_word_end is not None:
        upper = min(upper, next_word_end - .08)
    if upper - lower < .14:
        return {"resolved": False, "reason": "insufficient-edge-word-window"}
    offset = math.floor(lower * sr)
    chunk = audio[offset:math.floor(upper * sr)]
    positions, rms, frame = frame_energy(chunk, sr)
    # After protecting the terminal end, the search itself may be mostly EOF
    # silence. Calibrate voiced energy from the actual terminal-word region,
    # not a 90th percentile dominated by the ensuing silence.
    calibration = audio[math.floor(max(0, terminal_start) * sr):math.floor(min(duration, nominal_end + .05) * sr)]
    _, terminal_rms, _ = frame_energy(calibration, sr)
    voiced = max(float(np.quantile(rms, .9)) if len(rms) else 0,
                 float(np.quantile(terminal_rms, .9)) if len(terminal_rms) else 0)
    if not len(rms) or voiced < 1e-6:
        return {"resolved": False, "reason": "no-voiced-boundary-evidence"}
    floor = float(np.quantile(rms, .1))
    # A noisy or continuous voiced window must NOT be declared all-silence.
    threshold = min(max(1e-7, floor * 1.8, voiced * .015), voiced * .10)
    gaps = []
    for first, stop in true_runs(rms <= threshold):
        start = (offset + positions[first]) / sr
        end = (offset + positions[stop - 1] + frame) / sr
        if end - start < .10:
            continue
        if end < minimum_cut + .02:
            continue  # internal waqf before the terminal word is never admitted
        # Ignore gaps touching either search edge: evidence must include speech
        # before the decay; with a following word also require resumed speech.
        if first == 0 or (next_start is not None and stop == len(rms)):
            continue
        gaps.append((abs((start + end) / 2 - center), start, end))
    if not gaps:
        return {"resolved": False, "reason": "no-measured-pause-continuous-or-noisy-join",
                "search_start": lower, "search_end": upper}
    # Prefer a sustained pause to a tiny low-energy dip inside a madd/word.
    # Distance alone can select an internal breath/fricative before the tail.
    _, pause_start, pause_end = min(gaps, key=lambda gap: (-(gap[2] - gap[1]) + .15 * gap[0], gap[0]))
    # Keep the low-energy release as well: an arbitrary first-threshold crossing
    # can be the soft end of a vowel. Cut near the far end of a SHORT pause, while
    # avoiding excessive final-file silence and retaining onset headroom.
    # A shared midpoint preserves both soft vowel release AND low-energy word
    # onsets. Cutting the right clip at the end of a quiet run can lose a weak
    # first consonant; advancing only the left end can duplicate/leak the tail.
    midpoint = max((pause_start + pause_end) / 2, minimum_cut)
    left_end = midpoint if next_start is not None else max(pause_start + .08, minimum_cut)
    right_start = midpoint
    return {"resolved": True, "method": "joint-measured-pause", "left_end": left_end,
            "right_start": right_start, "pause_start": pause_start, "pause_end": pause_end,
            "shift_seconds": left_end - nominal_end, "needs_listening_review": True}


def refine_ayah_boundaries(audio, sr, bounds, preamble_wall, search_seconds=1.75):
    adjusted = {number: {**candidate, "flags": list(candidate.get("flags", []))} for number, candidate in bounds.items()}
    ordered = sorted(adjusted, key=lambda number: adjusted[number]["start"])
    events = []
    # Also keep the tail of the preamble out of the first ayah.
    if ordered and preamble_wall > 0:
        first = adjusted[ordered[0]]
        event = propose_pause_boundary(audio, sr, max(0, preamble_wall - .6), preamble_wall,
                                       first["start"], first.get("first_word_end"), search_seconds)
        events.append({"after_ayah": "preamble", **event})
        if event["resolved"]:
            preamble_wall = event["left_end"]
            first["start"] = event["right_start"]
        else:
            first["flags"].append("unresolved-preamble-tail")
    for number in ordered:
        candidate = adjusted[number]
        next_start = candidate.get("speech_wall")
        event = propose_pause_boundary(audio, sr, max(candidate["start"], candidate.get("last_word_start", candidate["start"])),
            candidate["end"], next_start, candidate.get("next_word_end"), search_seconds)
        events.append({"after_ayah": number, **event})
        neighbor = next((adjusted[n] for n in ordered if n != number and next_start is not None
                         and abs(adjusted[n]["start"] - next_start) < .025), None)
        if event["resolved"]:
            candidate["end"] = event["left_end"]
            candidate["acoustic_end"] = True
            if next_start is not None:
                candidate["speech_wall"] = event["right_start"]
            if neighbor is not None:
                neighbor["start"] = event["right_start"]
                neighbor["acoustic_start"] = True
                neighbor["previous_boundary_ayah"] = number
        else:
            candidate["flags"].append("unresolved-terminal-acoustic-boundary")
            if neighbor is not None:
                neighbor["flags"].append("unresolved-previous-madd-or-join")
    return adjusted, preamble_wall, events


def clean_audio(audio, sr, mode="conservative", hum_frequency=None, max_reduction_db=9.0):
    """Reversible, length-preserving DC/rumble + bounded stationary-noise reduction.

    No silence trimming, time stretching, AGC, compression or tail fades. Noise
    is learned only from sustained quiet regions, not arbitrary vowel frames.
    The original is always retained. This is an auditable DSP baseline, not a
    claim that aggressive enhancement meets a certified industrial standard.
    """
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all() or sr < 1000:
        raise ValueError("Cleaning requires finite nonempty mono audio and sr >= 1000")
    if mode not in {"none", "conservative"} or not 0 <= max_reduction_db <= 12:
        raise ValueError("Invalid cleaning mode or excessive denoising reduction")
    if hum_frequency not in {None, 50, 60}:
        raise ValueError("Hum frequency must be 50 or 60 Hz, or omitted")
    report = {"mode": mode, "input_frames": len(audio), "sample_rate": sr,
              "input_peak": float(np.max(np.abs(audio))),
              "clipped_fraction": float(np.mean(np.abs(audio) >= .999)),
              "max_reduction_db": max_reduction_db, "hum_frequency": hum_frequency,
              "noise_reduction_applied": False, "usable_for_training": False}
    if mode == "none":
        return audio.copy(), {**report, "output_peak": report["input_peak"], "output_frames": len(audio)}
    from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, stft, istft
    from scipy.ndimage import gaussian_filter
    signal = audio.astype(np.float64) - float(np.mean(audio))
    if len(signal) >= 128:
        signal = sosfiltfilt(butter(2, 40, btype="highpass", fs=sr, output="sos"), signal)
        if hum_frequency:
            b, a = iirnotch(hum_frequency, 35, fs=sr)
            signal = filtfilt(b, a, signal)
    positions, rms, frame = frame_energy(signal, sr, .032, .016)
    quiet_samples = []
    if len(rms) and np.quantile(rms, .9) > 1e-6:
        # Quiet profile runs must last >=350 ms; remove 64 ms at each edge to
        # avoid learning long-vowel release or a fricative as stationary noise.
        threshold = float(np.quantile(rms, .9)) * .065
        for first, stop in true_runs(rms < threshold):
            start, end = positions[first] + round(.064 * sr), positions[stop - 1] + frame - round(.064 * sr)
            if end - start >= round(.35 * sr):
                quiet_samples.append((start, end))
    nfft, hop = min(512, len(signal)), min(128, max(1, len(signal) // 4))
    if sum(end - start for start, end in quiet_samples) >= .5 * sr and nfft >= 128:
        _, times, spectrum = stft(signal, fs=sr, nperseg=nfft, noverlap=nfft - hop, boundary="zeros", padded=True)
        centers = times * sr
        noise_frames = np.zeros(len(times), dtype=bool)
        for start, end in quiet_samples:
            noise_frames |= (centers >= start + nfft / 2) & (centers <= end - nfft / 2)
        if np.count_nonzero(noise_frames) >= 8:
            power = np.abs(spectrum) ** 2
            noise_power = np.median(power[:, noise_frames], axis=1, keepdims=True)
            gain = np.sqrt(np.maximum(0, 1 - noise_power / np.maximum(power, 1e-20)))
            gain = np.maximum(10 ** (-max_reduction_db / 20), gaussian_filter(gain, sigma=(1, 2)))
            _, reconstructed = istft(spectrum * gain, fs=sr, nperseg=nfft, noverlap=nfft - hop, boundary=True)
            signal = reconstructed[:len(audio)]
            report.update({"noise_reduction_applied": True, "noise_profile_frames": int(noise_frames.sum()),
                           "noise_profile_seconds": sum(end - start for start, end in quiet_samples) / sr})
    if len(signal) != len(audio) or not np.isfinite(signal).all():
        raise ValueError("Denoiser violated finite, length-preserving output contract")
    # Headroom only when needed, not loudness normalization that boosts quiet
    # breaths/noise. Raw and cleaned versions have exactly the same frame grid.
    peak = float(np.max(np.abs(signal)))
    attenuation = min(1.0, .98 / peak) if peak else 1.0
    signal = (signal * attenuation).astype(np.float32)
    report.update({"output_frames": len(signal), "output_peak": float(np.max(np.abs(signal))),
                   "headroom_gain": attenuation, "noise_profile_status": "measured" if report["noise_reduction_applied"] else "insufficient-quiet-profile-no-spectral-denoising"})
    return signal, report


def speech_end(audio, sr, rough_end_sec, hard_limit_sec):
    """First sustained decay, bounded by the NEXT timestamped speech onset.

    Energy cannot tell an extended vowel from a new ayah. It must never override
    a text/timestamp wall. Continuous speech up to that wall is flagged for review.
    """
    duration = len(audio) / sr
    rough = max(0.0, min(float(rough_end_sec), duration))
    wall = max(0.0, min(float(hard_limit_sec), duration))
    if rough > wall:
        return wall, "timestamp-crosses-wall"
    start, limit = int(rough * sr), int(wall * sr)
    frame_size, hop = max(1, int(.025 * sr)), max(1, int(.01 * sr))
    reference = audio[max(0, start - int(.35 * sr)):min(len(audio), start + frame_size)]
    local_rms = float(np.sqrt(np.mean(reference.astype(np.float64) ** 2))) if len(reference) else 0
    threshold = max(1e-5, .04 * local_rms)
    last_voiced, silent, first_silent = start, 0, start
    for position in range(start, limit, hop):
        frame = audio[position:min(limit, position + frame_size)]
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2))) if len(frame) else 0
        if rms > threshold:
            last_voiced = min(limit, position + frame_size)
            silent = 0
        else:
            if silent == 0:
                first_silent = position
            silent += hop
            if silent >= int(.2 * sr):
                end = min(limit, max(start, last_voiced, first_silent) + int(.05 * sr))
                return end / sr, "sustained-decay"
    return wall, "word-timestamp-wall-no-silence"


def find_dynamic_speech_end(audio, sr, rough_end_sec, hard_limit_sec):
    return speech_end(audio, sr, rough_end_sec, hard_limit_sec)[0]


def find_surah_file_end(audio, sr, last_ayah_start_sec):
    """Legacy helper: first decay, not a backwards EOF scan or a 29.5s trim.

    The main slicer uses the matched TERMINAL word end instead of this start-only
    compatibility helper; EOF is never inferred from an ayah number.
    """
    return find_dynamic_speech_end(audio, sr, last_ayah_start_sec, len(audio) / sr)


def align_surah_with_hard_walls(whisper_words, all_refs, audio_duration, max_surah_ayah, is_fatiha=False, return_report=False):
    bounds, preamble, report = align_references(whisper_words, all_refs, audio_duration)
    return (bounds, preamble, report) if return_report else (bounds, preamble)


def slice_surah_with_waveform_tracking(audio, sr, ayah_raw_bounds, preamble_wall, max_surah_ayah, return_report=False, max_clip_seconds=30.0):
    if sr <= 0 or audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError("Need finite mono audio and a positive sample rate")
    duration = len(audio) / sr
    # Protect ALL known onsets, including an ayah whose own end is unsafe.
    ordered = sorted(((number, row) for number, row in ayah_raw_bounds.items() if number >= 1), key=lambda pair: pair[1]["start"])
    output, rejected = [], []
    previous_accepted_end = 0.0
    previous_accepted_end_frame = 0
    for index, (number, candidate) in enumerate(ordered):
        start, end = candidate["start"], candidate["end"]
        flags = list(candidate.get("flags", []))
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration):
            rejected.append({"ayah": number, "reason": "invalid-timestamps"}); continue
        walls = [duration]
        if candidate.get("speech_wall") is not None:
            walls.append(candidate["speech_wall"])
        if index + 1 < len(ordered):
            walls.append(ordered[index + 1][1]["start"])
        wall = min(walls)
        # Do not "repair" a fused label by chopping it and pretending its full
        # reference was spoken before the cut. The NEXT ayah stays independent.
        if flags or end > wall + .001 or start < preamble_wall or start < previous_accepted_end - .001:
            rejected.append({"ayah": number, "reason": "unsafe-overlapping-boundary", "flags": flags}); continue
        final_start = max(0, start if candidate.get("acoustic_start") else start - .04, previous_accepted_end, preamble_wall)
        final_end, method = (end, "joint-measured-pause") if candidate.get("acoustic_end") else speech_end(audio, sr, end, wall)
        # Keep the shared cut on the INTEGER sample grid. Seconds -> samples
        # round-trips can floor an exact prior end to end_frame-1 and duplicate
        # one sample at the next clip even when timestamp intervals look equal.
        start_frame = max(0, candidate.get("override_start_frame", math.floor(final_start * sr)), previous_accepted_end_frame, math.ceil(preamble_wall * sr))
        end_frame = min(len(audio), candidate.get("override_end_frame", math.floor(final_end * sr)))
        length = (end_frame - start_frame) / sr
        if length < .25 or length > max_clip_seconds:
            rejected.append({"ayah": number, "reason": "duration-out-of-range", "duration": length}); continue
        # Rejected candidates must NEVER move this boundary and damage later ones.
        previous_accepted_end = end_frame / sr
        previous_accepted_end_frame = end_frame
        output.append({"ayah": number, "data": candidate["data"], "start": start_frame / sr, "end": end_frame / sr,
            "start_frame": start_frame, "end_frame": end_frame, "is_true_final": number == max_surah_ayah,
            "boundary_method": method, "coverage": candidate.get("coverage"), "precision": candidate.get("precision"),
            "flags": ["timestamp-boundary-needs-listening"] if method not in {"sustained-decay", "joint-measured-pause"} else [],
            "alignment_status": "proposal-needs-listening-review", "usable_for_training": False})
    report = {"rejected": rejected, "usable_for_training": False}
    return (output, report) if return_report else output


def load_source_audio(source):
    """Decode to a documented 16 kHz mono frame grid; no silent decode skips."""
    import torch
    from torchaudio.functional import resample
    try:
        audio, sr = sf.read(source, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sr != 16000:
            audio = resample(torch.from_numpy(audio), sr, 16000).numpy()
        method = {"decoder": "libsndfile-torchaudio", "original_sample_rate": sr}
    except (RuntimeError, sf.LibsndfileError) as error:
        # Some otherwise valid publisher MP3 bitstreams are rejected by
        # libsndfile. Use PyAV/FFmpeg explicitly; don't skip malformed packets,
        # concatenate arbitrary clips or silently substitute another recording.
        import av
        pieces = []
        with av.open(str(source)) as container:
            if len(container.streams.audio) != 1:
                raise ValueError("Fallback expects exactly one source audio stream")
            original_sr = container.streams.audio[0].codec_context.sample_rate
            converter = av.AudioResampler(format="s16", layout="mono", rate=16000)
            for frame in container.decode(audio=0):
                for converted in converter.resample(frame):
                    pieces.append(converted.to_ndarray().reshape(-1).astype(np.float32) / 32768)
            for converted in converter.resample(None):
                pieces.append(converted.to_ndarray().reshape(-1).astype(np.float32) / 32768)
        audio = np.concatenate(pieces) if pieces else np.array([], dtype=np.float32)
        method = {"decoder": "PyAV-FFmpeg-s16-resampler", "original_sample_rate": original_sr,
                  "fallback_reason": f"{type(error).__name__}: {error}", "av_version": av.__version__}
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Empty/non-finite decoded recording")
    return audio, 16000, {**method, "output_frames": len(audio), "sample_rate": 16000}


def transcribe_words(model, audio, vad_filter=True):
    segments, _ = model.transcribe(audio, language="ar", task="transcribe", beam_size=5,
        word_timestamps=True, vad_filter=vad_filter, condition_on_previous_text=False, initial_prompt=None)
    words = []
    for segment in segments:
        for word in segment.words or []:
            if normalize_quran_for_asr(word.word):
                words.append({"clean_word": normalize_quran_for_asr(word.word), "raw_word": word.word,
                              "start": word.start, "end": word.end})
    return words


def pcm16_audio(audio, sr):
    """Verify exactly the samples that will be written, not pre-PCM floats."""
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    decoded, rate = sf.read(buffer, dtype="float32")
    if rate != sr or len(decoded) != len(audio):
        raise ValueError("PCM round-trip changed the sample grid")
    return decoded


def quarantine_neighbor_cuts(entries, bounds):
    """A failed left-terminal check invalidates its shared right start too.

    Snapshot failures, so quarantine does not cascade to every following ayah.
    The same recognizer may simply omit the leaked previous word in the right
    clip; right-only transcript agreement cannot absolve a failed shared cut.
    """
    by_number = {entry["candidate"]["ayah"]: entry for entry in entries}
    failed = {number for number, entry in by_number.items() if not entry["verification"]["passed"]}
    failed.update(set(bounds) - set(by_number))
    for number, entry in by_number.items():
        previous = bounds.get(number, {}).get("previous_boundary_ayah")
        if previous in failed and entry["verification"]["passed"]:
            entry["verification"] = {**entry["verification"], "own_content_passed": True, "passed": False,
                "reason": "shared-start-depends-on-rejected-previous-ayah", "depends_on_ayah": previous}
    return entries


def previous_terminal_evidence(report, number, root):
    """Preserve independent END evidence from a start-quarantined predecessor.

    Its whole clip is not approved/exported, but failing its START must not
    mechanically invalidate its independently checked terminal boundary too.
    An actual failed word/end check NEVER supplies this evidence.
    """
    check = next((r for r in report.get("clip_verification", []) if r["ayah"] == number), None)
    proposal = next((r for r in report.get("review_candidates", []) if r["ayah"] == number), None)
    event = next((r for r in report.get("joint_boundary_events", []) if r["after_ayah"] == number), None)
    cleaned_passed = (report.get("cleaning", {}).get("mode") == "none"
                      or bool(check and check.get("cleaned_check", {}).get("exact_check", {}).get("passed")))
    if not (report.get("version") == VERSION and check and check.get("own_content_passed")
            and check.get("reason") == "shared-start-depends-on-rejected-previous-ayah"
            and check.get("exact_check", {}).get("passed") and cleaned_passed and proposal and event and event.get("resolved")):
        return None
    path = (Path(root) / proposal["path"]).resolve()
    return {"ayah": number, "guard_version": VERSION, "source_sha256": report["source_sha256"],
        "raw_audio_path": str(path), "raw_audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "start_frame": proposal["start_frame"], "end_frame": proposal["end_frame"], "sample_rate": 16000,
        "exact_check": check["exact_check"], "cleaned_exact_passed": cleaned_passed,
        "terminal_boundary": event, "predecessor_whole_clip_approved": False}


def recover_missing_references(model, audio, sr, references, bounds, duration):
    """Blind local retries bounded by neighboring full-reference evidence.

    No expected-text prompt/forced transcription. Successful local matches must
    fit between untouched existing matches; uncertain retries only add evidence.
    This is not phoneme forced alignment or permission to invent skipped ayahs.
    """
    recovered, attempts = dict(bounds), []
    missing = [r for r in references if r["ayah"] >= 1 and r["ayah"] not in bounds]
    windows = {}
    for reference in missing:
        number = reference["ayah"]
        before = [b for n, b in bounds.items() if n < number]
        after = [b for n, b in bounds.items() if n > number]
        left = max((b["end"] for b in before), default=0)
        right = min((b["start"] for b in after), default=duration)
        first, last = max(0, left - 1), min(duration, right + 1)
        windows.setdefault((round(first, 3), round(last, 3)), []).append(reference)
    for (first, last), refs_to_find in windows.items():
        # Long unresolved regions use overlapping blind 28s windows; all
        # timestamps stay on the original recording's absolute sample grid.
        position = first
        while position < last:
            stop = min(last, position + 28)
            first_frame, last_frame = math.floor(position * sr), math.ceil(stop * sr)
            heard = transcribe_words(model, audio[first_frame:last_frame], vad_filter=False)
            shifted = [{**word, "start": word["start"] + first_frame / sr,
                        "end": word["end"] + first_frame / sr} for word in heard]
            prepared, invalid = prepare_words(shifted, duration)
            attempt = {"start": first_frame / sr, "end": last_frame / sr, "ayahs": [r["ayah"] for r in refs_to_find],
                       "words": shifted, "invalid_timestamps": invalid, "recovered": [], "rejected": []}
            for reference in refs_to_find:
                number = reference["ayah"]
                if number in recovered:
                    continue
                for candidate in candidates_for_reference(prepared, reference):
                    if candidate["flags"]:
                        continue
                    if any(candidate["start"] < b["end"] and b["start"] < candidate["end"] for b in recovered.values()):
                        attempt["rejected"].append({"ayah": number, "reason": "regional-match-overlaps-existing-evidence"})
                        continue
                    if any((n < number and b["start"] > candidate["start"]) or (n > number and b["start"] < candidate["start"])
                           for n, b in recovered.items()):
                        continue
                    recovered[number] = {**candidate, "recovery_method": "blind-neighbor-bounded-regional-ASR"}
                    attempt["recovered"].append(number)
                    break
            attempts.append(attempt)
            if stop == last:
                break
            position += 24
    return recovered, attempts


def coverage_manifest(database, scope, metadata, reports, reciter, reciter_key):
    """One honest schema-compatible inventory row per EXPECTED canonical ID.

    This is coverage.jsonl, NOT training metadata; absent audio stays null. A
    569-line count is never manufactured by duplicating/guessing training WAVs.
    """
    found = {(row["surah"], row["ayah"]): row for row in metadata}
    if len(found) != len(metadata):
        raise ValueError("Duplicate exported canonical IDs")
    decisions = {(report["surah"], row["ayah"]): row for report in reports for row in report.get("ayah_decisions", [])}
    contexts = {(report["surah"], row["ayah"]): row for report in reports for row in report.get("review_contexts", [])}
    contexts.update({(report["surah"], row["ayah"]): row for report in reports for row in report.get("review_candidates", [])})
    expected = {key for key in database if key[0] in scope}
    if set(found) - expected:
        raise ValueError("Exported IDs outside canonical scope")
    rows = []
    for surah, ayah in sorted(key for key in database if key[0] in scope):
        reference = database[surah, ayah]
        key = surah, ayah
        if key in found:
            rows.append({**found[key], "coverage_status": "exported-unapproved", "missing_reason": None})
            continue
        decision = decisions.get(key, {})
        rows.append({"surah": surah, "ayah": ayah, "audio_filename": None, "relative_audio_path": None,
            "text": reference["text"], "text_asr_normalized": normalize_quran_for_asr(reference["raw"]),
            "text_raw_uthmani": reference["raw"], "source_ayahs": reference.get("source_ayahs", [ayah]),
            "reciter": reciter, "reciter_key": reciter_key, "start_time": None, "end_time": None,
            "start_frame": None, "end_frame": None, "sample_rate": 16000,
            "normalized_with_harakat": normalize_with_harakat(reference["raw"]),
            "coverage_status": "withheld-for-review", "missing_reason": decision.get("reason", "source-file-missing"),
            "review_audio": contexts.get(key), "alignment_status": "unresolved", "usable_for_training": False,
            **geometry_fields(surah, reference.get("source_ayahs", [ayah]))})
    return rows


def apply_boundary_overrides(bounds, references, overrides, surah, source_sha256, frames, sr):
    """Auditable manual rescue, never an automatic content/training approval.

    Overrides reference immutable recording hashes and absolute decoded frames;
    no negative/overlapping/duplicate/unknown cuts or text replacement is allowed.
    Raw+cleaned blind content gates still run on every supplied cut.
    """
    adjusted = {number: {**row, "flags": list(row.get("flags", []))} for number, row in bounds.items()}
    canonical = {r["ayah"]: r for r in references if r["ayah"] >= 1}
    seen = set()
    chosen = sorted((r for r in overrides if r.get("surah") == surah), key=lambda r: r["ayah"])
    previous_end = -1
    for row in chosen:
        number = row["ayah"]
        start, end = row.get("start_frame"), row.get("end_frame")
        if type(number) is not int or number not in canonical or number in seen or row.get("source_sha256") != source_sha256:
            raise ValueError("Manual cut has unknown/duplicate canonical ID or wrong recording hash")
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= frames
                or type(row.get("sample_rate")) is not int or row.get("sample_rate") != sr or start < previous_end
                or not row.get("reviewed_by") or not row.get("reviewed_at")):
            raise ValueError("Manual cut requires reviewed identity/date, exact valid monotonic 16kHz frames")
        adjusted[number] = {"ayah": number, "data": canonical[number], "start": start / sr, "end": end / sr,
            "override_start_frame": start, "override_end_frame": end, "speech_wall": None, "flags": [],
            "acoustic_start": True, "acoustic_end": True, "manual_boundary_override": row, "usable_for_training": False}
        seen.add(number)
        previous_end = end
    for number, row in adjusted.items():
        previous = adjusted.get(number - 1)
        if previous and abs(previous["end"] - row["start"]) < 1 / sr:
            row["previous_boundary_ayah"] = number - 1
    return adjusted


def surah_references(database, surah):
    refs = [{"ayah": -2, "text": "أعوذ بالله من الشيطان الرجيم", "text_asr": "اعوذ بالله من الشيطان الرجيم",
             "raw": "أَعُوذُ بِاللَّهِ مِنَ الشَّيْطَانِ الرَّجِيمِ", "source_ayahs": []}]
    if surah != 9:
        refs.append({"ayah": -1, "text": "بسم الله الرحمن الرحيم", "text_asr": "بسم الله الرحمن الرحيم",
                     "raw": "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ", "source_ayahs": []})
    for s, number in sorted(database):
        if s == surah:
            row = database[(s, number)]
            refs.append({"ayah": number, "text": row["text"], "text_asr": normalize_quran_for_asr(row["raw"]),
                         "raw": row["raw"], "source_ayahs": row.get("source_ayahs", [number])})
    return refs


def load_segmentation_model(name, device, compute_type, tarteel_dir=None):
    from faster_whisper import WhisperModel
    if name != "tarteel-base":
        return WhisperModel(name, device=device, compute_type=compute_type), {"model": name}
    from src.dataset_collection.prepare_tarteel_segmenter import MODEL, REVISION, TOKEN_REVISION
    directory = Path(tarteel_dir or ROOT / "models/segmentation/tarteel-whisper-base-ct2")
    receipt = json.loads((directory / "segmentation_model.json").read_text(encoding="utf-8"))
    if (receipt.get("model") != MODEL or receipt.get("revision") != REVISION
            or receipt.get("token_metadata_revision") != TOKEN_REVISION or receipt.get("shared_token_ids_verified") is not True):
        raise ValueError("Tarteel conversion provenance missing or mismatched; run prepare_tarteel_segmenter.py")
    return WhisperModel(str(directory), device=device, compute_type=compute_type), receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "taha_al_fahad_juz_amma/mp3")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/segmentation_review/taha")
    parser.add_argument("--model", default="large-v3", choices=["tiny", "base", "small", "large-v3", "tarteel-base"])
    parser.add_argument("--tarteel-model-dir", type=Path, default=ROOT / "models/segmentation/tarteel-whisper-base-ct2")
    parser.add_argument("--compute-type", default="auto", choices=["auto", "float16", "int8_float16", "int8", "float32"], help="Use int8_float16 for a lower-memory CUDA large-v3 pilot")
    parser.add_argument("--reciter", default="Taha Mohamed Abdulrahman Al-Fahad (Qaloon)")
    parser.add_argument("--reciter-key", default="taha")
    parser.add_argument("--surahs", help="Comma-separated scoped pilot, e.g. 1,112 (default: Fatiha/Juz Amma files)")
    parser.add_argument("--skip-clip-verification", action="store_true", help="Explicitly disable blind second-pass ASR; output remains unapproved")
    parser.add_argument("--max-clip-seconds", type=float, default=30.0)
    parser.add_argument("--boundary-mode", choices=["silence", "timestamp"], default="silence", help="Joint measured-pause boundaries by default; timestamp mode is legacy review-only")
    parser.add_argument("--boundary-search-seconds", type=float, default=1.75)
    parser.add_argument("--cleaning", choices=["none", "conservative"], default="conservative")
    parser.add_argument("--hum-frequency", type=int, choices=[50, 60], help="Optional measured mains-hum notch; do not guess the frequency")
    parser.add_argument("--content-match", choices=["exact", "fuzzy"], default="exact", help="Exact normalized tokens by default; fuzzy is proposal-only legacy behavior")
    parser.add_argument("--no-regional-recovery", action="store_true", help="Disable bounded blind retries for missing full-text matches")
    parser.add_argument("--allow-incomplete", action="store_true", help="Explicit review pilot: otherwise any missing canonical ID exits 2")
    parser.add_argument("--boundary-overrides", type=Path, help="Reviewer-authored JSON array of source-hashed absolute frame cuts; never bypasses content checks")
    args = parser.parse_args()
    scope = {1, *range(78, 115)}
    if args.surahs:
        try:
            requested = {int(item.strip()) for item in args.surahs.split(",")}
        except ValueError:
            parser.error("surahs must be comma-separated integers")
        if not requested or not requested.issubset(scope):
            parser.error("This collector is scoped to Fatiha and surahs 78–114")
        scope = requested
    if not math.isfinite(args.max_clip_seconds) or args.max_clip_seconds <= 0:
        parser.error("max-clip-seconds must be positive and finite")
    if not math.isfinite(args.boundary_search_seconds) or not .2 <= args.boundary_search_seconds <= 3:
        parser.error("boundary-search-seconds must be between .2 and 3")
    if not args.input_dir.is_dir():
        parser.error("Input directory does not exist")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory is not empty; choose a NEW directory. Existing dataset/reviewer files are never overwritten.")
    audio_files = {}
    for file in sorted(args.input_dir.iterdir()):
        if file.suffix.lower() not in {".mp3", ".wav", ".ogg", ".flac"}:
            continue
        match = re.search(r"(?<!\d)(001|07[89]|08\d|09\d|10\d|11[0-4]|1)(?!\d)", file.stem)
        if match and int(match[1]) in scope:
            number = int(match[1])
            if number in audio_files:
                parser.error(f"Multiple sources for surah {number}; select one source per run, do not silently overwrite ayah filenames")
            audio_files[number] = file
    if not audio_files:
        parser.error("No scoped source audio files found")
    overrides = json.loads(args.boundary_overrides.read_text(encoding="utf-8")) if args.boundary_overrides else []
    if not isinstance(overrides, list) or any(not isinstance(row, dict) or row.get("surah") not in audio_files for row in overrides):
        parser.error("Overrides must be a JSON array restricted to the source surahs in this run")
    import torch
    from torchaudio.functional import resample
    from faster_whisper import WhisperModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = ("float16" if device == "cuda" else "int8") if args.compute_type == "auto" else args.compute_type
    print(f"Loading blind {args.model} on {device}/{compute_type}; second-pass verification: {not args.skip_clip_verification}", flush=True)
    model, model_provenance = load_segmentation_model(args.model, device, compute_type, args.tarteel_model_dir)
    database = load_quran()
    output_audio = args.output_dir / "audio"
    output_audio.mkdir(parents=True, exist_ok=True)
    raw_audio_dir = args.output_dir / "audio_raw"
    raw_audio_dir.mkdir()
    review_audio_dir = args.output_dir / "review_audio"
    review_audio_dir.mkdir()
    reports_dir = args.output_dir / "reports"
    reports_dir.mkdir()
    metadata, run_reports, full_reports, failed_sources = [], [], [], []
    for surah, source in sorted(audio_files.items()):
        refs = surah_references(database, surah)
        maximum = max(r["ayah"] for r in refs)
        report = {"surah": surah, "source": str(source.resolve()), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "model": args.model, "model_provenance": model_provenance, "device": device, "compute_type": compute_type, "reciter_key": args.reciter_key, "usable_for_training": False,
                  "version": VERSION, "boundary_mode": args.boundary_mode, "boundary_search_seconds": args.boundary_search_seconds,
                  "content_match": args.content_match,
                  "authorization_status": "needs-source-permission-review", "exported_ayahs": [], "clip_verification": [], "review_contexts": [], "review_candidates": []}
        try:
            audio, sr, decode_report = load_source_audio(source)
            report["source_decode"] = decode_report
            cleaned, cleaning_report = clean_audio(audio, sr, args.cleaning, args.hum_frequency)
            words = transcribe_words(model, cleaned)
            bounds, preamble, alignment = align_surah_with_hard_walls(words, refs, len(audio) / sr, maximum, surah == 1, True)
            if not args.no_regional_recovery:
                bounds, retries = recover_missing_references(model, audio, sr, refs, bounds, len(audio) / sr)
                report["regional_recovery"] = retries
                for row in alignment["ayahs"]:
                    if row["ayah"] in bounds and row["status"] != "matched-candidate":
                        candidate = bounds[row["ayah"]]
                        row.update({"status": "regional-recovered-candidate", "reason": None,
                            "occurrences": [{k: candidate[k] for k in ("start", "end", "word_start", "word_end", "coverage", "precision", "flags")}],
                            "candidate_count": 1})
            raw_bounds = bounds
            boundary_events = []
            if args.boundary_mode == "silence":
                # Analyze original waveform, not denoiser-created silence.
                bounds, preamble, boundary_events = refine_ayah_boundaries(audio, sr, bounds, preamble, args.boundary_search_seconds)
            if overrides:
                bounds = apply_boundary_overrides(bounds, refs, overrides, surah, report["source_sha256"], len(audio), sr)
                report["manual_boundary_overrides"] = [row for row in overrides if row["surah"] == surah]
            clips, slicing = slice_surah_with_waveform_tracking(audio, sr, bounds, preamble, maximum, True, args.max_clip_seconds)
            report.update({"words": words, "alignment": alignment, "slicing": slicing, "duration": len(audio) / sr,
                           "cleaning": cleaning_report, "joint_boundary_events": boundary_events})
            entries = []
            for candidate in clips:
                number, data = candidate["ayah"], candidate["data"]
                excerpt = pcm16_audio(audio[candidate["start_frame"]:candidate["end_frame"]], sr)
                cleaned_excerpt = pcm16_audio(cleaned[candidate["start_frame"]:candidate["end_frame"]], sr)
                verification = {"passed": False, "reason": "verification-explicitly-disabled"}
                if not args.skip_clip_verification:
                    clip_words = transcribe_words(model, excerpt, vad_filter=False)
                    verification = validate_clip_transcript(" ".join(w["clean_word"] for w in clip_words), data["text_asr"],
                         [r for r in refs if r["ayah"] > number])
                    if args.content_match == "exact":
                        strict = exact_text_check(data["text_asr"], " ".join(w["clean_word"] for w in clip_words))
                        verification.update({"exact_check": strict, "passed": strict["passed"],
                                             "reason": None if strict["passed"] else "non-exact-raw-clip-transcript"})
                    if args.cleaning != "none" and verification["passed"]:
                        cleaned_text = " ".join(w["clean_word"] for w in transcribe_words(model, cleaned_excerpt, vad_filter=False))
                        checked_clean = validate_clip_transcript(cleaned_text, data["text_asr"],
                            [r for r in refs if r["ayah"] > number])
                        if args.content_match == "exact":
                            strict_clean = exact_text_check(data["text_asr"], cleaned_text)
                            checked_clean.update({"exact_check": strict_clean, "passed": strict_clean["passed"],
                                                  "reason": None if strict_clean["passed"] else "non-exact-cleaned-clip-transcript"})
                        verification = {**verification, "raw_passed": True, "cleaned_check": checked_clean,
                                        "passed": checked_clean["passed"], "reason": checked_clean["reason"]}
                entries.append({"candidate": candidate, "excerpt": excerpt, "cleaned_excerpt": cleaned_excerpt, "verification": verification})
            entries = quarantine_neighbor_cuts(entries, bounds)
            for entry in entries:
                candidate, excerpt, cleaned_excerpt, verification = (entry[k] for k in ("candidate", "excerpt", "cleaned_excerpt", "verification"))
                number, data = candidate["ayah"], candidate["data"]
                report["clip_verification"].append({"ayah": number, **verification})
                if not verification["passed"]:
                    # Retain complete boundary proposals for manual rescue,
                    # distinct from training/accepted audio and wider contexts.
                    review_name = f"{surah:03d}{number:03d}.candidate.wav"
                    sf.write(review_audio_dir / review_name, excerpt, sr, subtype="PCM_16")
                    report["review_candidates"].append({"ayah": number, "path": f"review_audio/{review_name}",
                        "start_frame": candidate["start_frame"], "end_frame": candidate["end_frame"],
                        "reference": data["text_asr"], "reason": verification["reason"], "usable_for_training": False})
                    continue
                filename = f"{surah:03d}{number:03d}.wav"
                destination = output_audio / filename
                if destination.exists():
                    raise ValueError(f"Refusing to overwrite existing candidate {destination}")
                sf.write(destination, cleaned_excerpt, sr, subtype="PCM_16")
                raw_destination = raw_audio_dir / filename
                sf.write(raw_destination, excerpt, sr, subtype="PCM_16")
                metadata.append({"surah": surah, "ayah": number, "audio_filename": filename,
                    "relative_audio_path": f"audio/{filename}", "text": data["text"], "text_asr_normalized": data["text_asr"],
                    "text_raw_uthmani": data["raw"], "source_ayahs": data["source_ayahs"], "reciter": args.reciter,
                    "reciter_key": args.reciter_key, "start_time": round(candidate["start"] * 1000), "end_time": round(candidate["end"] * 1000),
                    "start_frame": candidate["start_frame"], "end_frame": candidate["end_frame"], "sample_rate": sr,
                    "normalized_with_harakat": normalize_with_harakat(data["raw"]),
                    "source_file": str(source.resolve()), "source_sha256": report["source_sha256"],
                    "source_decode": decode_report,
                    "audio_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                    "raw_audio_path": f"audio_raw/{filename}", "raw_audio_sha256": hashlib.sha256(raw_destination.read_bytes()).hexdigest(),
                    "cleaning": cleaning_report,
                    "alignment_status": "auto-checked-needs-listening-review", "usable_for_training": False,
                    "authorization_status": report["authorization_status"], "boundary_method": candidate["boundary_method"],
                    "boundary_flags": candidate["flags"], "clip_verification": verification,
                    "previous_boundary_ayah": bounds[number].get("previous_boundary_ayah"),
                    "manual_boundary_override": bounds[number].get("manual_boundary_override"),
                    **geometry_fields(surah, data["source_ayahs"])})
                report["exported_ayahs"].append(number)
            report["unexported_ayahs"] = [r["ayah"] for r in refs if r["ayah"] >= 1 and r["ayah"] not in report["exported_ayahs"]]
            for row in (r for r in metadata if r["surah"] == surah):
                previous = row.get("previous_boundary_ayah")
                if previous is not None and previous not in report["exported_ayahs"]:
                    row["previous_terminal_evidence"] = previous_terminal_evidence(report, previous, args.output_dir)
            # Preserve wider RAW contexts for failed proposals, clearly separate
            # from ayah WAVs. Never fabricate a timing for an unmatched reference.
            for row in alignment["ayahs"]:
                number = row["ayah"]
                if number in report["exported_ayahs"]:
                    continue
                proposal = raw_bounds.get(number)
                occurrence = proposal or next(iter(row["occurrences"]), None)
                if occurrence is None:
                    continue
                first = max(0, math.floor((occurrence["start"] - .25) * sr))
                last = min(len(audio), math.ceil((occurrence["end"] + 1.25) * sr))
                context_name = f"{surah:03d}{number:03d}.context.wav"
                sf.write(review_audio_dir / context_name, audio[first:last], sr, subtype="PCM_16")
                report["review_contexts"].append({"ayah": number, "path": f"review_audio/{context_name}",
                    "start_frame": first, "end_frame": last, "contains_context_not_an_ayah_clip": True, "usable_for_training": False})
            print(f"Surah {surah:03d}: {len(report['exported_ayahs'])}/{maximum} review clips; unexported {report['unexported_ayahs']}", flush=True)
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"
            report["unexported_ayahs"] = [r["ayah"] for r in refs if r["ayah"] >= 1 and r["ayah"] not in report["exported_ayahs"]]
            failed_sources.append(surah)
            print(f"Surah {surah:03d} failed: {error}", flush=True)
        alignment_rows = {r["ayah"]: r for r in report.get("alignment", {}).get("ayahs", [])}
        rejected_rows = {r["ayah"]: r for r in report.get("slicing", {}).get("rejected", [])}
        verification_rows = {r["ayah"]: r for r in report["clip_verification"]}
        report["ayah_decisions"] = [{"ayah": r["ayah"], "reference": r["text_asr"],
            "status": "exported-unapproved" if r["ayah"] in report["exported_ayahs"] else "withheld-for-review",
            "reason": None if r["ayah"] in report["exported_ayahs"] else (
                report.get("error") or rejected_rows.get(r["ayah"], {}).get("reason")
                or verification_rows.get(r["ayah"], {}).get("reason") or alignment_rows.get(r["ayah"], {}).get("reason") or "needs-review"),
            "alignment": alignment_rows.get(r["ayah"]), "boundary_rejection": rejected_rows.get(r["ayah"]),
            "clip_check": verification_rows.get(r["ayah"])} for r in refs if r["ayah"] >= 1]
        (reports_dir / f"{surah:03d}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        full_reports.append(report)
        run_reports.append({"surah": surah, "exported_ayahs": report["exported_ayahs"], "unexported_ayahs": report["unexported_ayahs"], "error": report.get("error")})
    (args.output_dir / "metadata.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in metadata), encoding="utf-8")
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    coverage = coverage_manifest(database, scope, metadata, full_reports, args.reciter, args.reciter_key)
    missing_ids = [[r["surah"], r["ayah"]] for r in coverage if r["coverage_status"] != "exported-unapproved"]
    completeness = {"expected_rows": len(coverage), "exported_rows": len(metadata), "missing_ids": missing_ids,
                    "complete": not missing_ids, "training_approved": False}
    (args.output_dir / "coverage.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in coverage), encoding="utf-8")
    (args.output_dir / "run_report.json").write_text(json.dumps({"surahs": run_reports, "failed_sources": failed_sources,
        "coverage": completeness,
        "missing_source_surahs": sorted(scope - set(audio_files)), "exported_candidates": len(metadata), "approved_clips": 0,
        "usable_for_training": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "review_bundle.json").write_text(json.dumps({"version": VERSION, "source_reports": full_reports,
        "coverage": completeness,
        "metadata": metadata, "failed_sources": failed_sources, "missing_source_surahs": sorted(scope - set(audio_files)),
        "approved_clips": 0, "usable_for_training": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(metadata)} unapproved review clips in {args.output_dir}; existing datasets unchanged.", flush=True)
    if failed_sources:
        raise SystemExit(1)
    if missing_ids and not args.allow_incomplete:
        print(f"INCOMPLETE: {len(missing_ids)} canonical IDs withheld; coverage.jsonl accounts for all {len(coverage)} references. Not a complete dataset.", flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
