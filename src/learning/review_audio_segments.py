"""Make auditable segmentation PROPOSALS, never trusted training labels.

Use timestamped blind ASR (optionally Whisper large-v3), then bounded local
reference alignment. Missing references are left missing; repeats stay separate
occurrences; basmalah is an explicit reference, never a fixed-duration trim.
"""
import argparse
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import sys

import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/dataset_collection"))
from qaloon_audio2text import normalize_quran_for_asr


def references(surah):
    data = json.loads((ROOT / f"web/public/quran/surahs/{surah:03d}.json").read_text(encoding="utf-8"))
    refs = [{"ayah": a["ayah"], "text": a["normalized"]} for a in data["ayahs"]]
    if surah != 9:
        refs.insert(0, {"ayah": 0, "text": "بسم الله الرحمن الرحيم"})
    return refs


def propose_segments(timestamped_words, refs, min_coverage=0.85):
    """Keep only strong LOCAL matches; never force the entire reference graph.

    Exact first/last anchors and ordered interior matches are required. Short
    repeated/common phrases stay flagged as ambiguous; no transcript repair.
    """
    heard = []
    for item in timestamped_words:
        start, end = item.get("start"), item.get("end")
        if start is None or end is None or not 0 <= start < end:
            continue
        for word in normalize_quran_for_asr(item.get("word", "")).split():
            heard.append({"word": word, "start": float(start), "end": float(end)})
    tokens = [w["word"] for w in heard]
    candidates = []
    for ref in refs:
        expected = ref["text"].split()
        if not expected:
            continue
        for first, token in enumerate(tokens):
            if token != expected[0]:
                continue
            possible = []
            limit = min(len(tokens), first + 2 * len(expected) + 6)
            for last in range(first + max(1, len(expected) // 2) - 1, limit):
                if tokens[last] != expected[-1]:
                    continue
                observed = tokens[first:last + 1]
                blocks = SequenceMatcher(None, expected, observed, autojunk=False).get_matching_blocks()
                matched = sum(block.size for block in blocks)
                coverage = matched / len(expected)
                precision = matched / len(observed)
                if coverage < min_coverage or precision < 0.8 or matched < min(2, len(expected)):
                    continue
                possible.append({"ayah": ref["ayah"], "start": heard[first]["start"], "end": heard[last]["end"],
                    "word_start": first, "word_end": last + 1, "coverage": round(coverage, 4),
                    "precision": round(precision, 4), "observed": " ".join(observed), "expected": ref["text"],
                    "status": "proposal-needs-listening-review", "usable_for_training": False})
            if possible:
                candidates.append(max(possible, key=lambda row: (row["coverage"] + row["precision"], -(row["end"] - row["start"]))))
    # Expose overlapping candidate references instead of choosing an arbitrary
    # religious/text label for a repeated common phrase.
    candidates.sort(key=lambda row: (row["start"], row["end"], row["ayah"]))
    occurrences = {}
    for row in candidates:
        overlaps = [other["ayah"] for other in candidates if other is not row
                    and other["start"] < row["end"] and other["end"] > row["start"]]
        row["ambiguous_with"] = sorted(set(overlaps))
        occurrences[row["ayah"]] = occurrences.get(row["ayah"], 0) + 1
        row["occurrence"] = occurrences[row["ayah"]]
        row["flags"] = (["repeated-reference-occurrence"] if row["occurrence"] > 1 else []) + (["overlapping-reference-candidates"] if overlaps else [])
    found = {row["ayah"] for row in candidates if row["ayah"] != 0}
    return {"candidates": candidates, "unmatched_reference_ayahs": [r["ayah"] for r in refs if r["ayah"] != 0 and r["ayah"] not in found],
            "note": "Unmatched is not proof of absent audio. File IDs are not ayah labels. Partial waqf/restarts may not produce a complete-ayah proposal."}


def main(args):
    from faster_whisper import WhisperModel
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperModel(args.model, device=device, compute_type="float16" if device == "cuda" else "int8")
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    clips = inventory.get("clips", [])
    if args.limit:
        clips = clips[:args.limit]
    if not clips:
        raise ValueError("Inventory needs local clips. Assabile currently has no permitted downloaded audio.")
    args.output.mkdir(parents=True, exist_ok=True)
    for row in clips:
        source = (args.inventory.parent / row["relative_audio_path"]).resolve()
        if not source.is_relative_to(args.inventory.parent.resolve()):
            raise ValueError("Audio path escapes inventory")
        output = args.output / f"{source.stem}.json"
        if output.exists():
            raise ValueError(f"Review output exists; choose a new directory, do not overwrite reviewer work: {output}")
        audio, rate = sf.read(source, dtype="float32", always_2d=True)
        # faster-whisper's array input must be 16kHz; resampling via its loader
        # is used for other formats/rates, not silently pretending the rate.
        data = audio.mean(axis=1) if rate == 16000 else str(source)
        segments, info = model.transcribe(data, language="ar", task="transcribe", beam_size=5,
            word_timestamps=True, vad_filter=True, condition_on_previous_text=False,
            initial_prompt=None)
        words = [{"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                 for segment in segments for w in segment.words or []]
        proposals = propose_segments(words, references(int(row["surah"])))
        result = {"source_file": row["relative_audio_path"], "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "surah": row["surah"], "model": args.model, "blind_decoder": True, "timestamp_method": "Whisper cross-attention, not ground-truth boundaries",
                  "authorization_status": inventory.get("authorization_status", "needs-permission-review"), "usable_for_training": False,
                  "duration": info.duration, "words": words, **proposals}
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{source.name}: {len(proposals['candidates'])} proposed occurrences; nothing approved", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="large-v3", help="faster-whisper model or a local CTranslate2 directory")
    parser.add_argument("--limit", type=int, default=10, help="Pilot first; 0 explicitly processes the whole inventory")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit must be nonnegative")
    main(args)
