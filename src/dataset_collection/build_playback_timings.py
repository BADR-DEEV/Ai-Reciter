"""Blind ASR word timing proposals bound to actual reader WAVs and display text.

No uniform word/letter interpolation. Disagreement or grouped/overlapping
timestamps withhold word tracking; UI may still highlight the entire ayah.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.dataset_collection.qaloon_audio2text import normalize_quran_for_asr
from src.dataset_collection.segment_and_slice import load_segmentation_model, load_source_audio, transcribe_words


def checked_timings(words, display_text, duration):
    expected = normalize_quran_for_asr(display_text).split()
    display = display_text.split()
    if len(display) != len(expected) or not expected or len(words) != len(expected):
        return None
    result, previous_end = [], 0
    for index, (word, token) in enumerate(zip(words, expected)):
        start, end = word.get("start"), word.get("end")
        if (normalize_quran_for_asr(word.get("clean_word", "")) != token
                or not isinstance(start, (float, int)) or not isinstance(end, (float, int))
                or not math.isfinite(start) or not math.isfinite(end)
                or not previous_end <= start < end <= duration):
            return None
        result.append({"index": index, "text": display[index], "start": start, "end": end})
        previous_end = end
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--reciter", required=True, choices=["huthaify", "husary", "dokali"])
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--surahs", help="Optional comma-separated pilot scope")
    p.add_argument("--model", choices=["tarteel-base", "large-v3"], default="tarteel-base")
    p.add_argument("--tarteel-model-dir", type=Path, default=ROOT / "models/segmentation/tarteel-whisper-base-ct2")
    args = p.parse_args()
    if args.output.exists():
        p.error("Output exists; preserve earlier timings and choose a new path")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, provenance = load_segmentation_model(args.model, device, "int8_float16" if device == "cuda" else "int8", args.tarteel_model_dir)
    rows = [json.loads(line) for line in (args.dataset / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    scope = {int(x) for x in args.surahs.split(",")} if args.surahs else None
    texts, clips = {}, []
    for row in rows:
        surah, ayah = row["surah"], row["ayah"]
        if scope is not None and surah not in scope:
            continue
        if surah not in texts:
            data = json.loads((ROOT / f"web/public/quran/surahs/{surah:03d}.json").read_text(encoding="utf-8"))
            texts[surah] = {a["ayah"]: a for a in data["ayahs"]}
        source = texts[surah][ayah]
        display = source.get("displayText") or source["text"]
        import re
        display = re.sub(r"[\d\u0660-\u0669]+", "", display).strip()
        path = (args.dataset / row["relative_audio_path"]).resolve()
        if not path.is_relative_to(args.dataset.resolve()):
            raise ValueError("Audio outside dataset")
        audio, sr, _ = load_source_audio(path)
        duration = len(audio) / sr
        # Never assign full-text labels to truncated 30-second targets.
        words = transcribe_words(model, audio, vad_filter=False) if duration <= 30 else []
        timings = checked_timings(words, display, duration)
        clips.append({"surah": surah, "ayah": ayah, "reciter": args.reciter, "displayText": display,
            "normalized": source["normalized"], "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "duration": duration, "status": "machine-timing-proposal" if timings else "ayah-only",
            "words": timings or [], "approved": False})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema_version": 1, "reciter": args.reciter, "model": provenance,
        "clips": clips, "letter_timings": False}, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{args.reciter}: {sum(bool(c['words']) for c in clips)}/{len(clips)} exact word timing proposals; rest ayah-only. {args.output}")


if __name__ == "__main__":
    main()
