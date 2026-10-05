"""Join 2-4 consecutive verified Qaloon ayah clips into one training clip.

The live studio decodes a rolling window that holds several ayahs, but every training clip held
one ayah, so fast readers lost words at ayah joins. This builds multi-ayah clips from the same
reciter and surah, only from clips already admitted for training (flagged clips excluded).
Half of the clips keep their natural pauses; the other half have the silence at each join cut
to 0.05-0.35 s, as when reading fast. Labels are the source ayahs' Qaloon texts in order.

    python src/dataset_collection/build_joined_ayahs.py --data-root D:/data/train_root \
        --include-reciter huthaify husary ... --exclude-clips flagged.json --output-dir D:/data/joined
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "training"))
from qaloon_data import load_splits  # noqa: E402
from qaloon_audio2text import normalize_quran_for_asr, normalize_with_harakat  # noqa: E402

SR = 16000
MAX_SECONDS = 25.0  # Leaves room for a 0.85x tempo change inside Whisper's 30 s window.


def trim(audio):
    """Cut leading/trailing silence (below -40 dB of the clip peak), keeping 50 ms."""
    hop = int(SR * 0.01)
    frames = audio[: len(audio) // hop * hop].reshape(-1, hop)
    level = np.sqrt((frames ** 2).mean(axis=1))
    voiced = np.where(level > level.max() * 0.01)[0]
    if not len(voiced):
        return audio
    start = max(0, voiced[0] * hop - int(0.05 * SR))
    end = min(len(audio), (voiced[-1] + 1) * hop + int(0.05 * SR))
    return audio[start:end]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--include-reciter", nargs="+", required=True)
    p.add_argument("--exclude-clips", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--passes", type=int, default=2, help="Groupings with different offsets, so joins fall in different places")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    flagged = {item["key"] for item in json.loads(args.exclude_clips.read_text(encoding="utf-8"))["flagged"]}
    splits, _ = load_splits(args.data_root, include=args.include_reciter)
    rows = sorted((r for part in splits.values() for r in part
                   if f"{r['reciter_key']}:{r['surah']}:{r['ayah']}" not in flagged),
                  key=lambda r: (r["reciter_key"], r["surah"], r["ayah"]))
    runs, current = [], []
    for row in rows:  # Maximal runs of consecutive ayahs per reciter and surah.
        if current and (row["reciter_key"], row["surah"]) == (current[-1]["reciter_key"], current[-1]["surah"]) \
                and row["ayah"] == current[-1]["ayah"] + 1:
            current.append(row)
        else:
            if len(current) >= 2:
                runs.append(current)
            current = [row]
    if len(current) >= 2:
        runs.append(current)

    (args.output_dir / "audio").mkdir(parents=True, exist_ok=True)
    out, seen = [], set()
    for _ in range(args.passes):
        for run in runs:
            i = rng.randint(0, 1)
            while i < len(run) - 1:
                size = rng.randint(2, 4)
                group = run[i:i + size]
                i += size
                if len(group) < 2:
                    continue
                key = tuple((g["reciter_key"], g["surah"], g["ayah"]) for g in group)
                if key in seen:
                    continue
                fast = rng.random() < 0.5
                parts = []
                for g in group:
                    audio = sf.read(g["path"], dtype="float32")[0]
                    if fast:
                        audio = trim(audio)
                        parts.append(np.zeros(int(rng.uniform(0.05, 0.35) * SR), dtype="float32"))
                    parts.append(audio)
                audio = np.concatenate(parts)
                if len(audio) > MAX_SECONDS * SR:
                    continue
                seen.add(key)
                first = group[0]
                name = f"{first['reciter_key']}_{first['surah']:03d}_{first['ayah']:03d}_{len(group)}{'f' if fast else 'n'}.wav"
                sf.write(args.output_dir / "audio" / name, audio, SR, subtype="PCM_16")
                raw = " ".join(g["text_raw_uthmani"] for g in group)
                label = normalize_quran_for_asr(raw)
                # Unique ayah key per joined clip so it never collides with single-ayah labels.
                out.append({"surah": first["surah"], "ayah": 10000 + len(out), "audio_filename": name,
                            "relative_audio_path": f"audio/{name}", "text": label, "text_asr_normalized": label,
                            "text_raw_uthmani": raw, "normalized_with_harakat": normalize_with_harakat(raw),
                            "source_ayahs": [a for g in group for a in g.get("source_ayahs", [g["ayah"]])],
                            "joined_ayahs": [g["ayah"] for g in group], "source_reciter": first["reciter_key"],
                            "pauses": "shortened" if fast else "natural", "duration_seconds": round(len(audio) / SR, 3)})
    (args.output_dir / "metadata.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    minutes = sum(r["duration_seconds"] for r in out) / 60
    print(f"{len(out)} joined clips ({minutes:.0f} min) from {len(rows)} single clips; "
          f"{sum(r['pauses'] == 'shortened' for r in out)} with shortened pauses")


if __name__ == "__main__":
    main()
