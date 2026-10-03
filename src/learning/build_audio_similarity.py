"""Index real Qaloon reference clips for difficulty-aware listening distractors.

MFCC summary cosine similarity is an acoustic heuristic, not phoneme correctness,
semantic similarity or a learned Quran embedding. No Quran audio is synthesized.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[2]


def build(metadata, output):
    transform = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=20,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40})
    rows = []
    features = []
    durations = []
    for line in metadata.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        s, a = row["surah"], row["ayah"]
        source = ROOT / "web/public/quran/surahs" / f"{s:03d}.json"
        if not source.exists():
            continue
        verses = json.loads(source.read_text(encoding="utf-8"))["ayahs"]
        if not any(v["ayah"] == a and v["normalized"] == row["text_asr_normalized"] for v in verses):
            continue
        path = (metadata.parent / row["relative_audio_path"]).resolve()
        if not path.is_relative_to(metadata.parent.resolve()):
            raise ValueError("Audio path escapes the dataset")
        audio, rate = sf.read(path, dtype="float32")
        if rate != 16000 or audio.ndim != 1 or not 0.25 < len(audio) / rate <= 30 or not np.isfinite(audio).all():
            continue
        with torch.inference_mode():
            mfcc = transform(torch.from_numpy(audio)).numpy()[1:]
        features.append(np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]))
        durations.append(len(audio) / rate)
        rows.append({"id": f"{s}:{a}", "label": row["text_asr_normalized"],
                     "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if len(rows) < 3:
        raise ValueError("Need at least three aligned local recordings")
    values = np.stack(features)
    values = (values - values.mean(axis=0)) / np.maximum(values.std(axis=0), 1e-6)
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-6)
    similarities = np.clip((values @ values.T + 1) / 2, 0, 1)
    neighbors = {}
    for i, row in enumerate(rows):
        ranked = []
        for j, other in enumerate(rows):
            if row["label"] == other["label"]:
                continue
            duration_score = min(durations[i], durations[j]) / max(durations[i], durations[j])
            ranked.append({"id": other["id"], "score": round(float(0.85 * similarities[i, j] + 0.15 * duration_score), 6)})
        neighbors[row["id"]] = sorted(ranked, key=lambda v: v["score"], reverse=True)[:30]
    result = {"method": "MFCC mean/std cosine (19 coefficients), corpus-standardized; 15% duration similarity",
              "reciter": "Al-Hutafi (Qaloon)", "sample_rate": 16000,
              "warning": "Acoustic heuristic, not pronunciation or learned phonetic similarity. Review hard distractors by ear.",
              "clips": rows, "neighbors": neighbors}
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(output)
    print(f"Indexed {len(rows)} aligned recordings: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=ROOT / "src/dataset_collection/dataset_qaloon_hutafi/metadata.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "web/public/quran/audio-similarity.json")
    args = parser.parse_args()
    build(args.metadata, args.output)
