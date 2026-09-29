"""Shared, reproducible Qālūn metadata and Whisper evaluation utilities."""

import hashlib
import json
import sys
import wave
from pathlib import Path


DATA_ROOT = Path(__file__).resolve().parents[1] / "dataset_collection"
RECITER_DIRS = {
    "huthaify": "dataset_qaloon_hutafi",
    "husary": "dataset_qaloon_Husary",
    "dokali": "dataset_qaloon_dokali",
    "waleed": "dataset_qaloon_waleed",
}


def selected_reciters(include, exclude):
    unknown = (set(include or []) | set(exclude or [])) - RECITER_DIRS.keys()
    if unknown:
        raise ValueError(f"Unknown reciters: {sorted(unknown)}")
    selected = set(include or RECITER_DIRS) - set(exclude or [])
    if not selected:
        raise ValueError("No reciters selected")
    return sorted(selected)


def split_name(surah, ayah, seed=42):
    """Group Fatiha as one unit: Waleed includes its basmalah and joins 6+7."""
    key = f"{surah}:fatiha" if surah == 1 else f"{surah}:{ayah}"
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    return "test" if fraction < 0.1 else "validation" if fraction < 0.2 else "train"


def load_splits(data_root=DATA_ROOT, include=None, exclude=None, seed=42,
                max_seconds=30.0, include_bismillah=False):
    root = Path(data_root)
    splits = {name: [] for name in ("train", "validation", "test")}
    skipped = []
    labels_by_ayah = {}
    for reciter in selected_reciters(include, exclude):
        folder = root / RECITER_DIRS[reciter]
        with (folder / "metadata.jsonl").open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                if row["surah"] == 1 and (row["ayah"] == 0 or row.get("source_ayahs") == [0]) and not include_bismillah:
                    continue
                path = folder / row["relative_audio_path"]
                if not path.is_file():
                    raise FileNotFoundError(f"{folder / 'metadata.jsonl'}:{line_number}: {path}")
                if not row.get("text_asr_normalized"):
                    raise ValueError(f"Missing ASR label: {reciter} {row['surah']}:{row['ayah']}")
                if row["surah"] != 1:
                    key = (row["surah"], row["ayah"])
                    previous = labels_by_ayah.setdefault(key, row["text_asr_normalized"])
                    if previous != row["text_asr_normalized"]:
                        raise ValueError(f"Conflicting Qaloon labels for {key}: {previous} / {row['text_asr_normalized']}")
                with wave.open(str(path), "rb") as wav:
                    if wav.getframerate() != 16000 or wav.getnchannels() != 1:
                        raise ValueError(f"Expected mono 16 kHz WAV: {path}")
                    wav_seconds = wav.getnframes() / wav.getframerate()
                if row.get("start_time") is not None and row.get("end_time") is not None:
                    seconds = (row["end_time"] - row["start_time"]) / 1000
                    if abs(seconds - wav_seconds) > 0.05:
                        raise ValueError(f"Metadata/WAV duration mismatch: {path}")
                else:
                    seconds = wav_seconds
                if seconds <= 0 or seconds > max_seconds:
                    skipped.append((reciter, row["surah"], row["ayah"], seconds))
                    continue  # Never truncate audio while retaining the full transcription.
                sample = {**row, "path": str(path), "duration_seconds": seconds,
                          "reciter_key": reciter}
                splits[split_name(row["surah"], row["ayah"], seed)].append(sample)
    for name, rows in splits.items():
        if not rows:
            raise ValueError(f"Empty {name} split; check reciter selection and metadata")
    return splits, skipped


def audio_features(processor, path):
    """Read validated 16 kHz mono PCM clips without implicit resampling."""
    import soundfile as sf

    audio, sample_rate = sf.read(path, dtype="float32")
    if sample_rate != 16000 or audio.ndim != 1:
        raise ValueError(f"Expected mono 16 kHz WAV: {path} ({sample_rate} Hz, {audio.shape})")
    return processor.feature_extractor(audio, sampling_rate=16000).input_features[0]


def score_predictions(references, hypotheses):
    from jiwer import cer, wer

    if len(references) != len(hypotheses) or not references:
        raise ValueError("Expected nonempty, equal-length references and predictions")
    return {"wer": wer(references, hypotheses), "cer": cer(references, hypotheses)}


def evaluate_model(model, processor, rows, batch_size=4):
    """Greedy Arabic transcription, scored against the stored ASR labels."""
    import torch
    sys.path.insert(0, str(DATA_ROOT))
    from qaloon_audio2text import normalize_quran_for_asr

    model.eval()
    device = next(model.parameters()).device
    references, hypotheses, details = [], [], []
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset + batch_size]
        features = [audio_features(processor, row["path"]) for row in batch]
        padded = processor.feature_extractor.pad(
            [{"input_features": feature} for feature in features], return_tensors="pt"
        )
        with torch.inference_mode():
            tokens = model.generate(
                input_features=padded.input_features.to(device),
                language="arabic", task="transcribe", num_beams=1,
            )
        predictions = processor.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        for row, prediction in zip(batch, predictions):
            reference = row["text_asr_normalized"].strip()
            hypothesis = normalize_quran_for_asr(prediction)
            references.append(reference)
            hypotheses.append(hypothesis)
            details.append({"surah": row["surah"], "ayah": row["ayah"],
                            "reciter": row["reciter_key"], "reference": reference,
                            "prediction": hypothesis})
    return score_predictions(references, hypotheses), details
