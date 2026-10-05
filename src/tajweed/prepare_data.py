"""Prepare tajweed-model data: plain-reading negatives and the held-out test voice.

A model trained only on correct recitation learns where rules belong from the
text and then "hears" them everywhere (the Evaluating-ASR report measured a
115% tajweed error rate on plain audio). Negatives fix that:

1. Plain readings: MMS Arabic TTS (facebook/mms-tts-ara, CC-BY-NC-4.0) reads
   every ayah without tajweed, in several takes with different seeds and
   speaking rates. A take is kept only when rattil-v4 recognises its words
   (WER <= --max-wer). Rows carry "tajweed": false, so targets.py gives them
   targets with no tajweed tokens.
2. Sheikh Waleed (never trained on): a copy with labels from the current
   normalizer, for testing on an unseen voice. Al-Fātiḥah is left out because
   its labels are shifted by one ayah.

    PYTHONPATH=src/deployment/mac_shim python -m src.tajweed.prepare_data \\
        --reader-root data/hf/qaloon-all-reciters \\
        --waleed data/hf/qaloon-reciter-experiments/dataset_qaloon_waleed --out data/tajweed
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf

from src.dataset_collection.qaloon_audio2text import normalize_quran_for_asr

ROOT = Path(__file__).resolve().parents[2]
RATES = (0.9, 1.0, 1.12)


def corpus_ayahs(reader_roots):
    """{(surah, ayah): (text_raw_uthmani, text_asr_normalized)} from the readers' metadata."""
    ayahs = {}
    for root in reader_roots:
        for meta in sorted(Path(root).glob("dataset_qaloon_*/metadata.jsonl")):
            for line in meta.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row.get("audit_flagged") or row["ayah"] == 0 or len(row.get("source_ayahs") or [1]) != 1:
                    continue
                ayahs.setdefault((row["surah"], row["ayah"]), (row["text_raw_uthmani"], row["text_asr_normalized"]))
    return ayahs


def synthesize(ayahs, out, takes, seed):
    import torch
    from transformers import AutoTokenizer, VitsModel
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ara")
    model = VitsModel.from_pretrained("facebook/mms-tts-ara").eval()
    (out / "audio").mkdir(parents=True, exist_ok=True)
    rows = []
    for n, ((surah, ayah), (raw, label)) in enumerate(sorted(ayahs.items())):
        for take in range(takes):
            torch.manual_seed(seed + 1000 * n + take)
            model.speaking_rate = RATES[take % len(RATES)]
            with torch.no_grad():
                wave = model(**tokenizer(label, return_tensors="pt")).waveform[0].numpy()
            wave = (0.9 * wave / max(1e-6, float(np.abs(wave).max()))).astype(np.float32)
            name = f"audio/{surah:03d}{ayah:03d}_{take}.wav"
            sf.write(out / name, wave, model.config.sampling_rate, subtype="PCM_16")
            rows.append({"surah": surah, "ayah": ayah, "source_ayahs": [ayah], "relative_audio_path": name,
                         "text_raw_uthmani": raw, "text_asr_normalized": label, "reciter": "MMS-TTS plain reading (no tajweed)",
                         "reciter_key": "ttsplain", "tajweed": False, "duration_seconds": round(len(wave) / model.config.sampling_rate, 3),
                         "tts_rate": model.speaking_rate, "quality_flags": []})
        if n % 50 == 0:
            print(f"  synthesized {n + 1}/{len(ayahs)} ayahs", flush=True)
    return rows


def recognise(rows, folder, model_dir, batch=16):
    """Word error of rattil-v4 on each take (the takes it cannot read are dropped)."""
    import jiwer
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir).to(device).eval()
    for start in range(0, len(rows), batch):
        part = rows[start:start + batch]
        audio = [sf.read(folder / r["relative_audio_path"], dtype="float32")[0] for r in part]
        features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            ids = model.generate(features, language="arabic", task="transcribe", max_new_tokens=200)
        for row, text in zip(part, processor.batch_decode(ids, skip_special_tokens=True)):
            row["v4_transcript"] = normalize_quran_for_asr(text)
            row["v4_wer"] = round(jiwer.wer(row["text_asr_normalized"], row["v4_transcript"] or "-"), 3)
    return rows


def waleed_copy(source, out, labels):
    folder = out / "dataset_qaloon_waleed"
    folder.mkdir(parents=True, exist_ok=True)
    if not (folder / "audio").exists():
        os.symlink(Path(source).resolve() / "audio", folder / "audio")
    rows, skipped = [], 0
    for line in (Path(source) / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        label = normalize_quran_for_asr(row["text_raw_uthmani"])
        expected = labels.get((row["surah"], row["ayah"]), (None, label))[1]
        if row["surah"] == 1 or row["ayah"] == 0 or label != expected:
            skipped += 1
            continue
        rows.append({**row, "text_asr_normalized": label, "reciter_key": "waleed"})
    (folder / "metadata.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"Waleed: {len(rows)} clips ({skipped} skipped: al-Fātiḥah or numbering mismatch)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reader-root", type=Path, action="append", required=True)
    parser.add_argument("--waleed", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "data/tajweed")
    parser.add_argument("--takes", type=int, default=3, help="TTS takes per ayah (different seed and speaking rate)")
    parser.add_argument("--max-wer", type=float, default=0.25)
    parser.add_argument("--model", type=Path, default=ROOT / "runs/rattil_qaloon_v4")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    ayahs = corpus_ayahs(args.reader_root)
    folder = args.out / "dataset_qaloon_ttsplain"
    print(f"{len(ayahs)} ayahs; {args.takes} plain takes each → {folder}")
    rows = recognise(synthesize(ayahs, folder, args.takes, args.seed), folder, args.model)
    kept = [r for r in rows if r["v4_wer"] <= args.max_wer]
    for r in rows:
        if r not in kept:
            (folder / r["relative_audio_path"]).unlink(missing_ok=True)
    (folder / "metadata.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
    print(f"Plain negatives: kept {len(kept)}/{len(rows)} takes with v4 WER <= {args.max_wer} "
          f"(mean WER of kept {np.mean([r['v4_wer'] for r in kept]):.3f})")
    if args.waleed:
        waleed_copy(args.waleed, args.out, ayahs)


if __name__ == "__main__":
    main()
