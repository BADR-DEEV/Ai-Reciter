"""Unbiased per-reciter ASR evaluation and correct-recitation false-alarm proxy."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dataset_collection"))
from qaloon_audio2text import normalize_quran_for_asr, normalize_with_harakat


def _scores(details):
    import jiwer

    refs = [row["reference"] for row in details]
    hyps = [row["prediction"] for row in details]
    alignments = [jiwer.process_words(ref, hyp) for ref, hyp in zip(refs, hyps)]
    return {
        "samples": len(details), "wer": jiwer.wer(refs, hyps), "cer": jiwer.cer(refs, hyps),
        "word_deletions": sum(a.deletions for a in alignments),
        "apparent_omission_ayah_rate": sum(a.deletions > 0 for a in alignments) / len(details),
    }


def evaluate_rows(model, processor, rows, batch_size=1,
                  label_field="text_asr_normalized"):
    import torch
    import soundfile as sf

    normalizer = normalize_with_harakat if label_field == "normalized_with_harakat" else normalize_quran_for_asr
    model.eval()
    details = []
    device = next(model.parameters()).device
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        signals = []
        for row in batch:
            audio, sr = sf.read(row["path"], dtype="float32")
            if sr != 16000 or audio.ndim != 1:
                raise ValueError(f"Expected 16 kHz mono WAV: {row['path']}")
            signals.append(audio)
        features = [processor.feature_extractor(audio, sampling_rate=16000).input_features[0]
                    for audio in signals]
        inputs = processor.feature_extractor.pad(
            [{"input_features": feature} for feature in features], return_tensors="pt")
        with torch.inference_mode():
            tokens = model.generate(inputs.input_features.to(device),
                                    language="arabic", task="transcribe", num_beams=1)
        predictions = processor.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        for row, prediction in zip(batch, predictions):
            details.append({"surah": row["surah"], "ayah": row["ayah"],
                            "reciter": row["reciter_key"],
                            "reference": normalizer(row[label_field]),
                             "prediction": normalizer(prediction)})
            if "text_seen_in_training" in row:
                details[-1]["text_seen_in_training"] = row["text_seen_in_training"]
    report = {"overall": _scores(details)}
    for reciter in sorted({row["reciter"] for row in details}):
        report[reciter] = _scores([row for row in details if row["reciter"] == reciter])
    for seen in (True, False):
        stratum = [row for row in details if row.get("text_seen_in_training") is seen]
        if stratum:
            report["seen_text" if seen else "unseen_text"] = _scores(stratum)
    if label_field == "text_asr_normalized":
        # Diagnostic only: do not rewrite training targets to force extra hamza folding.
        def without_hamza(value):
            return value.translate(str.maketrans({"ئ": "ي", "ؤ": "و", "ء": ""}))

        import jiwer
        report["hamza_insensitive_diagnostic"] = {
            "wer": jiwer.wer([without_hamza(r["reference"]) for r in details],
                             [without_hamza(r["prediction"]) for r in details]),
            "cer": jiwer.cer([without_hamza(r["reference"]) for r in details],
                             [without_hamza(r["prediction"]) for r in details]),
        }
    return report, details
