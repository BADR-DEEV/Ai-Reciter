"""Unbiased per-reciter ASR evaluation and correct-recitation false-alarm proxy."""

import sys
import time
import hashlib
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
                  label_field="text_asr_normalized", decode_profile="greedy", predictions_path=None,
                  normalizer_version=None, surah_constraint=False):
    import torch
    import soundfile as sf
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.training_with_gpu.decoding_safety import DECODE_PROFILES, generation_diagnostics
    if not rows or batch_size < 1 or decode_profile not in DECODE_PROFILES:
        raise ValueError("Nonempty rows, positive batch size and known decoding profile required")
    from src.dataset_collection.qaloon_audio2text import NORMALIZER_VERSION
    if normalizer_version is None:
        versions = {row.get("normalizer_version", "qaloon-asr-v1") for row in rows}
        if len(versions) != 1:
            raise ValueError("Mixed label-normalizer versions; provide explicit matched references")
        normalizer_version = versions.pop()
    if surah_constraint and (normalizer_version != NORMALIZER_VERSION or label_field != "text_asr_normalized"):
        raise ValueError("Current surah-word constraints require version-matched unvowelled references")
    if predictions_path is not None:
        # Exclusive creation preserves partial prior results. Flush each batch so
        # native/process failures cannot erase the entire evaluation's evidence.
        with Path(predictions_path).open("x", encoding="utf-8"):
            pass

    from src.dataset_collection.qaloon_audio2text import normalizer_for_version
    normalizer = normalize_with_harakat if label_field == "normalized_with_harakat" else normalizer_for_version(normalizer_version)
    constraint_cache = {}
    model.eval()
    details = []
    latencies, audio_seconds = [], 0.0
    device = next(model.parameters()).device
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        signals = []
        for row in batch:
            audio, sr = sf.read(row["path"], dtype="float32")
            if (sr != 16000 or audio.ndim != 1 or not len(audio)
                    or len(audio) > 30 * sr or not np.isfinite(audio).all()):
                raise ValueError(f"Expected nonempty finite <=30s 16 kHz mono WAV; never silently truncate: {row['path']}")
            signals.append(audio)
        inputs = processor.feature_extractor(signals, sampling_rate=16000,
                                             return_attention_mask=True, return_tensors="pt")
        constrained_kwargs = {}
        if surah_constraint:
            from src.training_with_gpu.surah_vocabulary import SurahVocabularyConstraint
            for row in batch:
                if row["surah"] not in constraint_cache:
                    constraint_cache[row["surah"]] = SurahVocabularyConstraint.from_canonical(processor.tokenizer, row["surah"])
            constrained_kwargs["prefix_allowed_tokens_fn"] = lambda batch_id, ids: constraint_cache[batch[batch_id]["surah"]].allowed(ids.tolist())
        with torch.inference_mode():
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            generated = model.generate(inputs.input_features.to(device, dtype=next(model.parameters()).dtype),
                                    attention_mask=inputs.attention_mask.to(device),
                                    language="arabic", task="transcribe", use_cache=True,
                                    return_dict_in_generate=True,
                                    max_length=model.config.max_target_positions, **DECODE_PROFILES[decode_profile], **constrained_kwargs)
            # Whisper's tensor-only API strips prefix/EOS during postprocessing.
            # ModelOutput preserves them; otherwise EOS/loop audits are invalid.
            tokens = generated.sequences
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latencies.append(time.perf_counter() - started)
            audio_seconds += sum(len(signal) / 16000 for signal in signals)
        predictions = processor.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        if len(predictions) != len(batch):
            raise ValueError("Decoder output count differs from frozen batch; refuse incomplete metrics")
        for index, (row, prediction) in enumerate(zip(batch, predictions)):
            details.append({"surah": row["surah"], "ayah": row["ayah"],
                            "reciter": row["reciter_key"],
                              "normalizer_version": normalizer_version,
                              "decode_profile": decode_profile, "expected_surah_supplied": surah_constraint,
                            "reference": normalizer(row[label_field]),
                              "prediction": normalizer(prediction), "raw_prediction": prediction,
                              "duration_seconds": len(signals[index]) / 16000,
                              "audio_sha256": hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest(),
                              **generation_diagnostics(tokens[index].tolist(), processor.tokenizer, model.config.max_target_positions)})
            if "text_seen_in_training" in row:
                details[-1]["text_seen_in_training"] = row["text_seen_in_training"]
            if surah_constraint and details[-1]["scorable"]:
                allowed_words = constraint_cache[row["surah"]].canonical_normalized_vocabulary
                if any(word not in allowed_words for word in details[-1]["prediction"].split()):
                    raise ValueError("Completed constrained output violated canonical word grammar")
        if predictions_path is not None:
            import json
            with Path(predictions_path).open("a", encoding="utf-8") as output:
                output.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in details[-len(batch):]))
    report = summarize_predictions(details, decode_profile, label_field)
    report["normalizer_version"] = normalizer_version
    report["decoding_mode"] = "surah-vocabulary-assisted-NOT-omission-grading" if surah_constraint else "blind-acoustic-ASR"
    report["expected_surah_supplied"] = surah_constraint
    report["qaloon_lexical_fidelity"]["constraint_can_force_qaloon_spelling"] = surah_constraint
    if surah_constraint:
        report["constraint"] = {"source": "QaloonData_v10(1).json; no expected ayah sequence",
            "spelling_policy": "canonical words + source-attested and exact-normalizer-equivalent alif-carrier forms",
            "initial_leading_space_supported": True, "order_repeats_and_skips_unrestricted": True,
            "complete_scorable_outputs_checked_for_canonical_normalized_words": True,
            "can_invent_absent_canonical_words": True}
    report["performance"] = {"batch_size": batch_size, "inference_seconds": sum(latencies), "audio_seconds": audio_seconds,
        "inference_real_time_factor": sum(latencies) / max(audio_seconds, 1e-9),
        "batch_inference_latency_ms_p50": float(np.quantile(latencies, .5) * 1000),
        "batch_inference_latency_ms_p95": float(np.quantile(latencies, .95) * 1000),
        "scope": "Synchronized generation-only batch latency; excludes file loading, feature extraction and network latency."}
    return report, details


def summarize_predictions(details, decode_profile="greedy", label_field="text_asr_normalized"):
    """Score preserved raw predictions without inventing lost timing measurements."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.training_with_gpu.decoding_safety import DECODE_PROFILES
    if not details or decode_profile not in DECODE_PROFILES:
        raise ValueError("Nonempty predictions and supported decoder required")
    report = {"overall": _scores(details)}
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.training.asr_metrics import report_predictions
    report["diagnostics"] = report_predictions(details)
    from src.training.recitation_diagnostics import lexical_fidelity
    report["qaloon_lexical_fidelity"] = lexical_fidelity(details)
    report["decoding_safety"] = {"profile": decode_profile, "kwargs": DECODE_PROFILES[decode_profile],
        "flagged_samples": sum(not r["scorable"] for r in details),
        "flag_rate": sum(not r["scorable"] for r in details) / len(details),
        "coverage": sum(r["scorable"] for r in details) / len(details),
        "eos_missing_samples": sum(not r["eos_emitted"] for r in details),
        "sustained_token_cycles": sum("sustained_token_cycle" in r["decode_flags"] for r in details),
        "sustained_word_cycles": sum("sustained_word_cycle" in r["decode_flags"] for r in details),
        "official_wer_includes_all_flagged_outputs": True,
        "learner_feedback": "flagged outputs must abstain, not report Quran mistakes",
        "unflagged_outputs_not_certified_correct": True}
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
    return report
