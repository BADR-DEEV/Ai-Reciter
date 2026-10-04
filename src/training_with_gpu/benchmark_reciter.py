"""Frozen blind-vs-assisted evaluation, skip probes and 28s CUDA latency.

Waleed is NEVER fitted/selected here. --partition heldout is descriptive after
prior exposure, not a new sealed test. No threshold, decoder or checkpoint sweep.
"""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.training_with_gpu.decoding_safety import load_private_adapter
from src.training_with_gpu.gpu_evaluation import evaluate_rows
from src.training_with_gpu.train_base_full import DEEPDML_BASE, DEEPDML_SMALL, MODEL_REVISIONS, configure_generation
from src.training.recitation_diagnostics import phrase_present, skipped_ayah_completion
from src.training.reviewed_audio import sha256


def build_skip_probes(rows, output, limit=12):
    """Concatenate WHOLE A/C clips, omit B; actual ASR labels remain A+C."""
    import numpy as np
    import soundfile as sf
    if limit < 1:
        raise ValueError("Positive skip probe limit required")
    indexed = {(r["reciter_key"], r["surah"], r["ayah"]): r for r in rows}
    probes, skips, provenance = [], [], []
    groups = {reader: sorted(key for key in indexed if key[0] == reader)
              for reader in sorted({r["reciter_key"] for r in rows})}
    ordered = [keys[index] for index in range(max((len(keys) for keys in groups.values()), default=0))
               for keys in groups.values() if index < len(keys)]
    for reader, surah, ayah in ordered:
        first = indexed[(reader, surah, ayah)]
        middle, last = indexed.get((reader, surah, ayah + 1)), indexed.get((reader, surah, ayah + 2))
        if not middle or not last:
            continue
        spoken = first["text_asr_normalized"] + " " + last["text_asr_normalized"]
        skipped = middle["text_asr_normalized"]
        if phrase_present(skipped, spoken):
            continue
        decoded = [sf.read(r["path"], dtype="float32") for r in (first, last)]
        if any(sr != 16000 or signal.ndim != 1 for signal, sr in decoded):
            raise ValueError("Skip fixture requires original mono 16kHz PCM")
        signals = [signal for signal, _ in decoded]
        audio = np.concatenate((signals[0], np.zeros(8000, dtype="float32"), signals[1]))
        if len(audio) > 28 * 16000:
            continue
        path = output / f"skip_{len(probes):03d}.wav"
        sf.write(path, audio, 16000, subtype="PCM_16")
        probes.append({"path": str(path), "surah": surah, "ayah": ayah, "reciter_key": reader,
            "text_asr_normalized": spoken})
        skips.append(skipped)
        provenance.append({"spoken_audio_hashes": [r["audio_sha256"] for r in (first, last)],
            "omitted_source_audio_sha256": middle["audio_sha256"], "spoken_reference": spoken,
            "constituent_text_seen_in_training": [r.get("text_seen_in_training") for r in (first, last)],
            "skipped_exercise_reference_not_ASR_label": skipped, "synthetic_pcm_sha256": sha256(path),
            "full_ayah_clips_used_no_internal_cuts": True})
        if len(probes) == limit:
            break
    (output / "skip_probe_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    return probes, skips


def latency_28s(model, processor, rows, repeats=6):
    """Timing-only concatenated voiced windows, NO training/ASR quality labels."""
    import numpy as np
    import soundfile as sf
    import torch
    audio = np.concatenate([sf.read(row["path"], dtype="float32")[0] for row in rows[:24]])
    if len(audio) < 28 * 16000:
        raise ValueError("Need at least 28s actual audio for voiced latency fixture")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    timings = []
    for i in range(repeats + 1):
        offset = (i * 16000) % (len(audio) - 28 * 16000 + 1)
        window = audio[offset:offset + 28 * 16000]
        torch.cuda.synchronize()
        started = time.perf_counter()
        inputs = processor.feature_extractor(window, sampling_rate=16000, return_attention_mask=True, return_tensors="pt")
        with torch.inference_mode():
            model.generate(inputs.input_features.to(device, dtype=dtype), attention_mask=inputs.attention_mask.to(device),
                language="arabic", task="transcribe", num_beams=3, max_length=448, use_cache=True,
                return_dict_in_generate=True)
        torch.cuda.synchronize()
        if i:
            timings.append(time.perf_counter() - started)
    return {"audio_seconds_per_window": 28, "samples": repeats, "beam": 3, "dtype": str(dtype),
        "seconds_p50": float(np.quantile(timings, .5)), "seconds_p95": float(np.quantile(timings, .95)),
        "warmup_windows_excluded": 1, "source": "concatenated-professional-clips; not-natural-recording-boundary",
        "scope": "feature-extraction+transfer+synchronized-generate; excludes model-load/network/UI",
        "quality_labels_attached_to_cropped_timing_windows": False}


def build_repeat_probes(rows, output, limit=10):
    """Two complete renditions of the SAME clip; synthetic, not natural restarts."""
    import numpy as np
    import soundfile as sf
    from src.training_with_gpu.train_tarteel_lora import balanced_subset
    eligible = [r for r in rows if r.get("surah") != 1 and
                sf.info(r["path"]).duration * 2 + .5 <= 28]
    probes, provenance = [], []
    for row in balanced_subset(eligible, limit, 42):
        audio, sr = sf.read(row["path"], dtype="float32")
        if sr != 16000 or audio.ndim != 1:
            raise ValueError("Repeat fixture requires original mono 16kHz PCM")
        path = output / f"repeat_{len(probes):03d}.wav"
        sf.write(path, np.concatenate((audio, np.zeros(8000, dtype="float32"), audio)), sr, subtype="PCM_16")
        reference = " ".join([row["text_asr_normalized"]] * 2)
        probes.append({"path": str(path), "surah": row["surah"], "ayah": row["ayah"],
            "reciter_key": row["reciter_key"], "text_asr_normalized": reference})
        provenance.append({"source_audio_sha256": row["audio_sha256"], "synthetic_pcm_sha256": sha256(path),
            "complete_clip_repetitions": 2, "actual_spoken_reference": reference,
            "not_natural_learner_restarts": True})
    (output / "repeat_probe_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    return probes


def build_nonspeech_probes(output, surah):
    """Deterministic silence/noise controls with truthful EMPTY speech labels."""
    import numpy as np
    import soundfile as sf
    rng = np.random.default_rng(42)
    probes = []
    for index, amplitude in enumerate((0.0, 1e-5, 1e-3, 1e-2)):
        signal = (rng.normal(size=28 * 16000) * amplitude).astype("float32")
        path = output / f"nonspeech_{index:03d}.wav"
        sf.write(path, signal, 16000, subtype="PCM_16")
        probes.append({"path": str(path), "surah": surah, "ayah": index,
            "reciter_key": "synthetic-nonspeech", "text_asr_normalized": ""})
    (output / "nonspeech_probe_provenance.json").write_text(json.dumps({"seed": 42, "seconds": 28,
        "nominal_noise_rms": [0.0, 1e-5, 1e-3, 1e-2], "no_quran_speech_present": True,
        "oracle_surah_for_assisted_mode": surah,
        "pcm_sha256": [sha256(r["path"]) for r in probes]}, indent=2), encoding="utf-8")
    return probes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--partition", choices=["validation", "test", "unheard_adaptation_reciter"], default="validation")
    p.add_argument("--unadapted", choices=[DEEPDML_BASE, DEEPDML_SMALL], help="Same-data pinned foundation control instead of adapter")
    p.add_argument("--assisted", action="store_true", help="Separately report oracle-surah vocabulary results; never learner grading")
    p.add_argument("--dtype", choices=["float32", "bf16", "fp16"], default="float32")
    p.add_argument("--batch-size", type=int, default=2)
    args = p.parse_args()
    if args.output_dir.exists() or args.batch_size <= 0:
        raise ValueError("New evidence directory and positive batch size required")
    manifest_path = args.run / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest["splits"][args.partition]
    version = manifest.get("normalizer_version", "qaloon-asr-v1")
    from src.dataset_collection.qaloon_audio2text import NORMALIZER_VERSION, normalizer_for_version, INPUT_JSON
    if args.assisted and version != NORMALIZER_VERSION:
        raise ValueError("Assisted vocabulary requires corrected, version-matched references; historical reports stay unchanged")
    normalize = normalizer_for_version(version)
    for row in rows:
        if sha256(row["path"]) != row["audio_sha256"]:
            raise ValueError("Frozen evaluation WAV changed")
        if row.get("text_raw_uthmani") and normalize(row["text_raw_uthmani"]) != row["text_asr_normalized"]:
            raise ValueError("Frozen reference normalization differs from original source/version")
    import torch
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "float32": torch.float32}[args.dtype]
    if not torch.cuda.is_available() or args.dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Required CUDA/BF16 support absent")
    args.output_dir.mkdir(parents=True)
    if args.unadapted:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        revision = MODEL_REVISIONS[args.unadapted]
        processor = WhisperProcessor.from_pretrained(args.unadapted, revision=revision, language="arabic", task="transcribe", trust_remote_code=False)
        processor.tokenizer.set_prefix_tokens(language="arabic", task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained(args.unadapted, revision=revision, trust_remote_code=False, torch_dtype=dtype).to("cuda").eval()
        configure_generation(model, processor)
    else:
        model, processor = load_private_adapter(args.run / "adapter", dtype=dtype)
    report = {"frozen_manifest_sha256": sha256(manifest_path), "partition": args.partition,
        "initialization_or_adapter": args.unadapted or str(args.run / "adapter"),
        "upstream_exposure_unknown": True, "normalizer_version": version,
        "no_fit_or_checkpoint_selection": True, "production_approved": False,
        "comparison_to_photo": "screenshot-only; exact preprocessing/definitions/precision/split unavailable"}
    report["evidence_fingerprints"] = {"canonical_source_sha256": sha256(INPUT_JSON),
        "normalizer_code_sha256": sha256(ROOT / "src/dataset_collection/qaloon_audio2text.py"),
        "benchmark_code_sha256": sha256(__file__), "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "inference_dtype": args.dtype}
    if not args.unadapted:
        report["evidence_fingerprints"].update({"adapter_weights_sha256": sha256(args.run / "adapter/adapter_model.safetensors"),
            "adapter_config_sha256": sha256(args.run / "adapter/adapter_config.json"),
            "saved_generation_sha256": sha256(args.run / "adapter/generation_config.json")})
    def persist():
        (args.output_dir / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    modes = [False, True] if args.assisted else [False]
    for assisted in modes:
        name = "assisted_surah_vocabulary" if assisted else "official_blind"
        metrics, _ = evaluate_rows(model, processor, rows, args.batch_size, decode_profile="beam3", normalizer_version=version,
            surah_constraint=assisted,
            predictions_path=args.output_dir / f"{name}_predictions.jsonl")
        report[name] = metrics
        persist()
        print(name, metrics["overall"], metrics["qaloon_lexical_fidelity"]["eligible_occurrences"], flush=True)
    # Skip probes only use development audio, even when the final Waleed partition
    # is scored. They are a fixed diagnostic, never an unseen learner test.
    probe_rows = manifest["splits"]["validation"]
    for row in probe_rows:
        if sha256(row["path"]) != row["audio_sha256"]:
            raise ValueError("Frozen development probe PCM changed")
    probes, skips = build_skip_probes(probe_rows, args.output_dir)
    if probes:
        for assisted in modes:
            name = "assisted_surah_vocabulary" if assisted else "official_blind"
            metrics, details = evaluate_rows(model, processor, probes, args.batch_size, decode_profile="beam3", normalizer_version=version,
                surah_constraint=assisted,
                predictions_path=args.output_dir / f"{name}_skip_predictions.jsonl")
            report[name + "_skip_probe"] = {"spoken_text_asr": metrics, "completion": skipped_ayah_completion(details, skips)}
            persist()
    repeats = build_repeat_probes(probe_rows, args.output_dir)
    nonspeech = build_nonspeech_probes(args.output_dir, probe_rows[0]["surah"])
    for assisted in modes:
        name = "assisted_surah_vocabulary" if assisted else "official_blind"
        if repeats:
            metrics, _ = evaluate_rows(model, processor, repeats, args.batch_size, decode_profile="beam3", normalizer_version=version,
                surah_constraint=assisted, predictions_path=args.output_dir / f"{name}_repeat_predictions.jsonl")
            report[name + "_repeat_probe"] = {"actual_repeated_text_asr": metrics,
                "scope": "synthetic-two-complete-clips; natural-restarts/unfamiliar-learners-unassessed"}
            persist()
        _, details = evaluate_rows(model, processor, nonspeech, args.batch_size, decode_profile="beam3", normalizer_version=version,
            surah_constraint=assisted, predictions_path=args.output_dir / f"{name}_nonspeech_predictions.jsonl")
        report[name + "_nonspeech_probe"] = {"probes": len(details),
            "nonempty_outputs": sum(bool(r["prediction"]) for r in details),
            "nonempty_output_rate": sum(bool(r["prediction"]) for r in details) / len(details),
            "generated_word_count": sum(len(r["prediction"].split()) for r in details),
            "unflagged_nonempty_outputs": sum(r["scorable"] and bool(r["prediction"]) for r in details),
            "scope": "raw-model-no-speech-hallucinations; no streaming RMS gate; synthetic not-real-background-noise"}
        persist()
    report["latency_28s"] = latency_28s(model, processor, probe_rows)
    report["actual_gpu"] = {"name": torch.cuda.get_device_name(), "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9}
    report["status"] = "complete"
    persist()
    print("Benchmark complete:", args.output_dir, report["latency_28s"], flush=True)


if __name__ == "__main__":
    main()
