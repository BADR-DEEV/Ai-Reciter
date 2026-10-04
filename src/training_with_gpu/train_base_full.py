"""Full Whisper base/tiny Qaloon adaptation with pinned OpenAI/Tarteel starts."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
sys.path.insert(0, str(TRAINING_DIR))
from qaloon_data import DATA_ROOT, RECITER_DIRS, load_splits, speaker_disjoint_splits

TARTEEL_MODEL = "tarteel-ai/whisper-base-ar-quran"
TARTEEL_REVISION = "5c3c53fdf9272c4f6ee0bee09a1e5a4a615ee25c"
OPENAI_REVISION = "e37978b90ca9030d5170a5c07aadb050351a65bb"
TARTEEL_TINY_MODEL = "tarteel-ai/whisper-tiny-ar-quran"
DEEPDML_BASE = "deepdml/whisper-base-ar-quran-mix-norm"
DEEPDML_SMALL = "deepdml/whisper-small-ar-quran-mix-norm"
MODEL_REVISIONS = {
    DEEPDML_BASE: "23758f3e6877ce46077962c2e31a14794e700f1f",
    DEEPDML_SMALL: "2d0929ee4d62cf3642e4184c16e9cb939d86ea78",
    TARTEEL_MODEL: TARTEEL_REVISION,
    "openai/whisper-base": OPENAI_REVISION,
    TARTEEL_TINY_MODEL: "c3d7e624af5c81bef25a10a2af3a5a84cb4dd0f0",
    "openai/whisper-tiny": "169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
}


def configure_generation(model, processor, template=None):
    """Upgrade legacy decoding metadata only; never substitute model weights."""
    if template is None:
        from transformers import GenerationConfig
        template = GenerationConfig.from_pretrained("openai/whisper-base", revision=OPENAI_REVISION)
    vocabulary_ids = set(processor.tokenizer.get_vocab().values())
    if not vocabulary_ids or min(vocabulary_ids) < 0 or max(vocabulary_ids) >= model.config.vocab_size:
        raise ValueError("Checkpoint/processor vocabulary IDs exceed model embeddings")
    for token, expected in (("<|ar|>", template.lang_to_id["<|ar|>"]),
                            ("<|transcribe|>", template.task_to_id["transcribe"]),
                            ("<|notimestamps|>", template.no_timestamps_token_id)):
        if processor.tokenizer.convert_tokens_to_ids(token) != expected:
            raise ValueError(f"Modern decoding template/processor token mismatch: {token}")
    if max(vocabulary_ids) + 1 != model.config.vocab_size:
        # This 2022 Tarteel processor contains all text/control tokens, but omits
        # the 1,501 timestamp tokens at the tail of Whisper's output vocabulary.
        # Keep its matching text tokenizer and suppress ONLY those absent IDs.
        # Any other mismatch fails; never resize/reinitialize trained embeddings.
        if (max(vocabulary_ids) != template.no_timestamps_token_id
                or vocabulary_ids != set(range(template.no_timestamps_token_id + 1))
                or model.config.vocab_size - len(vocabulary_ids) != 1501):
            raise ValueError("Unsupported checkpoint/processor vocabulary mismatch")
        template.suppress_tokens = sorted(set(template.suppress_tokens or []) | set(range(len(vocabulary_ids), model.config.vocab_size)))
    model.generation_config = template
    model.generation_config.language = "arabic"
    model.generation_config.task = "transcribe"
    model.generation_config.return_timestamps = False
    model.generation_config.max_length = model.config.max_target_positions
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    # Prevent save_pretrained from copying the legacy max_length=1024 back into
    # the modern generation config. 20 is the deprecated config-field default;
    # the ACTUAL decoding maximum lives above in generation_config (448).
    model.config.max_length = 20


def parse_args(argv=None, *, default_model="openai/whisper-base", default_reciters=None, default_output="runs/gpu_base_full", default_dokali_weight=.35):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--output-dir", type=Path, default=Path(default_output))
    p.add_argument("--init-model", choices=MODEL_REVISIONS, default=default_model)
    p.add_argument("--model-revision", help="Override pinned initialization revision deliberately")
    p.add_argument("--reciters", nargs="+", choices=RECITER_DIRS, default=default_reciters)
    p.add_argument("--trabulsi-reviewed-manifest", type=Path)
    p.add_argument("--trabulsi-permission-file", type=Path)
    p.add_argument("--trabulsi-validation-report", type=Path, help="Complete strict WAV/text audit of the corrected dataset")
    p.add_argument("--dokali-weight", type=float, default=default_dokali_weight)
    p.add_argument("--noise-prob", type=float, default=0.12)
    p.add_argument("--speed-prob", type=float, default=0.40, help="Probability of speed/tempo perturbation")
    p.add_argument("--tempo-min", type=float, default=.95)
    p.add_argument("--tempo-max", type=float, default=1.05)
    p.add_argument("--label-field", default="text_asr_normalized")
    p.add_argument("--epochs", type=int, default=7)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1.25e-5, help="Standard safe LR for full fine-tuning")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--validation-reciter", choices=RECITER_DIRS)
    p.add_argument("--test-reciter", choices=RECITER_DIRS)
    p.add_argument("--dry-run", action="store_true", help="Validate splits without loading a model or training")
    p.add_argument("--max-steps", type=int, default=-1, help="Positive value for an explicitly limited smoke experiment")
    p.add_argument("--max-samples-per-split", type=int, help="Stratified smoke subset; not a benchmark")
    args = p.parse_args(argv)
    if any((args.trabulsi_reviewed_manifest, args.trabulsi_permission_file, args.trabulsi_validation_report)) and not all((args.trabulsi_reviewed_manifest, args.trabulsi_permission_file, args.trabulsi_validation_report)):
        p.error("Pass reviewed Trabulsi manifest, source permission receipt AND strict validation report")
    if (not math.isfinite(args.dokali_weight) or not 0 < args.dokali_weight or not 0 <= args.noise_prob <= 1
            or not 0 <= args.speed_prob <= 1 or not 0 < args.tempo_min <= args.tempo_max < float("inf")):
        p.error("Require positive Dokali weight and augmentation probabilities in [0,1]")
    if args.max_samples_per_split is not None and args.max_samples_per_split <= 0:
        p.error("max-samples-per-split must be positive")
    if (min(args.batch_size, args.eval_batch_size, args.gradient_accumulation, args.epochs, args.patience) <= 0
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or args.max_steps == 0 or args.max_steps < -1):
        p.error("Training sizes/epochs/patience/LR must be positive; max-steps is -1 or positive")
    return args


def tempo_perturb(audio, sample_rate, rate):
    """Pitch-preserving phase-vocoder tempo; retain all content, never crop."""
    target_length = round(len(audio) / rate)
    if target_length > sample_rate * 30:
        return audio.copy()  # Reject augmentation, not the terminal recitation.
    tensor = torch.from_numpy(audio).float()
    fft, hop = 512, 128
    window = torch.hann_window(fft)
    spectrum = torch.stft(tensor, n_fft=fft, hop_length=hop, window=window, return_complex=True)
    phase = torch.linspace(0, math.pi * hop, spectrum.shape[-2]).unsqueeze(-1)
    stretched = F.phase_vocoder(spectrum, rate, phase)
    return torch.istft(stretched, n_fft=fft, hop_length=hop, window=window,
                       length=target_length).numpy().astype("float32")


def add_snr_noise(audio, snr_db):
    signal_power = float(np.mean(audio.astype(np.float64) ** 2))
    if signal_power == 0:
        return audio.copy()
    noise = np.random.normal(size=audio.shape)
    noise *= math.sqrt(signal_power / (10 ** (snr_db / 10) * float(np.mean(noise ** 2))))
    mixed = audio + noise
    # Avoid introducing saturation; a common gain preserves relative SNR.
    peak = float(np.max(np.abs(mixed)))
    if peak > .999:
        mixed *= .999 / peak
    return mixed.astype("float32")


class AugmentedAyahDataset:
    """In-memory cached dataset with dynamic speed and noise augmentation."""
    def __init__(self, rows, processor, label_field, noise_prob=0.0, speed_prob=0.0, tempo_range=None):
        self.processor = processor
        self.label_field = label_field
        self.noise_prob = noise_prob
        self.speed_prob = speed_prob
        self.tempo_range = tempo_range
        if tempo_range is not None and not 0 < tempo_range[0] <= tempo_range[1]:
            raise ValueError("Positive ordered tempo range required")
        self.cached_samples = []

        print(f"Pre-caching {len(rows)} audio files into RAM...")
        for row in rows:
            audio, sr = sf.read(row["path"], dtype="float32")
            if sr != 16000 or audio.ndim != 1:
                raise ValueError(f"Expected 16 kHz mono WAV: {row['path']}")
            if not len(audio) or len(audio) > sr * 30 or not np.isfinite(audio).all():
                raise ValueError(f"Invalid or >30s audio (never silently truncate): {row['path']}")
            
            labels = self.processor.tokenizer(row[self.label_field]).input_ids
            if len(labels) > 448:
                raise ValueError(f"Whisper target exceeds 448-token decoder context: {row['path']}")
            self.cached_samples.append({
                "audio": audio,
                "sr": sr,
                "labels": labels,
                "reciter_key": row.get("reciter_key", "")
            })
        print("RAM pre-caching complete.")

    def __len__(self):
        return len(self.cached_samples)

    def __getitem__(self, index):
        sample = self.cached_samples[index]
        audio = sample["audio"].copy()
        sr = sample["sr"]

        # 1. Dynamic Speed / Tempo Perturbation (0.92x to 1.08x)
        if self.speed_prob and np.random.random() < self.speed_prob:
            if self.tempo_range is not None:
                candidate = tempo_perturb(audio, sr, np.random.uniform(*self.tempo_range))
            else:
                # Retained solely for reproduction of historical recipes.
                speed_factor = np.random.uniform(0.92, 1.08)
                candidate = F.resample(torch.from_numpy(audio).float(), int(sr * speed_factor), sr).numpy()
            # Slowing a 29s clip can make it >30s. Do not silently truncate its
            # end while keeping the full transcript: reject that augmentation.
            if len(candidate) <= sr * 30:
                audio = candidate

        # 2. Mild Acoustic Noise (25 to 35 dB SNR)
        if self.noise_prob and np.random.random() < self.noise_prob:
            snr = np.random.uniform(25, 35)
            audio = add_snr_noise(audio, snr)

        extracted = self.processor.feature_extractor(audio, sampling_rate=sr, return_attention_mask=True)
        return {
            "input_features": extracted.input_features[0],
            "attention_mask": extracted.attention_mask[0],
            "labels": sample["labels"],
        }


def main(argv=None, **defaults):
    args = parse_args(argv, **defaults)
    readers = set(args.reciters or ["huthaify", "dokali", "husary"])
    readers.update(r for r in (args.validation_reciter, args.test_reciter) if r)
    splits, skipped = load_splits(args.data_root, sorted(readers), None, args.seed)
    if args.trabulsi_reviewed_manifest:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src.training.reviewed_audio import load_reviewed_trabulsi
        reviewed = load_reviewed_trabulsi(args.trabulsi_reviewed_manifest, args.trabulsi_permission_file, args.seed,
                                         validation_report=args.trabulsi_validation_report)
        for partition in splits:
            splits[partition].extend(reviewed[partition])
    if bool(args.validation_reciter) != bool(args.test_reciter):
        raise ValueError("Pass both --validation-reciter and --test-reciter for speaker-disjoint training")
    if args.test_reciter:
        splits = speaker_disjoint_splits(splits, args.validation_reciter, args.test_reciter)
    if any(row["reciter_key"] == "waleed" for row in splits["train"]):
        raise ValueError("Waleed is adaptation-held-out and must never enter fitting")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.training.label_integrity import corrected_label_overlay
    from src.dataset_collection.qaloon_audio2text import normalize_quran_for_asr, NORMALIZER_VERSION
    label_changes = []
    for name in splits:
        splits[name], changes = corrected_label_overlay(splits[name])
        label_changes.extend(changes)
    if args.max_samples_per_split:
        # Round-robin readers for a smoke run; don't accidentally smoke-test only
        # the first reader in alphabetically ordered metadata.
        for name, rows in splits.items():
            groups = {reader: [r for r in rows if r["reciter_key"] == reader] for reader in sorted({r["reciter_key"] for r in rows})}
            subset = []
            while groups and len(subset) < args.max_samples_per_split:
                for reader in list(groups):
                    subset.append(groups[reader].pop(0))
                    if not groups[reader]:
                        del groups[reader]
                    if len(subset) == args.max_samples_per_split:
                        break
            splits[name] = subset
    counts = {k: len(v) for k, v in splits.items()}
    protocol = ("speaker-disjoint-adaptation-pretraining-overlap-unknown" if args.test_reciter else
                "mixed-existing-ayah-split-Trabulsi-recording-groups-not-unseen-ayah" if args.trabulsi_reviewed_manifest else "known-voice-unseen-ayah")
    print("Split counts:", counts, "Excluded >30s:", skipped)
    print("Reciters:", {k: sorted({row["reciter_key"] for row in rows}) for k, rows in splits.items()})
    print("Initialization:", args.init_model, "revision:", args.model_revision or MODEL_REVISIONS[args.init_model])
    if args.dry_run:
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Choose a NEW empty output directory; never overwrite existing runs/reviewer work")

    from torch.utils.data import WeightedRandomSampler
    from train_qaloon_lora import WhisperCollator
    from transformers import (EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments,
                              WhisperForConditionalGeneration, WhisperProcessor, set_seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")
    print("GPU:", torch.cuda.get_device_name(0), "VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), "GB")

    model_id = args.init_model
    revision = args.model_revision or MODEL_REVISIONS[model_id]
    set_seed(args.seed)
    processor = WhisperProcessor.from_pretrained(model_id, revision=revision, language="arabic", task="transcribe", trust_remote_code=False)
    processor.tokenizer.set_prefix_tokens(language="arabic", task="transcribe")

    # FULL FINE-TUNING: Load entire model without PEFT/LoRA
    model = WhisperForConditionalGeneration.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    model.config.use_cache = False
    configure_generation(model, processor)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Training FULL MODEL: {trainable_params:,} learnable parameters; fixed sinusoidal positions remain fixed (no LoRA/adapters).")

    weights = [args.dokali_weight if row["reciter_key"] == "dokali" else 1.0 for row in splits["train"]]

    class WeightedTrainer(Seq2SeqTrainer):
        def _get_train_sampler(self, *sampler_args, **sampler_kwargs):
            return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    def generated_metrics(prediction):
        from src.training.asr_metrics import report_predictions
        ids = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
        hypotheses = processor.tokenizer.batch_decode(ids, skip_special_tokens=True)
        if len(hypotheses) != len(splits["validation"]):
            raise ValueError("Generated development predictions differ from frozen rows")
        details = [{"reciter": row["reciter_key"], "surah": row["surah"],
                    "reference": row["text_asr_normalized"], "prediction": normalize_quran_for_asr(text)}
                   for row, text in zip(splits["validation"], hypotheses)]
        scores = report_predictions(details)
        return {"wer": scores["overall"]["wer"], "cer": scores["overall"]["cer"],
                "macro_reciter_wer": scores["macro_reciter_wer"]}

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.10,
        weight_decay=0.01,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_reciter_wer",
        greater_is_better=False,
        remove_unused_columns=False,
        label_names=["labels"],
        predict_with_generate=True,
        generation_max_length=448,
        generation_num_beams=3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=25,
        max_grad_norm=1.0,
        dataloader_num_workers=0,
        report_to="none",
        seed=args.seed,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=AugmentedAyahDataset(splits["train"], processor, args.label_field, args.noise_prob, args.speed_prob,
                                          (args.tempo_min, args.tempo_max)),
        eval_dataset=AugmentedAyahDataset(splits["validation"], processor, args.label_field),
        data_collator=WhisperCollator(processor, model.config.decoder_start_token_id),
        compute_metrics=generated_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"init_model": model_id, "init_revision": revision or getattr(model.config, "_commit_hash", None),
        "fine_tuning": "full-all-learnable-layers", "seed": args.seed, "label_field": args.label_field,
        "normalizer_version": NORMALIZER_VERSION, "label_overlay_changes": label_changes,
        "selection": "generated-macro-reader-development-WER-beam3-not-cross-entropy",
        "fixed_parameters": [name for name, parameter in model.named_parameters() if not parameter.requires_grad],
        "generation_template": {"model": "openai/whisper-base", "revision": OPENAI_REVISION, "purpose": "language-task-token-metadata-only-not-weights"},
        "split_protocol": protocol,
        "max_steps": args.max_steps, "max_samples_per_split": args.max_samples_per_split,
        "is_smoke_run": args.max_steps > 0 or args.max_samples_per_split is not None,
        "tarteel_pretraining_data_overlap": "unknown-model-card-incomplete" if model_id.startswith(("tarteel-ai/", "deepdml/")) else "not-a-quran-specialist-checkpoint",
        "trabulsi_admitted": bool(args.trabulsi_reviewed_manifest),
        "splits": {name: [{**{k: row.get(k) for k in ("surah", "ayah", "reciter_key", "path", "source_sha256", "permission_receipt_sha256", "validation_report_sha256", "source_provider", "source_url", "text_asr_normalized")},
                          "audio_sha256": hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()} for row in rows] for name, rows in splits.items()},
        "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    (args.output_dir / "experiment_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    model.config.use_cache = True

    # Test evaluation
    from gpu_evaluation import evaluate_rows
    result = {"model": model_id, "best_validation_macro_reciter_wer": trainer.state.best_metric, "seed": args.seed,
               "init_revision": manifest["init_revision"], "is_smoke_run": manifest["is_smoke_run"],
               "pretraining_overlap": manifest["tarteel_pretraining_data_overlap"],
              "label_field": args.label_field, "split_counts": counts,
               "split_protocol": protocol,
              "validation_reciter": args.validation_reciter, "test_reciter": args.test_reciter,
              "split_reciters": {k: sorted({r["reciter_key"] for r in rows}) for k, rows in splits.items()},
              "augmentation": {"noise_prob": args.noise_prob, "speed_prob": args.speed_prob, "max_seconds": 30}}
    for split in ("validation", "test"):
        scores, details = evaluate_rows(model, processor, splits[split], args.eval_batch_size,
                                        decode_profile="beam3", normalizer_version=NORMALIZER_VERSION)
        result[split] = scores
        print(f"\n{split.upper()} RESULTS:", json.dumps(scores, ensure_ascii=False, indent=2))
        with (args.output_dir / f"{split}_predictions.jsonl").open("w", encoding="utf-8") as handle:
            for row in details:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    (args.output_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
