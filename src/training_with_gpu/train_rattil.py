"""Rattil Qaloon training (v1-v4): full fine-tune from a Quran Whisper checkpoint with tempo augmentation."""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
sys.path.insert(0, str(TRAINING_DIR))
from qaloon_data import DATA_ROOT, RECITER_DIRS, load_splits, speaker_disjoint_splits


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--output-dir", type=Path, default=Path("runs/gpu_base_full"))
    p.add_argument("--dokali-weight", type=float, default=0.35)
    p.add_argument("--noise-prob", type=float, default=0.15)
    p.add_argument("--speed-prob", type=float, default=0.40, help="Probability of speed/tempo perturbation")
    p.add_argument("--tempo-prob", type=float, default=0.0,
                   help="Probability of a pitch-preserving tempo change (phase vocoder), e.g. for fast readers")
    p.add_argument("--tempo-min", type=float, default=0.85, help="Slowest tempo factor")
    p.add_argument("--tempo-max", type=float, default=1.45, help="Fastest tempo factor")
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
    p.add_argument("--base-model", default="openai/whisper-base",
                   help="Checkpoint to start from, e.g. a Quran-pretrained Whisper of any size")
    p.add_argument("--include-reciter", nargs="+", choices=RECITER_DIRS,
                   help="Only these reciters (default: all reciter folders)")
    p.add_argument("--exclude-clips", type=Path,
                   help="audit_labels.py output; its flagged clips are left out of training only")
    p.add_argument("--freeze-encoder", action="store_true", help="Train the decoder only")
    p.add_argument("--mask-time-prob", type=float, default=0.0, help="SpecAugment time masking (0 disables)")
    p.add_argument("--select-by", choices=("loss", "wer"), default="loss",
                   help="Checkpoint selection: validation loss, or generated validation WER")
    p.add_argument("--train-on-all", action="store_true",
                   help="Release model: train on every split for a fixed --epochs, with no selection or "
                        "self-evaluation; evaluate it afterwards on reciters left out with --include-reciter")
    p.add_argument("--dry-run", action="store_true", help="Validate splits without loading a model or training")
    return p.parse_args()


def stretch_tempo(audio, rate, n_fft=512, hop=128):
    """Phase-vocoder time stretch: rate > 1 is faster speech at the same pitch."""
    window = torch.hann_window(n_fft)
    spec = torch.stft(torch.from_numpy(audio), n_fft, hop, window=window, return_complex=True)
    advance = torch.linspace(0, np.pi * hop, spec.shape[-2])[..., None]
    stretched = F.phase_vocoder(spec, rate, advance)
    return torch.istft(stretched, n_fft, hop, window=window).numpy().astype("float32")


class AugmentedAyahDataset:
    """In-memory cached dataset with dynamic speed and noise augmentation."""
    def __init__(self, rows, processor, label_field, noise_prob=0.0, speed_prob=0.0, tempo=(0.0, 1.0, 1.0)):
        self.processor = processor
        self.label_field = label_field
        self.noise_prob = noise_prob
        self.speed_prob = speed_prob
        self.tempo_prob, self.tempo_min, self.tempo_max = tempo
        self.cached_samples = []

        print(f"Pre-caching {len(rows)} audio files into RAM...")
        for row in rows:
            audio, sr = sf.read(row["path"], dtype="float32")
            if sr != 16000 or audio.ndim != 1:
                raise ValueError(f"Expected 16 kHz mono WAV: {row['path']}")
            
            labels = self.processor.tokenizer(row[self.label_field]).input_ids
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
            speed_factor = np.random.uniform(0.92, 1.08)
            audio_tensor = torch.from_numpy(audio).float()
            audio_tensor = F.resample(audio_tensor, int(sr * speed_factor), sr)
            candidate = audio_tensor.numpy()
            # Slowing a 29s clip can make it >30s. Do not silently truncate its
            # end while keeping the full transcript: reject that augmentation.
            if len(candidate) <= sr * 30:
                audio = candidate

        # 2. Pitch-preserving tempo change: the same voice reading faster or slower.
        if self.tempo_prob and np.random.random() < self.tempo_prob:
            candidate = stretch_tempo(audio, np.random.uniform(self.tempo_min, self.tempo_max))
            if len(candidate) <= sr * 30:
                audio = candidate

        # 3. Mild Acoustic Noise (25 to 35 dB SNR)
        if self.noise_prob and np.random.random() < self.noise_prob:
            snr = np.random.uniform(25, 35)
            amplitude = np.sqrt(np.mean(audio ** 2) + 1e-9) / (10 ** (snr / 20))
            audio = np.clip(audio + np.random.normal(0, amplitude, audio.shape), -1, 1).astype("float32")

        return {
            "input_features": self.processor.feature_extractor(audio, sampling_rate=sr).input_features[0],
            "labels": sample["labels"],
        }


def validation_wer(processor):
    """Generated-text WER with the project's normalizer, for checkpoint selection."""
    import jiwer
    sys.path.insert(0, str(TRAINING_DIR.parent / "dataset_collection"))
    from qaloon_audio2text import normalize_quran_for_asr

    def decode(ids):
        # The trainer pads predictions and labels with -100 across batches.
        return processor.batch_decode(np.where(ids != -100, ids, processor.tokenizer.pad_token_id),
                                      skip_special_tokens=True)

    def compute(prediction):
        references = [normalize_quran_for_asr(text) for text in decode(prediction.label_ids)]
        hypotheses = [normalize_quran_for_asr(text) for text in decode(prediction.predictions)]
        return {"wer": jiwer.wer(references, hypotheses)}
    return compute


def main():
    args = parse_args()
    splits, skipped = load_splits(args.data_root, args.include_reciter, None, args.seed)
    if bool(args.validation_reciter) != bool(args.test_reciter):
        raise ValueError("Pass both --validation-reciter and --test-reciter for speaker-disjoint training")
    if args.test_reciter:
        splits = speaker_disjoint_splits(splits, args.validation_reciter, args.test_reciter)
    if args.train_on_all:
        splits = {"train": [row for part in splits.values() for row in part], "validation": [], "test": []}
    excluded = []
    if args.exclude_clips:
        flagged = {item["key"] for item in json.loads(args.exclude_clips.read_text(encoding="utf-8"))["flagged"]}
        key = lambda row: f"{row['reciter_key']}:{row['surah']}:{row['ayah']}"
        excluded = sorted(key(row) for row in splits["train"] if key(row) in flagged)
        splits["train"] = [row for row in splits["train"] if key(row) not in flagged]
        print(f"Excluded {len(excluded)} flagged clips from training")
    evaluate = not args.train_on_all
    counts = {k: len(v) for k, v in splits.items()}
    print("Split counts:", counts, "Excluded >30s:", skipped)
    print("Reciters:", {k: sorted({row["reciter_key"] for row in rows}) for k, rows in splits.items()})
    if args.dry_run:
        return
    if (args.output_dir / "model.safetensors").exists():
        raise ValueError("Choose a new output directory; refusing to overwrite an existing full model")

    from torch.utils.data import WeightedRandomSampler
    from train_qaloon_lora import WhisperCollator
    from transformers import (EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments,
                              WhisperForConditionalGeneration, WhisperProcessor)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")
    print("GPU:", torch.cuda.get_device_name(0), "VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), "GB")

    model_id = args.base_model
    processor = WhisperProcessor.from_pretrained(model_id, language="arabic", task="transcribe")
    processor.tokenizer.set_prefix_tokens(language="arabic", task="transcribe")

    # FULL FINE-TUNING: Load entire model without PEFT/LoRA
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    if not hasattr(model.generation_config, "lang_to_id"):
        # Some community checkpoints were saved without the multilingual generation config.
        from transformers import GenerationConfig
        from benchmark_models import OPENAI_BASES
        c = model.config
        model.generation_config = GenerationConfig.from_pretrained(OPENAI_BASES[(c.d_model, c.encoder_layers, c.decoder_layers)])
    model.config.use_cache = False
    model.generation_config.language = "arabic"
    model.generation_config.task = "transcribe"
    if args.mask_time_prob:
        model.config.apply_spec_augment = True
        model.config.mask_time_prob = args.mask_time_prob
    if args.freeze_encoder:
        model.model.encoder.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Training {trainable_params:,} of {total_params:,} parameters from {model_id}")

    weights = [args.dokali_weight if row["reciter_key"] == "dokali" else 1.0 for row in splits["train"]]

    class WeightedTrainer(Seq2SeqTrainer):
        def _get_train_sampler(self, *sampler_args, **sampler_kwargs):
            return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.10,
        weight_decay=0.01,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        eval_strategy="epoch" if evaluate else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=evaluate,
        metric_for_best_model="eval_wer" if args.select_by == "wer" else "eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        label_names=["labels"],
        predict_with_generate=args.select_by == "wer",
        generation_max_length=225,
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
                                           (args.tempo_prob, args.tempo_min, args.tempo_max)),
        eval_dataset=AugmentedAyahDataset(splits["validation"], processor, args.label_field) if evaluate else None,
        data_collator=WhisperCollator(processor, model.config.decoder_start_token_id),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)] if evaluate else [],
        compute_metrics=validation_wer(processor) if evaluate and args.select_by == "wer" else None,
    )

    if evaluate and args.select_by == "wer":
        print("Validation before training:", trainer.evaluate())
    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    model.config.use_cache = True

    # Test evaluation
    from gpu_evaluation import evaluate_rows
    result = {"model": model_id, "selected_by": args.select_by, "best_eval_metric": trainer.state.best_metric,
              "seed": args.seed, "freeze_encoder": args.freeze_encoder, "mask_time_prob": args.mask_time_prob,
              "learning_rate": args.learning_rate, "excluded_training_clips": excluded,
              "label_field": args.label_field, "split_counts": counts,
              "split_protocol": "speaker-disjoint" if args.test_reciter else "known-voice-unseen-ayah",
              "validation_reciter": args.validation_reciter, "test_reciter": args.test_reciter,
              "split_reciters": {k: sorted({r["reciter_key"] for r in rows}) for k, rows in splits.items()},
              "augmentation": {"noise_prob": args.noise_prob, "speed_prob": args.speed_prob, "max_seconds": 30,
                               "tempo_prob": args.tempo_prob, "tempo_range": [args.tempo_min, args.tempo_max]},
              "trained_on_all_splits": args.train_on_all, "epochs": args.epochs}
    for split in ("validation", "test") if evaluate else ():
        scores, details = evaluate_rows(model, processor, splits[split], args.eval_batch_size, args.label_field)
        result[split] = scores
        print(f"\n{split.upper()} RESULTS:", json.dumps(scores, ensure_ascii=False, indent=2))
        with (args.output_dir / f"{split}_predictions.jsonl").open("w", encoding="utf-8") as handle:
            for row in details:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    (args.output_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
