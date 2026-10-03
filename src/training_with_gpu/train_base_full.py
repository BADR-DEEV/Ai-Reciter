"""Full Fine-Tuning of Whisper-Base on 4 Qaloon Reciters with Speed Augmentation."""

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
    return p.parse_args()


class AugmentedAyahDataset:
    """In-memory cached dataset with dynamic speed and noise augmentation."""
    def __init__(self, rows, processor, label_field, noise_prob=0.0, speed_prob=0.0):
        self.processor = processor
        self.label_field = label_field
        self.noise_prob = noise_prob
        self.speed_prob = speed_prob
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

        # 2. Mild Acoustic Noise (25 to 35 dB SNR)
        if self.noise_prob and np.random.random() < self.noise_prob:
            snr = np.random.uniform(25, 35)
            amplitude = np.sqrt(np.mean(audio ** 2) + 1e-9) / (10 ** (snr / 20))
            audio = np.clip(audio + np.random.normal(0, amplitude, audio.shape), -1, 1).astype("float32")

        return {
            "input_features": self.processor.feature_extractor(audio, sampling_rate=sr).input_features[0],
            "labels": sample["labels"],
        }


def main():
    args = parse_args()
    splits, skipped = load_splits(args.data_root, None, None, args.seed)
    if bool(args.validation_reciter) != bool(args.test_reciter):
        raise ValueError("Pass both --validation-reciter and --test-reciter for speaker-disjoint training")
    if args.test_reciter:
        splits = speaker_disjoint_splits(splits, args.validation_reciter, args.test_reciter)
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

    model_id = "openai/whisper-base"
    processor = WhisperProcessor.from_pretrained(model_id, language="arabic", task="transcribe")
    processor.tokenizer.set_prefix_tokens(language="arabic", task="transcribe")

    # FULL FINE-TUNING: Load entire model without PEFT/LoRA
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.config.use_cache = False
    model.generation_config.language = "arabic"
    model.generation_config.task = "transcribe"
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔥 Training FULL MODEL: {trainable_params:,} parameters (100% of Whisper-Base)!")

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
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        label_names=["labels"],
        predict_with_generate=False,
        logging_steps=25,
        max_grad_norm=1.0,
        dataloader_num_workers=0,
        report_to="none",
        seed=args.seed,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=AugmentedAyahDataset(splits["train"], processor, args.label_field, args.noise_prob, args.speed_prob),
        eval_dataset=AugmentedAyahDataset(splits["validation"], processor, args.label_field),
        data_collator=WhisperCollator(processor, model.config.decoder_start_token_id),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    model.config.use_cache = True

    # Test evaluation
    from gpu_evaluation import evaluate_rows
    result = {"model": model_id, "best_eval_loss": trainer.state.best_metric, "seed": args.seed,
              "label_field": args.label_field, "split_counts": counts,
              "split_protocol": "speaker-disjoint" if args.test_reciter else "known-voice-unseen-ayah",
              "validation_reciter": args.validation_reciter, "test_reciter": args.test_reciter,
              "split_reciters": {k: sorted({r["reciter_key"] for r in rows}) for k, rows in splits.items()},
              "augmentation": {"noise_prob": args.noise_prob, "speed_prob": args.speed_prob, "max_seconds": 30}}
    for split in ("validation", "test"):
        scores, details = evaluate_rows(model, processor, splits[split], args.eval_batch_size, args.label_field)
        result[split] = scores
        print(f"\n{split.upper()} RESULTS:", json.dumps(scores, ensure_ascii=False, indent=2))
        with (args.output_dir / f"{split}_predictions.jsonl").open("w", encoding="utf-8") as handle:
            for row in details:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    (args.output_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
