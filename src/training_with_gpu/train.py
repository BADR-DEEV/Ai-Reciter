"""GPU Whisper LoRA: four Qaloon reciters, cautious audio augmentation, best-checkpoint selection."""

import argparse
import json
import sys
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
sys.path.insert(0, str(TRAINING_DIR))
from qaloon_data import DATA_ROOT, RECITER_DIRS, evaluate_model, load_splits
from train_qaloon_lora import WhisperCollator


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["tiny", "base"], default="base")
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--include-reciter", nargs="+", choices=RECITER_DIRS)
    p.add_argument("--exclude-reciter", nargs="+", choices=RECITER_DIRS)
    p.add_argument("--dokali-weight", type=float, default=0.35,
                   help="Relative sampling weight during TRAINING only; validation/test remain unweighted")
    p.add_argument("--noise-prob", type=float, default=0.0,
                   help="Optional training-only mild white-noise probability (0-1)")
    p.add_argument("--label-field", choices=["text_asr_normalized", "normalized_with_harakat"],
                   default="text_asr_normalized")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="Validate paths, splits and duration without GPU")
    args = p.parse_args()
    if not 0 < args.dokali_weight <= 1 or not 0 <= args.noise_prob <= 1:
        p.error("--dokali-weight must be (0,1] and --noise-prob must be [0,1]")
    return args


class AyahDataset:
    def __init__(self, rows, processor, label_field, noise_prob=0):
        self.rows, self.processor = rows, processor
        self.label_field, self.noise_prob = label_field, noise_prob

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import numpy as np
        import soundfile as sf

        row = self.rows[index]
        audio, sr = sf.read(row["path"], dtype="float32")
        if sr != 16000 or audio.ndim != 1:
            raise ValueError(f"Expected 16 kHz mono WAV: {row['path']}")
        if self.noise_prob and np.random.random() < self.noise_prob:
            # Mild recording noise only: never change the length or word content.
            snr = np.random.uniform(25, 35)
            amplitude = np.sqrt(np.mean(audio ** 2) + 1e-9) / (10 ** (snr / 20))
            audio = np.clip(audio + np.random.normal(0, amplitude, audio.shape), -1, 1).astype("float32")
        return {
            "input_features": self.processor.feature_extractor(audio, sampling_rate=sr).input_features[0],
            "labels": self.processor.tokenizer(row[self.label_field]).input_ids,
        }


def main():
    args = parse_args()
    splits, skipped = load_splits(args.data_root, args.include_reciter, args.exclude_reciter, args.seed)
    if args.label_field == "normalized_with_harakat":
        for rows in splits.values():
            for row in rows:
                if not row.get(args.label_field):
                    raise ValueError(f"Missing {args.label_field}: {row['path']}; run add_harakat_metadata.py")
    counts = {k: len(v) for k, v in splits.items()}
    print("Split counts:", counts, "Excluded >30s / invalid:", skipped)
    print("Training reciters:", {k: sum(r["reciter_key"] == k for r in splits["train"])
                                 for k in RECITER_DIRS})
    if args.dry_run:
        return

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.utils.data import WeightedRandomSampler
    from transformers import (EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments,
                              WhisperForConditionalGeneration, WhisperProcessor)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required. For RTX 5070, install a PyTorch CUDA build supporting sm_120.")
    print("GPU:", torch.cuda.get_device_name(0), "capability:", torch.cuda.get_device_capability(0))
    model_id = f"openai/whisper-{args.model}"
    processor = WhisperProcessor.from_pretrained(model_id, language="arabic", task="transcribe")
    processor.tokenizer.set_prefix_tokens(language="arabic", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.config.use_cache = False
    model.generation_config.language, model.generation_config.task = "arabic", "transcribe"
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        bias="none", task_type=TaskType.SEQ_2_SEQ_LM,
    ))
    model.print_trainable_parameters()
    model.enable_input_require_grads()

    weights = [args.dokali_weight if row["reciter_key"] == "dokali" else 1.0
               for row in splits["train"]]

    class WeightedTrainer(Seq2SeqTrainer):
        def _get_train_sampler(self, *sampler_args, **sampler_kwargs):
            return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate, warmup_ratio=0.08, weight_decay=0.01,
        bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        remove_unused_columns=False, label_names=["labels"],
        predict_with_generate=False, logging_steps=25, max_grad_norm=1.0,
        dataloader_num_workers=0, report_to="none", seed=args.seed, data_seed=args.seed,
    )
    trainer = WeightedTrainer(
        model=model, args=training_args,
        train_dataset=AyahDataset(splits["train"], processor, args.label_field, args.noise_prob),
        eval_dataset=AyahDataset(splits["validation"], processor, args.label_field),
        data_collator=WhisperCollator(processor, model.config.decoder_start_token_id),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    trainer.train()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    model.config.use_cache = True
    result = {"model": model_id, "seed": args.seed, "label_field": args.label_field,
              "split_counts": counts, "excluded": skipped,
              "best_checkpoint": trainer.state.best_model_checkpoint,
              "best_eval_loss": trainer.state.best_metric}
    from gpu_evaluation import evaluate_rows
    for split in ("validation", "test"):
        scores, details = evaluate_rows(model, processor, splits[split], args.eval_batch_size,
                                        args.label_field)
        result[split] = scores
        print(split, json.dumps(scores, ensure_ascii=False))
        with (args.output_dir / f"{split}_predictions.jsonl").open("w", encoding="utf-8") as handle:
            for row in details:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
