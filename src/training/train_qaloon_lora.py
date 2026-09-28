"""Train Whisper tiny/base LoRA adapters on the Qaloon ayah WAV datasets."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from qaloon_data import DATA_ROOT, RECITER_DIRS, audio_features, evaluate_model, load_splits


class AyahDataset:
    def __init__(self, rows, processor):
        self.rows = rows
        self.processor = processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return {
            "input_features": audio_features(self.processor, row["path"]),
            "labels": self.processor.tokenizer(row["text_asr_normalized"]).input_ids,
        }


@dataclass
class WhisperCollator:
    processor: object
    decoder_start_token_id: int

    def __call__(self, examples):
        features = self.processor.feature_extractor.pad(
            [{"input_features": item["input_features"]} for item in examples],
            return_tensors="pt",
        )
        targets = self.processor.tokenizer.pad(
            [{"input_ids": item["labels"]} for item in examples], return_tensors="pt"
        )
        labels = targets.input_ids.masked_fill(targets.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]
        features["labels"] = labels
        return features


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["tiny", "base"], required=True)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-reciter", nargs="+", choices=RECITER_DIRS)
    parser.add_argument("--exclude-reciter", nargs="+", choices=RECITER_DIRS)
    parser.add_argument("--include-bismillah", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                              WhisperForConditionalGeneration, WhisperProcessor)

    splits, skipped = load_splits(
        args.data_root, args.include_reciter, args.exclude_reciter,
        args.seed, include_bismillah=args.include_bismillah,
    )
    print("Split sizes:", {key: len(rows) for key, rows in splits.items()})
    print("Excluded clips longer than 30 seconds:", skipped)
    model_id = f"openai/whisper-{args.model}"
    processor = WhisperProcessor.from_pretrained(model_id, language="arabic", task="transcribe")
    processor.tokenizer.set_prefix_tokens(language="arabic", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.config.use_cache = False
    model.generation_config.language = "arabic"
    model.generation_config.task = "transcribe"
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, 
        lora_alpha=args.lora_alpha, 
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], 
        bias="none",
    ))
    model.print_trainable_parameters()

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate, warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        predict_with_generate=False, logging_steps=25,
        remove_unused_columns=False, label_names=["labels"],
        report_to="none", seed=args.seed, data_seed=args.seed,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=training_args,
        train_dataset=AyahDataset(splits["train"], processor),
        eval_dataset=AyahDataset(splits["validation"], processor),
        data_collator=WhisperCollator(processor, model.config.decoder_start_token_id),
    )
    trainer.train()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    model.config.use_cache = True
    results = {"model": model_id, "seed": args.seed,
               "reciters": sorted({r["reciter_key"] for r in splits["train"]}),
               "counts": {key: len(rows) for key, rows in splits.items()},
               "skipped_over_30s": skipped}
    for name in ("validation", "test"):
        scores, details = evaluate_model(model, processor, splits[name], args.eval_batch_size)
        results[name] = scores
        with (args.output_dir / f"{name}_predictions.jsonl").open("w", encoding="utf-8") as handle:
            for detail in details:
                handle.write(json.dumps(detail, ensure_ascii=False) + "\n")
        print(name, scores)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
