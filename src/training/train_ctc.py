from multiprocessing import freeze_support
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Union

import torchaudio
from datasets import Dataset
import evaluate

from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer
)
# --------------------------------------------------
# PATHS
# --------------------------------------------------
script_dir = Path(__file__).resolve().parent.parent.parent

INPUT_CSV = script_dir / "data" / "train_with_phonemes.csv"
VOCAB_FILE = script_dir / "data" / "vocab.json"
OUTPUT_DIR = script_dir / "data" / "quran_reciter_model_v1"

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"
BATCH_SIZE = 4
EPOCHS = 15
LEARNING_RATE = 1e-4
SAMPLING_RATE = 16000

# --------------------------------------------------
# DATA PREPARATION FUNCTION
# --------------------------------------------------
def prepare_dataset(batch):
    speech, sr = torchaudio.load(batch["path"])

    if sr != SAMPLING_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLING_RATE)
        speech = resampler(speech)

    batch["input_values"] = processor(
        speech.squeeze().numpy(),
        sampling_rate=SAMPLING_RATE
    ).input_values[0]

    with processor.as_target_processor():
        batch["labels"] = processor(batch["phonemes"]).input_ids

    return batch

# --------------------------------------------------
# DATA COLLATOR
# --------------------------------------------------
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt"
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        return batch

# --------------------------------------------------
# MAIN (WINDOWS SAFE)
# --------------------------------------------------
def main():
    print("1. Loading CSV...")
    df = pd.read_csv(INPUT_CSV)
    dataset = Dataset.from_pandas(df)

    print("2. Initializing Processor...")
    tokenizer = Wav2Vec2CTCTokenizer(
        VOCAB_FILE,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="_"
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=SAMPLING_RATE,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False
    )

    global processor
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )

    print("3. Preprocessing audio (this takes time)...")
    dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset.column_names,
        num_proc=3  # safe now
    )
    print("hello")
 
    print("4. Loading model...")
    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_NAME,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )

    model.freeze_feature_extractor()

    data_collator = DataCollatorCTCWithPadding(processor)

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = np.argmax(pred.predictions, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        group_by_length=True,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        evaluation_strategy="steps",
        save_steps=500,
        eval_steps=500,
        logging_steps=100,
        learning_rate=LEARNING_RATE,
        warmup_steps=1000,
        save_total_limit=2,
        fp16=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor
    )

    print("5. Starting Training... Bismillah 🤲")
    trainer.train()

    print("6. Saving model...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    print("✅ Training complete.")

# --------------------------------------------------
# ENTRY POINT (CRITICAL FOR WINDOWS)
# --------------------------------------------------
if __name__ == "__main__":
    freeze_support()
    main()
