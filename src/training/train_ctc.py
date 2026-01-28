from multiprocessing import freeze_support
from pathlib import Path
import pandas as pd
import torch
import numpy as np
import torchaudio
import soundfile as sf
from dataclasses import dataclass
from typing import Dict, List, Union
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

# --- CONFIG ---
script_dir = Path(__file__).resolve().parent.parent.parent
INPUT_CSV = script_dir / "data" / "train_with_phonemes.csv"
VOCAB_FILE = script_dir / "data" / "vocab.json"
OUTPUT_DIR = script_dir / "data" / "quran_reciter_model_v1"

# MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"
MODEL_NAME = "facebook/wav2vec2-base"
BATCH_SIZE = 1
EPOCHS = 15
LEARNING_RATE = 1e-4
SAMPLING_RATE = 16000

# --------------------------------------------------
# OPTIMIZED DATA COLLATOR (Loads audio on demand)
# --------------------------------------------------
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    sampling_rate: int = 16000

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # 1. Process Audio on the fly
        input_values = []
        for feature in features:
            # Load audio
            speech, sr = sf.read(feature["path"])
            if len(speech.shape) > 1:
                speech = np.mean(speech, axis=-1)
            
            # Resample if needed
            if sr != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
                speech = resampler(torch.from_numpy(speech).float()).numpy()
            
            # Extract features
            processed = self.processor(speech, sampling_rate=self.sampling_rate).input_values[0]
            input_values.append({"input_values": processed})

        # 2. Pad Audio
        batch = self.processor.feature_extractor.pad(
            input_values,
            padding=True,
            return_tensors="pt"
        )

        # 3. Pad Labels (Phonemes are already tokenized)
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=True,
            return_tensors="pt"
        )

        # Mask padding for loss calculation
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        return batch

# --------------------------------------------------
# PRE-PROCESS LABELS ONLY (Very fast & low RAM)
# --------------------------------------------------
def tokenize_labels(batch):
    batch["labels"] = processor(text=batch["phonemes"]).input_ids
    return batch

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
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    print("3. Tokenizing text labels (fast)...")
    # We DON'T process audio here, just the phoneme text
    dataset = dataset.map(
        tokenize_labels,
        remove_columns=["phonemes"], # Keep "path" so the collator can find the audio
        num_proc=None
    )

    print("4. Loading model...")
    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_NAME,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )
    model.freeze_feature_extractor()

    # Initialize the new On-The-Fly collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, sampling_rate=SAMPLING_RATE)

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = np.argmax(pred.predictions, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}

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
        fp16=True, # Set to False if you don't have an NVIDIA GPU
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

if __name__ == "__main__":
    freeze_support()
    main()