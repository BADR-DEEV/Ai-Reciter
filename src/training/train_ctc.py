# from multiprocessing import freeze_support
# from pathlib import Path
# import pandas as pd
# import torch
# import numpy as np
# import torchaudio
# import soundfile as sf
# from dataclasses import dataclass
# from typing import Dict, List, Union
# from datasets import Dataset
# import evaluate
# from transformers import (
# Wav2Vec2CTCTokenizer,
# Wav2Vec2FeatureExtractor,
# Wav2Vec2Processor,
# Wav2Vec2ForCTC,
# TrainingArguments,
# Trainer,
# EarlyStoppingCallback
# )


# script_dir = Path(file).resolve().parent.parent.parent
# INPUT_CSV = script_dir / "data" / "train_with_phonemes.csv"
# VOCAB_FILE = script_dir / "data" / "vocab.json"
# OUTPUT_DIR = script_dir / "data" / "quran_reciter_model_v1"

# REPORT = script_dir / "data" / "report_model"

# MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"

# MODEL_NAME = "facebook/wav2vec2-base"
# BATCH_SIZE = 1
# EPOCHS = 15
# LEARNING_RATE = 1e-4
# SAMPLING_RATE = 16000

# @dataclass
# class DataCollatorCTCWithPadding:
#     processor: Wav2Vec2Processor
#     sampling_rate: int = 16000

#     def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
#         # 1. Process Audio on the fly
#         input_values = []
#         for feature in features:
#             # Load audio
#             speech, sr = sf.read(feature["path"])
#             if len(speech.shape) > 1:
#                 speech = np.mean(speech, axis=-1)
            
#             # Resample if needed
#             if sr != self.sampling_rate:
#                 resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
#                 speech = resampler(torch.from_numpy(speech).float()).numpy()
            
#             # Extract features
#             processed = self.processor(speech, sampling_rate=self.sampling_rate).input_values[0]
#             input_values.append({"input_values": processed})

#         # 2. Pad Audio
#         batch = self.processor.feature_extractor.pad(
#             input_values,
#             padding=True,
#             return_tensors="pt"
#         )
#         # --- ADD THIS LINE ---
#         batch["input_values"].requires_grad = True

#         # 3. Pad Labels (Phonemes are already tokenized)
#         label_features = [{"input_ids": f["labels"]} for f in features]
#         labels_batch = self.processor.tokenizer.pad(
#             label_features,
#             padding=False,
#             return_tensors="pt"
#         )

#         # Mask padding for loss calculation
#         labels = labels_batch["input_ids"].masked_fill(
#             labels_batch.attention_mask.ne(1), -100
#         )

#         batch["labels"] = labels
#         return batch

# def tokenize_labels(batch):
#     batch["labels"] = processor(text=batch["phonemes"]).input_ids
#     batch["input_length"] = int(batch["duration"] * SAMPLING_RATE)

#     return batch

# def main():
#     print("1. Loading CSV...")
#     df = pd.read_csv(INPUT_CSV)
#     dataset = Dataset.from_pandas(df)


# print("2. Initializing Processor...")
# tokenizer = Wav2Vec2CTCTokenizer(
#     VOCAB_FILE,
#     unk_token="[UNK]",
#     pad_token="[PAD]",
#     word_delimiter_token="_"
# )

# feature_extractor = Wav2Vec2FeatureExtractor(
#     feature_size=1,
#     sampling_rate=SAMPLING_RATE,
#     padding_value=0.0,
#     do_normalize=True,
#     return_attention_mask=False
# )

# global processor
# processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# print("3. Tokenizing text labels (fast)...")
# # We DON'T process audio here, just the phoneme text
# dataset = dataset.map(
#     tokenize_labels,
#     num_proc=None,
#     remove_columns=[col for col in dataset.column_names if col not in ["path", "labels", "input_length"]],

# )

# print("4. Loading model...")
# model = Wav2Vec2ForCTC.from_pretrained(
#     MODEL_NAME,
#     ctc_loss_reduction="mean",
#     pad_token_id=processor.tokenizer.pad_token_id,
#     vocab_size=len(processor.tokenizer)
# )
# model.config.use_cache = False 
# model.gradient_checkpointing_enable()
# model.freeze_feature_encoder()

# # Initialize the new On-The-Fly collator
# data_collator = DataCollatorCTCWithPadding(processor=processor, sampling_rate=SAMPLING_RATE)

# wer_metric = evaluate.load("wer")

# def compute_metrics(pred):
#     pred_ids = np.argmax(pred.predictions, axis=-1)
#     pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
#     pred_str = processor.batch_decode(pred_ids)
#     label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
#     return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}

# training_args = TrainingArguments(
#     output_dir=OUTPUT_DIR,
#     group_by_length=True,           # Optimizes padding to save memory
#     per_device_train_batch_size=BATCH_SIZE,
#     gradient_accumulation_steps=4,  # Simulates batch size 4 for better learning
#     num_train_epochs=EPOCHS,
#     gradient_checkpointing=True,    # ESSENTIAL for 4GB VRAM
#     fp16=True, 
#     eval_strategy="steps",
#     save_steps=500,
#     eval_steps=500,
#     logging_steps=100,
#     learning_rate=LEARNING_RATE,
#     warmup_steps=1000,
#     save_total_limit=2,
#     report_to=[],
#     remove_unused_columns=False,
#     length_column_name="input_length",
#         # --- ADD THESE THREE LINES ---
#     load_best_model_at_end=True,   # Required for EarlyStopping
#     metric_for_best_model="wer",   # or "loss"
#     greater_is_better=False,  
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset,
#     eval_dataset=dataset,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
#     processing_class=processor,
#     callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]

#       # Pass the full processor, not just the extractor

# )

# print("5. Starting Training... Bismillah 🤲")
# trainer.train()

# print("6. Saving model...")
# trainer.save_model(OUTPUT_DIR)
# processor.save_pretrained(OUTPUT_DIR)
# print("✅ Training complete.")

# if __name__ == "__main__":
#     freeze_support()
#     main()



# # from multiprocessing import freeze_support
# # from pathlib import Path
# # import pandas as pd
# # import torch
# # import numpy as np
# # import torchaudio
# # import soundfile as sf
# # from dataclasses import dataclass
# # from typing import Dict, List, Union
# # from datasets import Dataset
# # import evaluate
# # from transformers import (
# #     WhisperProcessor,
# #     WhisperForConditionalGeneration,
# #     WhisperTokenizer,
# #     WhisperFeatureExtractor,
# #     TrainingArguments,
# #     Trainer,
# #     EarlyStoppingCallback,
# #     Seq2SeqTrainingArguments, 
# #     Seq2SeqTrainer,
    
# # )

# # # --- CONFIG ---
# # script_dir = Path(__file__).resolve().parent.parent.parent
# # INPUT_CSV = script_dir / "data" / "train_with_phonemes.csv"
# # OUTPUT_DIR = script_dir / "data" / "quran_whisper_tiny_v1"

# # MODEL_NAME = "openai/whisper-tiny" # Whisper Tiny is excellent for 4GB VRAM
# # BATCH_SIZE = 1 
# # EPOCHS = 15
# # LEARNING_RATE = 1e-4
# # SAMPLING_RATE = 16000 # Whisper expects 16kHz

# # # --------------------------------------------------
# # # WHISPER DATA COLLATOR
# # # --------------------------------------------------
# # @dataclass
# # class DataCollatorWhisperWithPadding:
# #     processor: WhisperProcessor

# #     def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
# #         # 1. Whisper expects Log-Mel Spectrograms
# #         input_features = []
# #         for feature in features:
# #             speech, sr = sf.read(feature["path"])
# #             if len(speech.shape) > 1:
# #                 speech = np.mean(speech, axis=-1)
            
# #             # Resample if needed
# #             if sr != 16000:
# #                 resampler = torchaudio.transforms.Resample(sr, 16000)
# #                 speech = resampler(torch.from_numpy(speech).float()).numpy()

# #             # Feature extractor converts raw audio to Log-Mel Spectrogram
# #             # Whisper always pads/truncates to 30 seconds internally
# #             processed = self.processor.feature_extractor(speech, sampling_rate=16000).input_features[0]
# #             input_features.append({"input_features": processed})

# #         batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

# #         # 2. Tokenize labels
# #         label_features = [{"input_ids": feature["labels"]} for feature in features]
# #         labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

# #         # replace padding with -100 to ignore loss correctly
# #         labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

# #         batch["labels"] = labels
# #         return batch

# # # --------------------------------------------------
# # # PRE-PROCESS LABELS
# # # --------------------------------------------------
# # def tokenize_labels(batch):
# #     # Whisper tokenizer handles text/phonemes differently than CTC
# #     batch["labels"] = processor.tokenizer(batch["phonemes"]).input_ids
# #     # Whisper needs to know length for group_by_length
# #     batch["input_length"] = int(batch["duration"] * SAMPLING_RATE)
# #     return batch

# # def main():
# #     print("1. Loading CSV...")
# #     df = pd.read_csv(INPUT_CSV)
# #     dataset = Dataset.from_pandas(df)

# #     print("2. Initializing Whisper Processor...")
# #     # Whisper uses a specific Arabic setup
# #     processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="arabic", task="transcribe")
    
# #     # Global for the map function
# #     global global_processor
# #     global_processor = processor

# #     print("3. Tokenizing text labels...")
# #     dataset = dataset.map(
# #         tokenize_labels,
# #         num_proc=None,
# #         remove_columns=[col for col in dataset.column_names if col not in ["path", "labels", "input_length"]],
# #     )

# #     print("4. Loading Whisper Model...")
# #     model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    
# #     # Optimizationf for 4GB VRAM
# #     model.config.forced_decoder_ids = None
# #     model.config.suppress_tokens = []
# #     model.config.use_cache = False
# #     model.gradient_checkpointing_enable()

# #     data_collator = DataCollatorWhisperWithPadding(processor=processor)

# #     wer_metric = evaluate.load("wer")

# #     def compute_metrics(pred):
# #         pred_ids = pred.predictions
# #         label_ids = pred.label_ids

# #         # replace -100 with the pad_token_id
# #         label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

# #         # decode predictions and labels
# #         pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
# #         label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

# #         wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
# #         return {"wer": wer}
# #     has_gpu = torch.cuda.is_available()
# # training_args = Seq2SeqTrainingArguments(
# #         output_dir=OUTPUT_DIR,
# #         per_device_train_batch_size=BATCH_SIZE,
# #         gradient_accumulation_steps=4,
# #         learning_rate=LEARNING_RATE,
# #         num_train_epochs=EPOCHS,
        
# #         # --- FIX FOR THE STRATEGY ERROR ---
# #         eval_strategy="steps",       # Explicitly set to steps
# #         save_strategy="steps",       # Must match eval_strategy
# #         eval_steps=500,              # How often to run evaluation
# #         save_steps=500,              # How often to save a checkpoint
# #         # ----------------------------------

# #         fp16=has_gpu,                
# #         gradient_checkpointing=has_gpu, 
        
# #         predict_with_generate=True,
# #         generation_max_length=225,
# #         logging_steps=100,
# #         report_to=[],
# #         load_best_model_at_end=True,   # Now it knows to look at the 'steps'
# #         metric_for_best_model="wer",
# #         greater_is_better=False,
# #         remove_unused_columns=False
# #     )
    
# #     trainer = Seq2SeqTrainer(
# #         model=model,
# #         args=training_args,
# #         train_dataset=dataset,
# #         eval_dataset=dataset,
# #         data_collator=data_collator,
# #         compute_metrics=compute_metrics,
# #         tokenizer=processor.feature_extractor,
# #         callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
# #     )
# #     print("5. Starting Whisper Training... Bismillah 🤲")
# #     trainer.train()

# #     print("6. Saving Whisper model...")
# #     trainer.save_model(OUTPUT_DIR)
# #     processor.save_pretrained(OUTPUT_DIR)
# #     print("✅ Training complete.")

# # # Global processor helper for map
# # global_processor = None
# # def tokenize_labels(batch):
# #     batch["labels"] = global_processor.tokenizer(batch["phonemes"]).input_ids
# #     batch["input_length"] = int(batch["duration"] * SAMPLING_RATE)
# #     return batch

# # if __name__ == "__main__":
# #     freeze_support()
# #     main()