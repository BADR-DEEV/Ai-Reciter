# import json
# from pathlib import Path
# import pandas as pd
# import epitran
# from tqdm import tqdm
# import os

# # Load the manifest you just created

# os.environ["PYTHONIOENCODING"] = "utf-8"

# script_dir = Path(__file__).resolve().parent.parent.parent # ousdie to root

# MANIFEST_FILE = "train.csv"
# OUTPUT_CSV_PHONEMES = "train_with_phonemes.csv"


# INPUT_CSV = script_dir / "data" / MANIFEST_FILE
# OUTPUT_CSV_PHONEMES = script_dir / "data" / OUTPUT_CSV_PHONEMES


# MISTAKES_FILE = "mistakes.csv"
# MISTAKES_FILE_FINAL = script_dir / "data" / MISTAKES_FILE

# OUTPUT_MISTAKES_FILE = "mistakes_final.csv"
# OUTPUT_MISTAKE_PATH = script_dir / "data" / OUTPUT_MISTAKES_FILE



# OUTPUT_VOCAB_FILE = "vocab.json"
# OUTPUT_VOCAB_PATH = script_dir / "data" / OUTPUT_VOCAB_FILE

# # 1. Load Data
# df = pd.read_csv(OUTPUT_CSV_PHONEMES)

# # 2. Get all unique characters
# all_text = " ".join(df['phonemes'].tolist())
# vocab_set = list(set(all_text))
# vocab_set.sort()

# # 3. Create Dictionary
# vocab_dict = {v: k for k, v in enumerate(vocab_set)}

# # --- FIX 1: Handle Space safely ---
# # We will use "_" (Underscore) as the separator because "|" is already used for (آ)
# if " " in vocab_dict:
#     vocab_dict["_"] = vocab_dict.pop(" ") # Rename Space to Underscore

# # --- FIX 2: Handle Special Tokens without collision ---
# # We find the highest number in the dict to ensure we don't overwrite anything
# max_id = max(vocab_dict.values())

# vocab_dict["[UNK]"] = max_id + 1
# vocab_dict["[PAD]"] = max_id + 2

# # 4. Save
# with open(OUTPUT_VOCAB_PATH, "w", encoding="utf-8") as f:
#     json.dump(vocab_dict, f)

# print(f"Fixed! Vocab Size: {len(vocab_dict)}")
# print(f"Space is now '_': {vocab_dict.get('_')}")
# print(f"Shadda (~): {vocab_dict.get('~')}")
# print(f"[UNK]: {vocab_dict.get('[UNK]')}")