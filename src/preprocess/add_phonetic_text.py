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


# # --- 1. DEFINE THE MAPPING FUNCTION ---
# def arabic_to_buckwalter(text):
#     if not isinstance(text, str): return ""
    
#     mapping = {
#         'ا': 'A', 'ب': 'b', 'ت': 't', 'ث': 'v', 'ج': 'j', 'ح': 'H', 'خ': 'x', 
#         'د': 'd', 'ذ': '*', 'ر': 'r', 'ز': 'z', 'س': 's', 'ش': '$', 'ص': 'S', 
#         'ض': 'D', 'ط': 'T', 'ظ': 'Z', 'ع': 'E', 'غ': 'g', 'ف': 'f', 'ق': 'q', 
#         'ك': 'k', 'ل': 'l', 'م': 'm', 'ن': 'n', 'ه': 'h', 'و': 'w', 'ي': 'y',
#         'ة': 'p', 'ى': 'Y', 'ئ': '}', 'ؤ': '&', 'أ': '>', 'إ': '<', 'آ': '|', 'ء': '\'',
#         # HARAKAT (Crucial for your grading)
#         '\u064e': 'a', # Fatha
#         '\u0650': 'i', # Kasra
#         '\u064f': 'u', # Damma
#         '\u064b': 'F', # Fathatan
#         '\u064d': 'K', # Kasratan
#         '\u064c': 'N', # Dammatan
#         '\u0651': '~', # Shadda
#         '\u0652': 'o', # Sukun
#         '\u0670': '`', # Dagger Alif
#         ' ': ' '
#     }
    
#     return "".join([mapping.get(char, char) for char in text])

# # --- 2. FIX THE HEALTHY DATASET ---
# print("Fixing Healthy Dataset...")
# df_train = pd.read_csv(INPUT_CSV) # Or whatever your file is named

# # Apply conversion
# # "phonemes" column will now be strictly ASCII characters
# df_train['phonemes'] = df_train['text'].apply(arabic_to_buckwalter)

# # Save it back
# df_train.to_csv(OUTPUT_CSV_PHONEMES, index=False)
# print("Example Healthy:", df_train.iloc[0]['phonemes'])
# # EXPECTED OUTPUT: bisomi {ll~ahi {lr~aHoma`ni {lr~aHiymi
# # (Notice: No Arabic characters left. Only English letters and symbols representing sounds)


# # --- 3. GENERATE & FIX MISTAKES DATASET ---
# # You asked: "Do I also update the generated mistakes?"
# # YES. They must use the exact same format.

# print("Fixing/Generating Mistakes Dataset...")
# # If you already have mistakes.csv, load it. If not, generate it from df_train like before.
# if os.path.exists(MISTAKES_FILE_FINAL):
#     df_mistakes = pd.read_csv(MISTAKES_FILE_FINAL)
    
#     # IMPORTANT: The 'text' in mistakes.csv is ALREADY wrong (has missing words),
#     # so we just convert that wrong text to Buckwalter.
#     df_mistakes['phonemes'] = df_mistakes['text'].apply(arabic_to_buckwalter)
    
#     df_mistakes.to_csv(OUTPUT_MISTAKE_PATH, index=False)
#     print("Example Mistake:", df_mistakes.iloc[0]['phonemes'])

# print("DONE. Use 'train_final.csv' and 'mistakes_final.csv' for AI Training.")