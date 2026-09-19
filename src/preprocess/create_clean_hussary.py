# import os
# import pandas as pd
# import librosa
# import soundfile as sf
# from tqdm import tqdm
# from pathlib import Path

# # --- CONFIGURATION ---
# HUSSARY_FOLDER_NAME = "Husary_Muslim_128kbps"
# OUTPUT_HUSSARY_FOLDER = "processed_audio_16k" 
# TEXT_FILE = "quran-uthmani.txt"
# MANIFEST_FILE = "train.csv"


# script_dir = Path(__file__).resolve().parent.parent.parent # ousdie to root
# hussary_directory = script_dir / "data" / HUSSARY_FOLDER_NAME 
# hussary_16k_directory = script_dir / "data" / OUTPUT_HUSSARY_FOLDER
# quran_text = script_dir / "data" / TEXT_FILE
# csv_manifest = script_dir / "data" / MANIFEST_FILE

# os.makedirs(hussary_16k_directory, exist_ok=True)



# print(f"Hussary Directory: {hussary_directory}")
# print(f"Hussary 16k Directory: {hussary_16k_directory}")

# # 1. Load Quran Text
# # Assumption: Text file has exactly 6236 lines, ordered from 1:1 to 114:6
# with open(quran_text, 'r', encoding='utf-8') as f:
#     lines = [line.strip() for line in f.readlines()]

# if len(lines) != 6236:
#     print(f"WARNING: Text file has {len(lines)} lines. Expected 6236.")

# # 2. Map Surah/Ayah counts (Standard Hafs counts)
# # We need this to know which file corresponds to which line in the text
# # Simple helper: You need a list of how many ayahs are in each surah
# surah_ayah_counts = [7, 286, 200, 176, 120, 165, 206, 75, 129, 109, 123, 111, 43, 52, 99, 128, 111, 110, 98, 135, 112, 78, 118, 64, 77, 227, 93, 88, 69, 60, 34, 30, 73, 54, 45, 83, 182, 88, 75, 85, 54, 53, 89, 59, 37, 35, 38, 29, 18, 45, 60, 49, 62, 55, 78, 96, 29, 22, 24, 13, 14, 11, 11, 18, 12, 12, 30, 52, 52, 44, 28, 28, 20, 56, 40, 31, 50, 40, 46, 42, 29, 19, 36, 25, 22, 17, 19, 26, 30, 20, 15, 21, 11, 8, 8, 19, 5, 8, 8, 11, 11, 8, 3, 9, 5, 4, 7, 3, 6, 3, 5, 4, 5, 6]

# data = []
# global_ayah_index = 0

       
# print("Script file:", Path(__file__).resolve())
# print("Script dir:", script_dir)
# print("Hussary dir:", hussary_directory)
# print("Exists:", hussary_directory.exists())

# print("Processing Audio and aligning with Text...")

# # Loop through Surahs
# for surah_idx, ayah_count in enumerate(tqdm(surah_ayah_counts), 1):
#     for ayah_idx in range(1, ayah_count + 1):
        
#         # Define expected filename structure
#         # You must manually rename your chaotic files to this format first: 114006.mp3 -> 114_006.mp3
#         # Or adjust this line to match your specific file naming convention
#         raw_filename = f"{surah_idx:03d}{ayah_idx:03d}.mp3" 
#         input_path = os.path.join(hussary_directory, raw_filename)
        
#         output_filename = f"{surah_idx:03d}_{ayah_idx:03d}.wav"
#         output_path = os.path.join(hussary_16k_directory, output_filename)
 
#         # Get the text for this ayah
#         if global_ayah_index < len(lines):
#             text = lines[global_ayah_index]
#         else:
#             text = ""
            
#         if os.path.exists(input_path):
#             # --- AUDIO PROCESSING ---
#             # Load and resample to 16kHz
#             y, sr = librosa.load(input_path, sr=16000, mono=True)
            
#             # Trim Silence (Optional but recommended for industry quality)
#             # This removes dead air at start/end
#             y, _ = librosa.effects.trim(y, top_db=20)
            
#             # Save processed file
#             sf.write(output_path, y, 16000)
            
#             # Add to data list
#             data.append({
#                 "path": output_path,
#                 "text": text,
#                 "surah": surah_idx,
#                 "ayah": ayah_idx,
#                 "duration": len(y)/16000
#             })
#         else:
#             # Log missing files so you know your dataset holes
#             print(f"Missing file: {raw_filename} at {input_path}")

#         global_ayah_index += 1

# # 3. Save Manifest
# df = pd.DataFrame(data)
# df.to_csv(csv_manifest, index=False)
# print(f"Done! Created {csv_manifest} with {len(df)} records.")