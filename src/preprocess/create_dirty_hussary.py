from pathlib import Path
import random
import pandas as pd
# Assume we have the dataframe 'df' from the previous step
mistake_data = []

script_dir = Path(__file__).resolve().parent.parent.parent # ousdie to root
csv_manifest = script_dir / "data" / "train.csv"
csv_mistakes = script_dir / "data" / "mistakes.csv"

print(f"Manifest: {csv_manifest}")
df = pd.read_csv(csv_manifest)

for index, row in df.iterrows():
    # Only generate mistakes for 20% of the data
    if random.random() > 0.2:
        continue

    # Strategy 1: The "Wrong Text" (Hallucination simulation)
    # We keep the audio correct, but we CHANGE the text label.
    # This teaches the model: "When you hear X, it is definitely NOT Y."
    
    # Example: Delete a random word from the text
    words = row['text'].split()
    if len(words) > 3:
        word_to_remove = random.randint(0, len(words)-1)
        # Store the wrong text
        bad_text = words[:word_to_remove] + words[word_to_remove+1:]
        bad_text_str = " ".join(bad_text)
        
        mistake_data.append({
            "path": row['path'], # Correct Audio
            "text": bad_text_str, # Wrong Text (Missing word)
            "label": "missing_word" 
        })

# Save mistakes manifest
df_mistakes = pd.DataFrame(mistake_data)
df_mistakes.to_csv(csv_mistakes, index=False)