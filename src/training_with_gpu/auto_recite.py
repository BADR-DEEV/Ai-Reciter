"""Automatic Quran Reciter: Auto-detects Surah & Ayah, then checks omissions."""

import argparse
from difflib import SequenceMatcher
from pathlib import Path
import re
import sys
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load existing project dataset to get all known Ayahs
TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
sys.path.insert(0, str(TRAINING_DIR))
from qaloon_data import DATA_ROOT, load_splits


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio", type=Path, required=True, help="Path to your audio file")
    p.add_argument("--model-path", type=Path, default=Path("runs/gpu_base_full"))
    p.add_argument("--beams", type=int, default=5)
    return p.parse_args()


def normalize_quran_text(text: str) -> str:
    text = re.sub(r"[\u064B-\u0652\u0670]", "", text)
    text = re.sub(r"[\u06D6-\u06ED\u06E9۩ۚۖۗۘۙ]", "", text)
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = re.sub(r"ئ", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def load_mushaf_database():
    """Build an in-memory dictionary of all ayahs from the dataset metadata."""
    splits, _ = load_splits(DATA_ROOT, None, None, seed=42)
    mushaf = {}
    
    # Collect unique ayahs across all splits
    for partition in splits.values():
        for row in partition:
            key = (row["surah"], row["ayah"])
            if key not in mushaf:
                mushaf[key] = {
                    "surah": row["surah"],
                    "ayah": row["ayah"],
                    "raw_text": row.get("text_raw", row["text_asr_normalized"]),
                    "normalized": normalize_quran_text(row["text_asr_normalized"]),
                }
    return list(mushaf.values())


def find_best_matching_ayah(transcription: str, mushaf: list):
    """Find the best matching Surah & Ayah in the Quran database in milliseconds."""
    norm_transcript = normalize_quran_text(transcription)
    best_match = None
    best_ratio = 0.0

    for item in mushaf:
        # Calculate text similarity
        ratio = SequenceMatcher(None, norm_transcript, item["normalized"]).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = item

    return best_match, best_ratio


def load_and_preprocess_audio(audio_path: Path):
    audio, sample_rate = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        audio_tensor = torch.from_numpy(audio).float()
        audio_tensor = F.resample(audio_tensor, sample_rate, 16000)
        audio = audio_tensor.numpy()
    if len(audio) / 16000 > 30.0:
        audio = audio[: 16000 * 30]
    return audio


def analyze_omissions(expected_norm: str, predicted_norm: str):
    exp_words = expected_norm.split()
    pred_words = predicted_norm.split()
    
    true_omissions = []
    pronunciation_slips = []
    correct_matches = []
    
    for exp_w in exp_words:
        best_match = None
        best_score = 0.0
        for pred_w in pred_words:
            sim = SequenceMatcher(None, exp_w, pred_w).ratio()
            if pred_w.endswith("ه") and SequenceMatcher(None, exp_w, pred_w[:-1]).ratio() > sim:
                sim = SequenceMatcher(None, exp_w, pred_w[:-1]).ratio()
            if sim > best_score:
                best_score = sim
                best_match = pred_w
        
        if best_score >= 0.80:
            correct_matches.append(exp_w)
        elif best_score >= 0.45:
            pronunciation_slips.append(f"{exp_w} (heard: '{best_match}')")
        else:
            true_omissions.append(exp_w)

    accuracy = (len(correct_matches) / max(1, len(exp_words))) * 100
    return {
        "accuracy": accuracy,
        "correct": correct_matches,
        "slips": pronunciation_slips,
        "true_omissions": true_omissions,
    }


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading local Mushaf database...")
    mushaf = load_mushaf_database()
    print(f"Indexed {len(mushaf)} Quranic verses in memory.")

    print(f"Loading audio: {args.audio} ...")
    audio = load_and_preprocess_audio(args.audio)

    print(f"Loading full fine-tuned model: {args.model_path} ...")
    processor = WhisperProcessor.from_pretrained(str(args.model_path))
    model = WhisperForConditionalGeneration.from_pretrained(str(args.model_path)).to(device)
    model.eval()

    # Step 1: Transcribe audio blind (No expected text, no prompt)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    print("Transcribing blind audio with Whisper...")
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features, 
            language="arabic", 
            task="transcribe", 
            num_beams=args.beams
        )
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

    # Step 2: Auto-detect which Ayah this is
    best_ayah, confidence = find_best_matching_ayah(transcription, mushaf)

    print("\n" + "=" * 65)
    print(f"🎙️  What Whisper Heard: {transcription}")
    print("-" * 65)
    
    if best_ayah and confidence > 0.40:
        print(f"🔍 AUTO-DETECTED AYAH: Surah {best_ayah['surah']}, Ayah {best_ayah['ayah']}")
        print(f"📖 Database Text:     {best_ayah['raw_text']}")
        print(f"🎯 Detection Match:    {confidence * 100:.1f}%")
        print("-" * 65)
        
        # Step 3: Grade omissions against the auto-detected Ayah
        analysis = analyze_omissions(best_ayah["normalized"], normalize_quran_text(transcription))
        print(f"📊 Recitation Score:   {analysis['accuracy']:.1f}%")
        
        if analysis["true_omissions"]:
            print(f"🚨 Missing / Omitted:  {analysis['true_omissions']}")
        else:
            print("✅ Perfect! No missing words detected.")
            
        if analysis["slips"]:
            print(f"⚠️  Pronunciation:     {analysis['slips']}")
    else:
        print("❓ Could not confidently identify the Ayah. Please recite clearly.")
    print("=" * 65)


if __name__ == "__main__":
    main()