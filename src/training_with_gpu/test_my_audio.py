"""Test personal recitation on Full Model or LoRA adapter."""

import argparse
from difflib import SequenceMatcher
from pathlib import Path
import re
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio", type=Path, required=True, help="Path to your audio file")
    p.add_argument("--model-path", type=Path, default=Path("runs/gpu_base_full"), 
                   help="Path to trained model (Full model or LoRA dir)")
    p.add_argument("--base-model", type=str, default="openai/whisper-base")
    p.add_argument("--expected-text", type=str, default=None, help="The correct Quranic ayah text")
    p.add_argument("--prompt", type=str, default="بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ")
    p.add_argument("--beams", type=int, default=5, help="Beam search size (default: 5)")
    p.add_argument("--compare-zero-shot", action="store_true", help="Compare with vanilla Whisper-base")
    return p.parse_args()


def normalize_quran_text(text: str) -> str:
    text = re.sub(r"[\u064B-\u0652\u0670]", "", text)
    text = re.sub(r"[\u06D6-\u06ED\u06E9۩ۚۖۗۘۙ]", "", text)
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = re.sub(r"ئ", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def load_and_preprocess_audio(audio_path: Path):
    audio, sample_rate = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        audio_tensor = torch.from_numpy(audio).float()
        audio_tensor = F.resample(audio_tensor, sample_rate, 16000)
        audio = audio_tensor.numpy()
    duration = len(audio) / 16000
    if duration > 30.0:
        audio = audio[: 16000 * 30]
    return audio


def transcribe(model, processor, audio, device, prompt_text=None, num_beams=5):
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    gen_kwargs = {
        "language": "arabic",
        "task": "transcribe",
        "num_beams": num_beams,
    }
    if prompt_text:
        prompt_ids = processor.get_prompt_ids(prompt_text, return_tensors="pt").to(device)
        gen_kwargs["prompt_ids"] = prompt_ids

    with torch.no_grad():
        predicted_ids = model.generate(input_features, **gen_kwargs)
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


def fuzzy_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def analyze_omissions(expected: str, predicted: str):
    norm_exp = normalize_quran_text(expected).split()
    norm_pred = normalize_quran_text(predicted).split()
    
    true_omissions = []
    pronunciation_slips = []
    correct_matches = []
    
    for exp_w in norm_exp:
        best_match = None
        best_score = 0.0
        for pred_w in norm_pred:
            sim = fuzzy_similarity(exp_w, pred_w)
            if pred_w.endswith("ه") and fuzzy_similarity(exp_w, pred_w[:-1]) > sim:
                sim = fuzzy_similarity(exp_w, pred_w[:-1])
            if sim > best_score:
                best_score = sim
                best_match = pred_w
        
        if best_score >= 0.80:
            correct_matches.append(exp_w)
        elif best_score >= 0.45:
            pronunciation_slips.append(f"{exp_w} ──► heard: '{best_match}' (match: {int(best_score*100)}%)")
        else:
            true_omissions.append(exp_w)

    accuracy = (len(correct_matches) / max(1, len(norm_exp))) * 100
    return {
        "accuracy": accuracy,
        "correct": correct_matches,
        "slips": pronunciation_slips,
        "true_omissions": true_omissions,
    }


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio = load_and_preprocess_audio(args.audio)

    # Optional Zero-Shot check
    if args.compare_zero_shot:
        print("Transcribing with Vanilla Whisper-Base (Zero-Shot)...")
        zero_processor = WhisperProcessor.from_pretrained(args.base_model)
        zero_model = WhisperForConditionalGeneration.from_pretrained(args.base_model).to(device)
        zero_shot_result = transcribe(zero_model, zero_processor, audio, device, num_beams=1)
        print(f"\n[Vanilla Base]: {zero_shot_result}\n")

    # Load Full Model or LoRA
    adapter_config = args.model_path / "adapter_config.json"
    if adapter_config.exists():
        print(f"Loading base model + LoRA adapter from: {args.model_path} ...")
        processor = WhisperProcessor.from_pretrained(args.base_model)
        base = WhisperForConditionalGeneration.from_pretrained(args.base_model).to(device)
        model = PeftModel.from_pretrained(base, str(args.model_path)).to(device)
    else:
        print(f"🔥 Loading FULL FINE-TUNED MODEL from: {args.model_path} ...")
        processor = WhisperProcessor.from_pretrained(str(args.model_path))
        model = WhisperForConditionalGeneration.from_pretrained(str(args.model_path)).to(device)
    
    model.eval()
    print(f"Transcribing (Beams: {args.beams}, Prompt: '{args.prompt}')...")
    result = transcribe(model, processor, audio, device, prompt_text=args.prompt, num_beams=args.beams)

    print("\n" + "=" * 65)
    print(f"🎙️  Model Output:     {result}")
    
    if args.expected_text:
        print(f"📖 Expected Ayah:    {args.expected_text}")
        analysis = analyze_omissions(args.expected_text, result)
        print("-" * 65)
        print(f"📊 Match Accuracy:   {analysis['accuracy']:.1f}%")
        if analysis["true_omissions"]:
            print(f"🚨 True Omissions:   {analysis['true_omissions']}")
        else:
            print("✅ No missing words! (All words detected)")
        if analysis["slips"]:
            print(f"⚠️  Pronunciation:   {analysis['slips']}")
    print("=" * 65)


if __name__ == "__main__":
    main()