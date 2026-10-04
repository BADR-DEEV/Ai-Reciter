"""Blind personal-audio ASR diagnostics, NOT pronunciation/omission certification."""

import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.dataset_collection.qaloon_audio2text import normalize_quran_for_asr
from src.training_with_gpu.decoding_safety import generation_diagnostics, load_private_adapter
from src.training_with_gpu.train_base_full import configure_generation, MODEL_REVISIONS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio", type=Path, required=True, help="Path to your audio file")
    p.add_argument("--model-path", type=Path, default=Path("runs/gpu_base_full"), 
                   help="Path to trained model (Full model or LoRA dir)")
    p.add_argument("--base-model", type=str, default="openai/whisper-base")
    p.add_argument("--expected-text", type=str, default=None, help="The correct Quranic ayah text")
    p.add_argument("--prompt", type=str, default=None, help="Deprecated: prompts are rejected for unbiased learner assessment")
    p.add_argument("--beams", type=int, default=3, help="Beam search size (default: 3)")
    p.add_argument("--compare-zero-shot", action="store_true", help="Compare with vanilla Whisper-base")
    args = p.parse_args()
    if args.prompt:
        p.error("Expected/preamble prompts are not permitted in blind learner assessment")
    if args.beams < 1:
        p.error("Require positive beam count")
    return args


def normalize_quran_text(text: str) -> str:
    return normalize_quran_for_asr(text)


def load_and_preprocess_audio(audio_path: Path):
    audio, sample_rate = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        audio_tensor = torch.from_numpy(audio).float()
        audio_tensor = F.resample(audio_tensor, sample_rate, 16000)
        audio = audio_tensor.numpy()
    duration = len(audio) / 16000
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Audio must be nonempty and finite")
    if duration > 30.0:
        raise ValueError("Audio exceeds 30 seconds; use windowed studio decoding. Never silently truncate a complete recitation.")
    return audio


def transcribe(model, processor, audio, device, prompt_text=None, num_beams=3, return_diagnostics=False):
    if prompt_text:
        raise ValueError("No expected-text prompts in blind assessment")
    inputs = processor(audio, sampling_rate=16000, return_attention_mask=True, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    gen_kwargs = {
        "language": "arabic",
        "task": "transcribe",
        "num_beams": num_beams,
        "max_length": model.config.max_target_positions,
        "use_cache": True,
        "return_dict_in_generate": True,
        "attention_mask": inputs.attention_mask.to(device),
    }

    with torch.no_grad():
        predicted_ids = model.generate(input_features, **gen_kwargs).sequences
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    diagnostics = generation_diagnostics(predicted_ids[0].tolist(), processor.tokenizer, model.config.max_target_positions)
    return (text, diagnostics) if return_diagnostics else text


def analyze_omissions(expected: str, predicted: str):
    # Historical function name retained; output is sequence-aware ASR diagnostics.
    import jiwer
    reference, hypothesis = normalize_quran_text(expected), normalize_quran_text(predicted)
    if not reference:
        raise ValueError("Expected comparison text must be nonempty")
    alignment = jiwer.process_words(reference, hypothesis)
    return {"wer": alignment.wer, "exact_normalized_text_match": reference == hypothesis,
        "apparent_deletions": alignment.deletions, "substitutions": alignment.substitutions,
        "insertions": alignment.insertions,
        "warning": "Text recognition differences do not certify learner omissions, pronunciation or tajweed."}


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio = load_and_preprocess_audio(args.audio)

    # Optional Zero-Shot check
    if args.compare_zero_shot:
        print("Transcribing with Vanilla Whisper-Base (Zero-Shot)...")
        revision = MODEL_REVISIONS.get(args.base_model)
        zero_processor = WhisperProcessor.from_pretrained(args.base_model, revision=revision)
        zero_model = WhisperForConditionalGeneration.from_pretrained(args.base_model, revision=revision).to(device)
        zero_processor.tokenizer.set_prefix_tokens(language="arabic", task="transcribe")
        configure_generation(zero_model, zero_processor)
        zero_model.eval()
        zero_shot_result = transcribe(zero_model, zero_processor, audio, device, num_beams=1)
        print(f"\n[Vanilla Base]: {zero_shot_result}\n")
        del zero_model
        torch.cuda.empty_cache()

    # Load Full Model or LoRA
    adapter_config = args.model_path / "adapter_config.json"
    if adapter_config.exists():
        print(f"Loading base model + LoRA adapter from: {args.model_path} ...")
        adapter = json.loads(adapter_config.read_text(encoding="utf-8"))
        if adapter.get("base_model_name_or_path") == "tarteel-ai/whisper-base-ar-quran":
            model, processor = load_private_adapter(args.model_path, device)
        else:
            base_id = adapter.get("base_model_name_or_path", args.base_model)
            revision = adapter.get("revision") or MODEL_REVISIONS.get(base_id)
            processor = WhisperProcessor.from_pretrained(args.model_path)
            base = WhisperForConditionalGeneration.from_pretrained(base_id, revision=revision).to(device)
            configure_generation(base, processor)
            model = PeftModel.from_pretrained(base, str(args.model_path)).to(device)
    else:
        print(f"🔥 Loading FULL FINE-TUNED MODEL from: {args.model_path} ...")
        processor = WhisperProcessor.from_pretrained(str(args.model_path))
        model = WhisperForConditionalGeneration.from_pretrained(str(args.model_path)).to(device)
        configure_generation(model, processor)
    
    model.eval()
    processor.tokenizer.set_prefix_tokens(language="arabic", task="transcribe")
    print(f"Blind transcription (Beams: {args.beams}; no expected-text prompt)...")
    if np.sqrt(np.mean(audio ** 2)) < 1e-5:
        print("ABSTAIN: near-silent recording; no recitation score.")
        return
    result, diagnostics = transcribe(model, processor, audio, device, num_beams=args.beams, return_diagnostics=True)

    print("\n" + "=" * 65)
    print(f"🎙️  Model Output:     {result}")
    print("Decode diagnostics:", diagnostics)
    if not diagnostics["scorable"]:
        print("ABSTAIN: unreliable model decoding. Re-record or ask a teacher; no learner mistake is inferred.")
        return
    
    if args.expected_text:
        print(f"📖 Expected Ayah:    {args.expected_text}")
        analysis = analyze_omissions(args.expected_text, result)
        print("-" * 65)
        print("ASR text diagnostics:", analysis)
    print("=" * 65)


if __name__ == "__main__":
    main()
