"""Download a Qaloon model variant and transcribe a short recording.

python transcribe_hub.py clip.wav --variant full-finetuning
python transcribe_hub.py clip.wav --variant base-lora-v2
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as audio_functional
from huggingface_hub import snapshot_download

REPO_ID = "BadrSh/qalon-reciter"
VARIANTS = {
    "full-finetuning": ("models/full-finetuning", None),
    "base-lora-v2": ("models/base-lora-v2", "openai/whisper-base"),
    "legacy-tiny-lora": ("legacy/whisper-tiny-lora", "openai/whisper-tiny"),
}


def load_variant(variant="full-finetuning", repo_id=REPO_ID, revision="main", device=None):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    subfolder, base_id = VARIANTS[variant]
    snapshot = snapshot_download(repo_id, revision=revision, allow_patterns=[f"{subfolder}/*"])
    folder = Path(snapshot) / subfolder
    processor = WhisperProcessor.from_pretrained(str(folder))
    if base_id:
        from peft import PeftModel
        base = WhisperForConditionalGeneration.from_pretrained(base_id)
        model = PeftModel.from_pretrained(base, str(folder))
    else:
        model = WhisperForConditionalGeneration.from_pretrained(str(folder))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.config.use_cache = True
    return processor, model.to(device).eval(), device


def transcribe(path, processor, model, device):
    audio, sample_rate = sf.read(str(path), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError("Choose a nonempty recording with finite samples.")
    if len(audio) / sample_rate > 30:
        raise ValueError("Use a clip of at most 30 seconds; segment longer recordings first.")
    if sample_rate != 16000:
        # torchaudio applies an anti-aliasing filter when downsampling.
        audio = audio_functional.resample(torch.from_numpy(audio), sample_rate, 16000).numpy()
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    with torch.inference_mode():
        ids = model.generate(input_features=inputs.input_features.to(device),
                             attention_mask=inputs.attention_mask.to(device),
                             language="arabic", task="transcribe", num_beams=1, do_sample=False)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--variant", choices=VARIANTS, default="full-finetuning")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--revision", default="main", help="Pin a commit hash for reproducible downloads")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    args = parser.parse_args()
    processor, model, device = load_variant(args.variant, args.repo_id, args.revision, args.device)
    print(transcribe(args.audio, processor, model, device))


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
