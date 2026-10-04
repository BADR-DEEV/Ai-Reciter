"""Clean TRAINING-only target-format loss audit; never relabel from ASR output."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src/training"))
from src.training_with_gpu.train_base_full import TARTEEL_MODEL, TARTEEL_REVISION, configure_generation
from src.training_with_gpu.train_tarteel_lora import balanced_subset
from src.dataset_collection.qaloon_audio2text import load_quran, normalize_with_harakat


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Preserve prior audit outputs")
    import torch
    import soundfile as sf
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from train_qaloon_lora import WhisperCollator
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = balanced_subset(manifest["splits"]["train"], 8, 42)
    canonical = load_quran()
    model = WhisperForConditionalGeneration.from_pretrained(TARTEEL_MODEL, revision=TARTEEL_REVISION).to("cuda").eval()
    processor = WhisperProcessor.from_pretrained(TARTEEL_MODEL, revision=TARTEEL_REVISION, language="arabic", task="transcribe")
    processor.tokenizer.set_prefix_tokens(language="arabic", task="transcribe")
    configure_generation(model, processor)
    report = {"scope": "training-only diagnostics; not permission to use ASR-predicted training labels", "cases": []}
    for row in rows:
        audio, sr = sf.read(row["path"], dtype="float32")
        inputs = processor.feature_extractor(audio, sampling_rate=sr, return_tensors="pt", return_attention_mask=True)
        with torch.inference_mode():
            ids = model.generate(inputs.input_features.to("cuda"), attention_mask=inputs.attention_mask.to("cuda"),
                language="arabic", task="transcribe", num_beams=1, return_dict_in_generate=True, max_length=448).sequences
        raw = processor.tokenizer.decode(ids[0], skip_special_tokens=True)
        targets = {"text_asr_normalized": row["text_asr_normalized"],
            "canonical_with_harakat": normalize_with_harakat(canonical[(row["surah"], row["ayah"])]["raw"]),
            "blind_asr_diagnostic_NOT_training_target": raw}
        losses = {}
        for name, text in targets.items():
            collated = WhisperCollator(processor, model.config.decoder_start_token_id)([{"input_features": inputs.input_features[0].numpy(),
                "attention_mask": inputs.attention_mask[0].numpy(), "labels": processor.tokenizer(text).input_ids}])
            with torch.inference_mode():
                losses[name] = float(model(**{k: v.to("cuda") for k, v in collated.items()}).loss)
        report["cases"].append({"reciter": row["reciter_key"], "surah": row["surah"], "ayah": row["ayah"],
            "targets": targets, "losses": losses, "raw_sequence_start": ids[0, :8].tolist(), "raw_sequence_end": ids[0, -8:].tolist()})
    report["mean_losses"] = {name: sum(case["losses"][name] for case in report["cases"]) / len(rows) for name in targets}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TRAINING-fixture target-format losses:", report["mean_losses"], flush=True)
