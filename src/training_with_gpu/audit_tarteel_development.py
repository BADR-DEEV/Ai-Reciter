"""Pinned unadapted/adapter controls and decoder ablations on DEVELOPMENT only.

Never opens test predictions or heldout voice audio. Every WER includes every
output, including flagged repetitive or context-exhausted transcripts.
"""
import argparse
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src/training"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.training_with_gpu.decoding_safety import load_private_adapter, DECODE_PROFILES
from src.training_with_gpu.train_base_full import configure_generation, TARTEEL_MODEL, TARTEEL_REVISION, AugmentedAyahDataset
from src.training_with_gpu.gpu_evaluation import evaluate_rows
from src.training_with_gpu.train_tarteel_lora import balanced_subset
from src.training.reviewed_audio import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--profiles", nargs="+", choices=DECODE_PROFILES, default=["greedy", "beam3", "beam3-soft", "beam5-block3"])
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()
    if args.output_dir.exists():
        raise ValueError("Preserve previous audits: output directory must be new")
    frozen = json.loads((args.run / "experiment_manifest.json").read_text(encoding="utf-8"))
    normalizer_version = frozen.get("normalizer_version", "qaloon-asr-v1")
    development = frozen["splits"]["validation"]
    for row in development:
        if sha256(row["path"]) != row["audio_sha256"]:
            raise ValueError("Frozen development audio changed")
    args.output_dir.mkdir(parents=True)
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from train_qaloon_lora import WhisperCollator
    report = {"scope": "development-only; no test tuning", "run_manifest_sha256": sha256(args.run / "experiment_manifest.json"),
        "profiles": {}, "control": {}, "flags_do_not_rewrite_or_remove_predictions": True,
        "no_fixed_60_token_limit": True, "n_gram_blocking_is_ablation_not_default": True}
    base = WhisperForConditionalGeneration.from_pretrained(TARTEEL_MODEL, revision=TARTEEL_REVISION, trust_remote_code=False).to("cuda")
    processor = WhisperProcessor.from_pretrained(TARTEEL_MODEL, revision=TARTEEL_REVISION, language="arabic", task="transcribe", trust_remote_code=False)
    processor.tokenizer.set_prefix_tokens(language="arabic", task="transcribe")
    configure_generation(base, processor)
    # Clean training fixtures for prefix/EOS/shift and initial teacher-forced loss.
    probe = balanced_subset(frozen["splits"]["train"], 8, 42)
    dataset = AugmentedAyahDataset(probe, processor, "text_asr_normalized")
    batch = WhisperCollator(processor, base.config.decoder_start_token_id)([dataset[i] for i in range(len(dataset))])
    eos = processor.tokenizer.eos_token_id
    lengths = batch["labels"].ne(-100).sum(dim=1)
    ends = [int(batch["labels"][i, int(length) - 1]) for i, length in enumerate(lengths)]
    if any(token != eos for token in ends):
        raise ValueError("EOS lost by label padding")
    with torch.inference_mode():
        initial_loss = float(base(**{k: v.to("cuda") for k, v in batch.items()}).loss)
    report["control"] = {"model_revision": TARTEEL_REVISION, "decoder_start_token_id": base.config.decoder_start_token_id,
        "eos_token_id": eos, "prefix_tokens": processor.tokenizer.prefix_tokens,
        "all_collated_labels_retain_eos": True, "bos_removed_once": True,
        "initial_clean_training_fixture_loss": initial_loss, "samples": len(probe)}
    base.eval()
    for profile in ["greedy", "beam3"]:
        scores, details = evaluate_rows(base, processor, development, args.batch_size, decode_profile=profile, normalizer_version=normalizer_version)
        name = f"unadapted_{profile}"
        report["profiles"][name] = scores
        (args.output_dir / f"{name}_predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in details), encoding="utf-8")
        print(name, scores["overall"], scores["decoding_safety"], flush=True)
    del base, batch, dataset
    gc.collect(); torch.cuda.empty_cache()
    model, processor = load_private_adapter(args.run / "adapter")
    for profile in args.profiles:
        scores, details = evaluate_rows(model, processor, development, args.batch_size, decode_profile=profile, normalizer_version=normalizer_version)
        report["profiles"][f"adapter_{profile}"] = scores
        (args.output_dir / f"adapter_{profile}_predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in details), encoding="utf-8")
        (args.output_dir / "development_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(profile, scores["overall"], scores["decoding_safety"], flush=True)
    print(f"Development audit complete: {args.output_dir}; no claimed repaired test WER.", flush=True)


if __name__ == "__main__":
    main()
