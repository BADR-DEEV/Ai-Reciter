"""Same-audio paired comparison, with explicit historical-baseline contamination."""
import argparse
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.training.reviewed_audio import sha256
from src.training.asr_metrics import paired_comparison


def compare_existing(run, baseline, batch_size=4):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from gpu_evaluation import evaluate_rows
    from train_base_full import configure_generation
    from src.training.qaloon_data import load_splits
    run, baseline = Path(run), Path(baseline)
    if (run / "comparison.json").exists():
        raise ValueError("Comparison already exists; never overwrite a test result")
    manifest = json.loads((run / "experiment_manifest.json").read_text(encoding="utf-8"))
    partitions = [name for name in ("test", "unheard_adaptation_reciter") if (run / f"{name}_predictions.jsonl").is_file()]
    if not partitions:
        raise ValueError("Complete saved candidate predictions required before comparison")
    model = WhisperForConditionalGeneration.from_pretrained(baseline, trust_remote_code=False).to("cuda")
    processor = WhisperProcessor.from_pretrained(baseline, trust_remote_code=False)
    configure_generation(model, processor)
    result = {"baseline": str(baseline.resolve()), "candidate": str(run.resolve()),
        "comparison_scope": "same-audio observations, NOT an initialization/method-only causal comparison",
        "historical_baseline_training_readers": ["huthaify", "dokali", "husary", "waleed"],
        "baseline_unheard_reciter_claim": False,
        "historical_scores_not_directly_rankable_against_new_protocol": True,
        "original_baseline_metrics": json.loads((baseline / "metrics.json").read_text(encoding="utf-8")),
        "manifest_sha256": sha256(run / "experiment_manifest.json")}
    result["normalizer_version"] = manifest.get("normalizer_version", "qaloon-asr-v1")
    result["decode_profile"] = manifest.get("settings", {}).get("decode_profile", "greedy")
    legacy, _ = load_splits(include=["huthaify", "dokali", "husary", "waleed"], seed=42)
    legacy_hashes = {sha256(r["path"]) for r in legacy["train"]}
    result["legacy_training_membership_note"] = "Reconstructed historical default seed42; original checkpoint has no embedded split manifest. Do not certify contamination-free from this inference."
    for partition in partitions:
        rows = manifest["splits"][partition]
        for row in rows:
            if sha256(row["path"]) != row["audio_sha256"]:
                raise ValueError("Evaluation WAV changed since the frozen run")
        scores, details = evaluate_rows(model, processor, rows, batch_size,
            decode_profile=result["decode_profile"], normalizer_version=result["normalizer_version"],
            predictions_path=run / f"baseline_{partition}_predictions.jsonl")
        candidate = [json.loads(l) for l in (run / f"{partition}_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        result[partition] = {"baseline_scores": scores, "paired": paired_comparison(details, candidate),
            "reconstructed_baseline_train_audio_overlap": sum(r["audio_sha256"] in legacy_hashes for r in rows),
            "unheard_for_candidate_adaptation_only": partition == "unheard_adaptation_reciter"}
    (run / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    del model
    gc.collect(); torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()
    compare_existing(args.run, args.baseline, args.batch_size)
