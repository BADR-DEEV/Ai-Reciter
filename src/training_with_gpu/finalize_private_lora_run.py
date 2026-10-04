"""Finish evaluation of an unchanged saved adapter; never fit or select weights.

Interrupted original runs and their outputs stay intact. The new recovery folder
records provenance and writes prediction batches incrementally. Saved test
results are descriptive regressions, not fresh confirmatory tests.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.training.reviewed_audio import sha256
from src.training.asr_metrics import paired_comparison
from src.training_with_gpu.decoding_safety import load_private_adapter
from src.training_with_gpu.gpu_evaluation import evaluate_rows, summarize_predictions


def verify_saved_predictions(details, rows, normalizer_version=None):
    """Require complete frozen ordering, exact references and PCM byte hashes."""
    from src.dataset_collection.qaloon_audio2text import normalizer_for_version
    normalize_quran_for_asr = normalizer_for_version(normalizer_version)
    if len(details) != len(rows):
        raise ValueError("Saved prediction count differs from frozen partition")
    for prediction, row in zip(details, rows):
        if ((prediction["reciter"], prediction["surah"], prediction["ayah"]) !=
                (row["reciter_key"], row["surah"], row["ayah"]) or
                prediction["reference"] != normalize_quran_for_asr(row["text_asr_normalized"]) or
                prediction.get("audio_sha256") != row["audio_sha256"] or
                prediction.get("text_seen_in_training") != row.get("text_seen_in_training")):
            raise ValueError("Saved prediction identity/reference/audio/text stratum differs")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()
    if args.output_dir.exists() or args.batch_size < 1:
        raise ValueError("New recovery folder and positive batch size required")
    manifest_path = args.run / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profile = manifest["settings"]["decode_profile"]
    checkpoints = []
    for path in args.run.glob("checkpoint-*/trainer_state.json"):
        state = json.loads(path.read_text(encoding="utf-8"))
        checkpoints.append((state["global_step"], state, path))
    _, state, state_path = max(checkpoints, key=lambda item: item[0])
    best = args.run / Path(state["best_model_checkpoint"]).name
    adapter_path = args.run / "adapter/adapter_model.safetensors"
    if sha256(adapter_path) != sha256(best / "adapter_model.safetensors"):
        raise ValueError("Exported adapter differs from frozen validation-selected checkpoint")
    # All audio checked, not just the eight reload fixtures. No data relabelling.
    for partition in manifest["splits"].values():
        for row in partition:
            if sha256(row["path"]) != row["audio_sha256"]:
                raise ValueError("Frozen PCM changed")
    results = {"original_run": str(args.run.resolve()), "manifest_sha256": sha256(manifest_path),
        "original_process_status": "exited-code-1-after-export; no recorded traceback; cause-unresolved",
        "scope": "evaluation-only recovery; no fitting, checkpoint selection, decoding tuning or original-file overwrite",
        "adapter_sha256": sha256(adapter_path), "adapter_matches_best_checkpoint": True,
        "last_optimizer_step": state["global_step"], "last_epoch": state["epoch"],
        "best_checkpoint": str(best), "best_validation_macro_reciter_wer": state["best_metric"],
        "decode_profile": profile, "heldout_reader": manifest["heldout_reader"],
        "upstream_exposure_unknown": True, "is_smoke_run": manifest["is_smoke_run"],
        "test_scope": "descriptive regression; prior holdouts observed; no fresh confirmatory claim",
        "production_approved": False, "human_learner_validation": "not-established",
        "generalization_history": json.loads((args.run / "generalization_history.json").read_text(encoding="utf-8")),
        "unadapted_development_control": json.loads((args.run / "unadapted_development_control.json").read_text(encoding="utf-8")),
        "recovery_status": "in-progress"}
    args.output_dir.mkdir(parents=True)
    def persist():
        (args.output_dir / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "training_log_from_last_checkpoint.json").write_text(json.dumps(state["log_history"], indent=2), encoding="utf-8")
    references = {}
    for name in ("validation", "test"):
        path = args.run / f"{name}_predictions.jsonl"
        details = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        verify_saved_predictions(details, manifest["splits"][name], manifest.get("normalizer_version"))
        references[name] = details
        results[name] = summarize_predictions(details, profile)
        results[name]["evidence"] = {"predictions_path": str(path.resolve()), "predictions_sha256": sha256(path),
            "latency_unavailable": "Timing report was not persisted before original process exited"}
    original_base = [json.loads(line) for line in (args.run / "unadapted_development_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    verify_saved_predictions(original_base, manifest["splits"]["validation"], manifest.get("normalizer_version"))
    results["paired_development_comparison"] = paired_comparison(original_base, references["validation"])
    persist()
    print("Verified selected checkpoint and complete preserved validation/test predictions.", flush=True)
    model, processor = load_private_adapter(args.run / "adapter")
    if profile != json.loads((args.run / "adapter/decoding_policy.json").read_text(encoding="utf-8"))["profile"]:
        raise ValueError("Frozen manifest decoder differs from exported adapter policy")
    _, replay = evaluate_rows(model, processor, manifest["splits"]["validation"][:8], args.batch_size, decode_profile=profile,
        predictions_path=args.output_dir / "reload_development_predictions.jsonl", normalizer_version=manifest.get("normalizer_version", "qaloon-asr-v1"))
    verify_saved_predictions(replay, manifest["splits"]["validation"][:8], manifest.get("normalizer_version"))
    matched = all(a["prediction"] == b["prediction"] and a["raw_prediction"] == b["raw_prediction"] and
        a["decode_flags"] == b["decode_flags"] and a["generated_tokens"] == b["generated_tokens"]
        for a, b in zip(references["validation"][:8], replay))
    results["saved_adapter_reload"] = {"development_samples": len(replay), "same_predictions_and_audio": matched,
        "same_raw_text_flags_and_token_counts": matched, "pinned_base_and_saved_generation_config": True}
    persist()
    if not matched:
        raise ValueError("Reload fixture mismatch; do not claim stable export")
    print("Reload exactly matched all eight development fixtures. Evaluating untouched Waleed partition.", flush=True)
    heldout, details = evaluate_rows(model, processor, manifest["splits"]["unheard_adaptation_reciter"], args.batch_size,
        decode_profile=profile, predictions_path=args.output_dir / "unheard_adaptation_reciter_predictions.jsonl",
        normalizer_version=manifest.get("normalizer_version", "qaloon-asr-v1"))
    verify_saved_predictions(details, manifest["splits"]["unheard_adaptation_reciter"], manifest.get("normalizer_version"))
    results["unheard_adaptation_reciter"] = heldout
    base = results["unadapted_development_control"]
    results["readiness"] = {"saved_reload_matches": matched,
        "development_macro_wer_not_worse_than_unadapted": results["validation"]["diagnostics"]["macro_reciter_wer"] <= base["diagnostics"]["macro_reciter_wer"],
        "development_cer_not_worse_than_unadapted": results["validation"]["overall"]["cer"] <= base["overall"]["cer"],
        "development_deletions_not_worse_than_unadapted": results["validation"]["overall"]["word_deletions"] <= base["overall"]["word_deletions"],
        "no_flagged_development_decodes": results["validation"]["decoding_safety"]["flagged_samples"] == 0,
        "qualified_unfamiliar_learner_error_validation": False, "fresh_confirmatory_test": False,
        "public_release_source_clearance": False, "human_ready_or_deployment_approved": False}
    results["recovery_status"] = "evaluation-complete; original exit preserved"
    persist()
    for name in ("validation", "test", "unheard_adaptation_reciter"):
        print(name, results[name]["overall"], results[name]["decoding_safety"], flush=True)
    print(f"Completed evaluation-only recovery: {args.output_dir}; unchanged saved adapter, no deployment.", flush=True)


if __name__ == "__main__":
    main()
