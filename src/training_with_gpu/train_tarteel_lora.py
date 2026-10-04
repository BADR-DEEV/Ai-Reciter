"""Private LoRA adaptation with recording-safe splits and an unheard voice test.

Candidate acceptance is explicitly experimental, never production certification.
The held-out reader is never used for training, selection, sampling or stopping.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src/training"))
from src.training.qaloon_data import load_splits, DATA_ROOT
from src.training.experiment_data import experimental_candidates, recording_safe_splits
from src.training.reviewed_audio import sha256
from train_base_full import configure_generation, AugmentedAyahDataset, TARTEEL_MODEL, MODEL_REVISIONS
from src.training_with_gpu.decoding_safety import DECODE_PROFILES, save_decoding_bundle, load_private_adapter


def parse_args(argv=None, recipe=None, allowed_models=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--init-model", choices=allowed_models or MODEL_REVISIONS, default=TARTEEL_MODEL)
    p.add_argument("--supplement", nargs=2, action="append", default=[], metavar=("DATASET", "EXPERIMENT_DECISION"))
    p.add_argument("--holdout-reciter", choices=["waleed"], default="waleed")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, help="Default twice the rank")
    p.add_argument("--target-modules", nargs="+", choices=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"], default=["q_proj", "v_proj"])
    p.add_argument("--repair-normalization", action="store_true", help="Versioned canonical v2 label overlay; original metadata unchanged")
    p.add_argument("--quarantine-manifest", type=Path, help="Hash-bound training-only exclusions, never a fabricated 28-clip list")
    p.add_argument("--require-training-audit", action="store_true")
    p.add_argument("--tempo-min", type=float)
    p.add_argument("--tempo-max", type=float)
    p.add_argument("--precision", choices=["auto", "bf16", "fp16"], default="auto")
    p.add_argument("--skip-heldout-eval", action="store_true", help="Fit/choose on development only; final heldout evaluation is a separate fixed run")
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--decode-profile", choices=DECODE_PROFILES, default="beam3")
    p.add_argument("--eval-steps", type=int, default=39, help="Generated development evaluation/save interval; 0 uses epochs")
    p.add_argument("--noise-prob", type=float, default=.05)
    p.add_argument("--speed-prob", type=float, default=.1)
    p.add_argument("--label-field", choices=["text_asr_normalized", "normalized_with_harakat"], default="normalized_with_harakat")
    p.add_argument("--supplement-weight", type=float, default=.5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--max-samples-per-split", type=int)
    p.add_argument("--baseline-model-dir", type=Path)
    p.add_argument("--dry-run", action="store_true")
    if recipe:
        p.set_defaults(**recipe)
    args = p.parse_args(argv)
    args.alpha = 2 * args.rank if args.alpha is None else args.alpha
    if (min(args.epochs, args.patience, args.rank, args.batch_size, args.eval_batch_size, args.gradient_accumulation) <= 0
            or not np.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not np.isfinite(args.supplement_weight) or not 0 < args.supplement_weight <= 1
            or args.max_steps == 0 or args.max_steps < -1
            or args.eval_steps < 0 or not 0 <= args.noise_prob <= 1 or not 0 <= args.speed_prob <= 1
            or args.max_samples_per_split is not None and args.max_samples_per_split <= 0):
        p.error("Require positive finite settings; max-steps is -1 or positive, supplement weight <=1")
    if args.alpha <= 0 or bool(args.tempo_min is None) != bool(args.tempo_max is None):
        p.error("Positive alpha and both tempo bounds required")
    if args.tempo_min is not None and not 0 < args.tempo_min <= args.tempo_max < float("inf"):
        p.error("Positive finite ordered tempo range required")
    if args.require_training_audit and not args.quarantine_manifest:
        p.error("Provide the actual local --quarantine-manifest; no screenshot exclusions can be invented")
    return args


def balanced_subset(rows, limit, seed):
    if not limit or len(rows) <= limit:
        return rows
    rng = np.random.default_rng(seed)
    groups = {r: list(rng.permutation([x for x in rows if x["reciter_key"] == r])) for r in sorted({x["reciter_key"] for x in rows})}
    selected = []
    while groups and len(selected) < limit:
        for reader in list(groups):
            selected.append(groups[reader].pop())
            if not groups[reader]:
                del groups[reader]
            if len(selected) == limit:
                break
    return selected


def canonical_training_targets(rows, normalizer_version="qaloon-asr-v2-vocative-consonantal-yaa"):
    """Preserve supplied Qaloon vowels, never import model-generated/Hafs targets."""
    from src.dataset_collection.qaloon_audio2text import load_quran, normalize_with_harakat, normalize_quran_for_asr
    canonical = load_quran()
    from src.dataset_collection.qaloon_audio2text import normalizer_for_version
    normalize_quran_for_asr = normalizer_for_version(normalizer_version)
    for row in rows:
        target = normalize_with_harakat(canonical[(row["surah"], row["ayah"])]["raw"])
        # Derive both labels from the ORIGINAL source. NFC/combining-mark cleaning
        # need not commute with hamza/madd stripping for a few Uthmani spellings.
        if normalize_quran_for_asr(canonical[(row["surah"], row["ayah"])]["raw"]) != row["text_asr_normalized"]:
            raise ValueError(f"Canonical vocalized target conflicts with intact audio label: {row['reciter_key']} {row['surah']}:{row['ayah']}")
        row["normalized_with_harakat"] = target
    return rows


def main(argv=None, recipe=None, allowed_models=None):
    args = parse_args(argv, recipe, allowed_models)
    core, skipped = load_splits(args.data_root, ["huthaify", "dokali", "husary"], seed=args.seed)
    rows = [row for partition in core.values() for row in partition]
    for dataset, decision in args.supplement:
        rows.extend(experimental_candidates(dataset, decision))
    from src.dataset_collection.qaloon_audio2text import NORMALIZER_VERSION, normalizer_for_version
    from src.training.label_integrity import corrected_label_overlay, apply_training_quarantine
    normalizer_version = NORMALIZER_VERSION if args.repair_normalization else "qaloon-asr-v1"
    label_changes = []
    if args.repair_normalization:
        rows, label_changes = corrected_label_overlay(rows)
    if args.label_field == "normalized_with_harakat":
        canonical_training_targets(rows, normalizer_version)
    splits = recording_safe_splits(rows, args.seed)
    quarantine = None
    if args.quarantine_manifest:
        splits, quarantine = apply_training_quarantine(splits, args.quarantine_manifest)
    withheld, heldout_skipped = load_splits(args.data_root, [args.holdout_reciter], seed=args.seed)
    heldout = [row for partition in withheld.values() for row in partition]
    if args.repair_normalization:
        heldout, heldout_changes = corrected_label_overlay(heldout)
        label_changes.extend(heldout_changes)
    train_readers = {r["reciter_key"] for r in splits["train"]}
    if args.holdout_reciter in {r["reciter_key"] for r in rows}:
        raise ValueError("Held-out reader leaked into adaptation")
    known_hashes = {sha256(r["path"]) for r in rows}
    if any(sha256(r["path"]) in known_hashes for r in heldout):
        raise ValueError("Held-out voice includes duplicate adaptation audio")
    for name in splits:
        splits[name] = balanced_subset(splits[name], args.max_samples_per_split, args.seed)
    heldout = balanced_subset(heldout, args.max_samples_per_split, args.seed)
    training_texts = {row["text_asr_normalized"] for row in splits["train"]}
    for partition in [splits["validation"], splits["test"], heldout]:
        for row in partition:
            row["text_seen_in_training"] = row["text_asr_normalized"] in training_texts
    counts = {name: len(partition) for name, partition in splits.items()}
    print("Recording/surah-disjoint counts:", counts, "Unheard adaptation reciter:", args.holdout_reciter, len(heldout), flush=True)
    print("Training readers:", sorted(train_readers), "Excluded >30s:", skipped + heldout_skipped, flush=True)
    print("Canonical normalization overlay:", normalizer_version, len(label_changes), "changed labels; originals preserved", flush=True)
    if args.dry_run:
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Choose a NEW empty run directory; no resume/replacement inferred")
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import WeightedRandomSampler
    from transformers import (WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainer,
                              Seq2SeqTrainingArguments, EarlyStoppingCallback, TrainerCallback, set_seed)
    from train_qaloon_lora import WhisperCollator
    from gpu_evaluation import evaluate_rows
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this run")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 requested but unsupported on this CUDA GPU")
    use_bf16 = args.precision == "bf16" or args.precision == "auto" and torch.cuda.is_bf16_supported()
    # Seed before LoRA's random initialization, not just Trainer construction.
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True)
    frozen_splits = {**splits, "unheard_adaptation_reciter": heldout}
    manifest = {"initialization": args.init_model, "revision": MODEL_REVISIONS[args.init_model],
        "adaptation": "LoRA-not-full-finetuning", "private_experiment_only": True, "production_approved": False,
        "heldout_reader": args.holdout_reciter, "tarteel_upstream_reader_overlap": "unknown",
        "protocol": "surah-recording-and-canonical-ayah-disjoint-selection; separate speaker-heldout-test",
        "training_readers": sorted(train_readers), "seed": args.seed, "is_smoke_run": args.max_steps > 0 or args.max_samples_per_split is not None,
        "adapter_initialization_seeded": True,
        "label_field": args.label_field, "labels_never_from_asr_predictions": True,
        "normalizer_version": normalizer_version, "label_overlay_changes": label_changes, "training_quarantine": quarantine,
        "actual_gpu": {"name": torch.cuda.get_device_name(), "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9},
        "splits": {name: [{**{k: r.get(k) for k in ("surah", "ayah", "reciter_key", "path", "source_sha256", "text_asr_normalized", "original_text_asr_normalized", "text_raw_uthmani", "normalized_with_harakat", "normalization_reaudit_required", "experiment_decision_sha256", "validation_report_sha256", "text_seen_in_training")},
                         "source_use_basis": r.get("source_use_basis"), "rights_status": r.get("rights_status"),
                         "audio_sha256": sha256(r["path"])} for r in partition] for name, partition in frozen_splits.items()},
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    manifest_path = args.output_dir / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    processor = WhisperProcessor.from_pretrained(args.init_model, revision=MODEL_REVISIONS[args.init_model], language="arabic", task="transcribe", trust_remote_code=False)
    processor.tokenizer.set_prefix_tokens(language="arabic", task="transcribe")
    base = WhisperForConditionalGeneration.from_pretrained(args.init_model, revision=MODEL_REVISIONS[args.init_model], trust_remote_code=False)
    configure_generation(base, processor)
    for key, value in DECODE_PROFILES[args.decode_profile].items():
        setattr(base.generation_config, key, value)
    # Unadapted control is development-only and uses exactly the selected decoder.
    base.to("cuda")
    control_metrics, control_details = evaluate_rows(base, processor, splits["validation"], args.eval_batch_size, decode_profile=args.decode_profile, normalizer_version=normalizer_version)
    (args.output_dir / "unadapted_development_control.json").write_text(json.dumps(control_metrics, indent=2), encoding="utf-8")
    (args.output_dir / "unadapted_development_predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in control_details), encoding="utf-8")
    print("Unadapted DEVELOPMENT control:", control_metrics["overall"], control_metrics["decoding_safety"], flush=True)
    base.config.use_cache = False
    model = get_peft_model(base, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=.1, revision=MODEL_REVISIONS[args.init_model],
                                          target_modules=args.target_modules, bias="none"))
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    reader_counts = {reader: sum(r["reciter_key"] == reader for r in splits["train"]) for reader in train_readers}
    weights = [(args.supplement_weight if r.get("experimental_admission") else 1.0) / reader_counts[r["reciter_key"]] for r in splits["train"]]
    class BalancedTrainer(Seq2SeqTrainer):
        def _get_train_sampler(self, *unused, **unused_kw):
            return WeightedRandomSampler(weights, len(weights), replacement=True)
    normalizer = normalizer_for_version(normalizer_version)
    def compute_metrics(prediction):
        from src.training.asr_metrics import scores
        ids = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
        labels = np.where(prediction.label_ids == -100, processor.tokenizer.pad_token_id, prediction.label_ids)
        refs = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
        hyps = processor.tokenizer.batch_decode(ids, skip_special_tokens=True)
        if len(refs) != len(splits["validation"]) or len(hyps) != len(refs):
            raise ValueError("Generated validation predictions do not match frozen evaluation rows")
        # Always compare against the unchanged canonical ASR reference, not a
        # second normalization pass over the differently serialized vowel target.
        refs = [row["text_asr_normalized"] for row in splits["validation"]]
        result = scores([{"reference": normalizer(r), "prediction": normalizer(h)} for r, h in zip(refs, hyps)])
        output = {k: result[k] for k in ("wer", "cer", "exact_ayah_accuracy", "deletion_rate", "insertion_rate")}
        decoded = [{"reference": normalizer(r), "prediction": normalizer(h), "reciter": row["reciter_key"]}
                   for r, h, row in zip(refs, hyps, splits["validation"])]
        if len(decoded) != len(splits["validation"]):
            raise ValueError("Generated validation predictions do not match frozen evaluation rows")
        output["macro_reciter_wer"] = float(np.mean([scores([r for r in decoded if r["reciter"] == reader])["wer"]
            for reader in sorted({r["reciter"] for r in decoded})]))
        return output
    history, probe = [], balanced_subset(splits["train"], 48, args.seed)
    class GeneralizationMonitor(TrainerCallback):
        def on_evaluate(self, args_, state, control, metrics=None, **kwargs):
            clean_scores, _ = evaluate_rows(model, processor, probe, args.eval_batch_size, decode_profile=args.decode_profile, normalizer_version=normalizer_version)
            row = {"step": state.global_step, "epoch": state.epoch, "clean_train_probe_wer": clean_scores["overall"]["wer"],
                "validation_wer": metrics.get("eval_wer"), "validation_macro_reciter_wer": metrics.get("eval_macro_reciter_wer"),
                "validation_loss": metrics.get("eval_loss"), "train_probe_decoding_safety": clean_scores["decoding_safety"],
                "note": "Fixed clean train probe, not augmented train loss; test/heldout never used for selection."}
            row["validation_minus_clean_train_wer"] = row["validation_wer"] - row["clean_train_probe_wer"]
            history.append(row)
            (args.output_dir / "generalization_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            model.train()
        def on_save(self, args_, state, control, **kwargs):
            save_decoding_bundle(model, processor, args.output_dir / f"checkpoint-{state.global_step}", args.decode_profile)
    training_args = Seq2SeqTrainingArguments(output_dir=str(args.output_dir), num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation, learning_rate=args.learning_rate,
        lr_scheduler_type="cosine", warmup_ratio=.1, weight_decay=.01, max_grad_norm=1.,
        bf16=use_bf16, fp16=not use_bf16,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps" if args.eval_steps else "epoch", save_strategy="steps" if args.eval_steps else "epoch",
        eval_steps=args.eval_steps or None, save_steps=args.eval_steps or 500,
        save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="macro_reciter_wer", greater_is_better=False, predict_with_generate=True,
        generation_max_length=448, generation_num_beams=DECODE_PROFILES[args.decode_profile]["num_beams"], remove_unused_columns=False, label_names=["labels"],
        logging_steps=10, dataloader_num_workers=0, report_to="none", seed=args.seed, data_seed=args.seed)
    trainer = BalancedTrainer(model=model, args=training_args,
        train_dataset=AugmentedAyahDataset(splits["train"], processor, args.label_field, args.noise_prob, args.speed_prob,
            (args.tempo_min, args.tempo_max) if args.tempo_min is not None else None),
        eval_dataset=AugmentedAyahDataset(splits["validation"], processor, args.label_field),
        data_collator=WhisperCollator(processor, model.config.decoder_start_token_id), compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience), GeneralizationMonitor()])
    trainer.train()
    model.config.use_cache = True
    model.save_pretrained(args.output_dir / "adapter")
    processor.save_pretrained(args.output_dir / "adapter")
    save_decoding_bundle(model, processor, args.output_dir / "adapter", args.decode_profile)
    results = {"manifest_sha256": sha256(manifest_path), "best_validation_macro_reciter_wer": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()), "heldout_reader": args.holdout_reciter,
        "upstream_exposure_unknown": True, "is_smoke_run": manifest["is_smoke_run"], "generalization_history": history,
        "unadapted_development_control": control_metrics, "decode_profile": args.decode_profile,
        "production_approved": False, "human_learner_validation": "not-established",
        "last_optimizer_step": trainer.state.global_step, "last_epoch": trainer.state.epoch,
        "evaluation_status": "in-progress"}
    # Save selection evidence BEFORE expensive final inference; retain results
    # even if a native/process failure bypasses Python exception reporting.
    (args.output_dir / "training_log.json").write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Training finished; selected", results["best_checkpoint"], "at step", results["last_optimizer_step"], flush=True)
    # Train/eval PCM caches and optimizer states aren't required for inference.
    # This reduces host/GPU pressure without changing weights or decoder settings.
    import gc
    trainer.train_dataset.cached_samples.clear()
    trainer.eval_dataset.cached_samples.clear()
    del trainer
    for parameter in model.parameters():
        parameter.grad = None
    gc.collect(); torch.cuda.empty_cache()
    reload_reference = None
    evaluation_partitions = [("validation", splits["validation"]), ("test", splits["test"])]
    if not args.skip_heldout_eval:
        evaluation_partitions.append(("unheard_adaptation_reciter", heldout))
    for name, partition in evaluation_partitions:
        print("Final evaluation:", name, len(partition), "clips", flush=True)
        metrics, details = evaluate_rows(model, processor, partition, args.eval_batch_size, decode_profile=args.decode_profile,
            predictions_path=args.output_dir / f"{name}_predictions.jsonl", normalizer_version=normalizer_version)
        if name == "validation":
            reload_reference = details[:8]
        results[name] = metrics
        (args.output_dir / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    results["peak_cuda_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
    results["development_candidate_minus_unadapted_macro_wer"] = results["validation"]["diagnostics"]["macro_reciter_wer"] - control_metrics["diagnostics"]["macro_reciter_wer"]
    results["development_candidate_minus_unadapted_deletion_rate"] = results["validation"]["diagnostics"]["overall"]["deletion_rate"] - control_metrics["diagnostics"]["overall"]["deletion_rate"]
    del model, base
    gc.collect(); torch.cuda.empty_cache()
    reloaded, reloaded_processor = load_private_adapter(args.output_dir / "adapter")
    _, replay = evaluate_rows(reloaded, reloaded_processor, splits["validation"][:8], args.eval_batch_size, decode_profile=args.decode_profile, normalizer_version=normalizer_version)
    matched = all(a["prediction"] == b["prediction"] and a["audio_sha256"] == b["audio_sha256"] for a, b in zip(reload_reference, replay)) and len(replay) == len(reload_reference)
    results["saved_adapter_reload"] = {"development_samples": len(replay), "same_predictions_and_audio": matched, "pinned_base_and_saved_generation_config": True}
    del reloaded
    gc.collect(); torch.cuda.empty_cache()
    results["readiness"] = {"saved_reload_matches": matched,
        "development_macro_wer_not_worse_than_unadapted": results["development_candidate_minus_unadapted_macro_wer"] <= 0,
        "development_cer_not_worse_than_unadapted": results["validation"]["overall"]["cer"] <= control_metrics["overall"]["cer"],
        "development_deletions_not_worse_than_unadapted": results["development_candidate_minus_unadapted_deletion_rate"] <= 0,
        "no_flagged_development_decodes": results["validation"]["decoding_safety"]["flagged_samples"] == 0,
        "qualified_unfamiliar_learner_error_validation": False,
        "fresh_confirmatory_test": False,
        "public_release_source_clearance": False,
        "human_ready_or_deployment_approved": False,
        "warning": "Development/regression scores alone cannot certify pronunciation, tajweed or reliable learner grading."}
    results["evaluation_status"] = "complete" if matched else "reload-mismatch"
    (args.output_dir / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    if not matched:
        raise ValueError("Saved adapter reload changed fixed development decoding; refuse readiness claim")
    if args.baseline_model_dir:
        from compare_asr_runs import compare_existing
        compare_existing(args.output_dir, args.baseline_model_dir, args.eval_batch_size)
    print(f"Completed private LoRA run: {args.output_dir}; no deployment/upload. Results are protocol-specific.", flush=True)


if __name__ == "__main__":
    main()
