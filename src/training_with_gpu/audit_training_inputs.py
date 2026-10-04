"""Blind DeepDML fitting-audio audit; cannot inspect or filter Waleed/test clips.

Conservative machine quarantine, NOT a listening decision: boundary insertions/
deletions, unstable decoding and changed supplemental labels needing a fresh gate.
Riwayah substitutions alone never justify dropping difficult Qaloon examples.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src/training"))
from src.training.qaloon_data import load_splits
from src.training.experiment_data import experimental_candidates, recording_safe_splits
from src.training.label_integrity import corrected_label_overlay
from src.training.reviewed_audio import sha256
from src.dataset_collection.qaloon_audio2text import NORMALIZER_VERSION
from src.training_with_gpu.train_base_full import DEEPDML_BASE, MODEL_REVISIONS, configure_generation
from src.training_with_gpu.gpu_evaluation import evaluate_rows


def boundary_reasons(detail, requires_reaudit=False):
    import jiwer
    reasons = []
    reference, hypothesis = detail["reference"], detail["prediction"]
    if not detail["scorable"]:
        reasons.append("unreliable-blind-decoding-listening-needed")
    result = jiwer.process_words(reference, hypothesis)
    for chunk in result.alignments[0]:
        if chunk.type == "delete" and (chunk.ref_start_idx == 0 or chunk.ref_end_idx == len(reference.split())):
            reasons.append("apparent-missing-boundary-word-listening-needed")
        elif chunk.type == "insert" and (chunk.ref_start_idx == 0 or chunk.ref_start_idx == len(reference.split())):
            reasons.append("apparent-extra-boundary-word-listening-needed")
    if requires_reaudit and reference != hypothesis:
        reasons.append("changed-supplemental-normalization-needs-new-exact-content-gate")
    return sorted(set(reasons))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--batch-size", type=int, default=2)
    args = p.parse_args()
    if args.output_dir.exists() or args.batch_size < 1:
        raise ValueError("New audit directory and positive batch size required")
    core, _ = load_splits(include=["huthaify", "dokali", "husary"])
    rows = [row for partition in core.values() for row in partition]
    for reader in ("trabulsi", "taha"):
        folder = ROOT / "src/dataset_collection" / f"dataset_qaloon_{reader}"
        rows.extend(experimental_candidates(folder, folder / "experiment_decision.json"))
    rows, changes = corrected_label_overlay(rows)
    training = recording_safe_splits(rows)["train"]
    if any(row["reciter_key"] == "waleed" for row in training):
        raise ValueError("Holdout entered audit")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "label_overlay_changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    processor = WhisperProcessor.from_pretrained(DEEPDML_BASE, revision=MODEL_REVISIONS[DEEPDML_BASE], language="arabic", task="transcribe", trust_remote_code=False)
    processor.tokenizer.set_prefix_tokens(language="arabic", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(DEEPDML_BASE, revision=MODEL_REVISIONS[DEEPDML_BASE], trust_remote_code=False).to("cuda").eval()
    configure_generation(model, processor)
    report, details = evaluate_rows(model, processor, training, args.batch_size, decode_profile="beam3",
        predictions_path=args.output_dir / "blind_training_predictions.jsonl")
    flagged = []
    for row, detail in zip(training, details):
        reasons = boundary_reasons(detail, row.get("normalization_reaudit_required", False))
        if reasons:
            flagged.append({"reciter": row["reciter_key"], "surah": row["surah"], "ayah": row["ayah"],
                "audio_sha256": detail["audio_sha256"], "reason": reasons,
                "reference": detail["reference"], "observed": detail["prediction"],
                "human_approval": False, "diagnosis": "machine-suspect-not-proven-corrupted-audio"})
    manifest = {"scope": "training-only-boundary-quarantine", "normalizer_version": NORMALIZER_VERSION,
        "basis": "conservative-blind-unadapted-ASR-boundary/unstable/normalization-reaudit-flags; no fabricated Claude list",
        "audit_model": DEEPDML_BASE, "audit_revision": MODEL_REVISIONS[DEEPDML_BASE], "expected_prompt": False,
        "training_inventory": [{"reciter": r["reciter_key"], "surah": r["surah"], "ayah": r["ayah"],
            "audio_sha256": d["audio_sha256"], "reference": d["reference"]} for r, d in zip(training, details)],
        "validation_test_waleed_not_audited_or_filtered": True, "clips": flagged,
        "counts": {r: sum(x["reciter"] == r for x in flagged) for r in sorted({x["reciter_key"] for x in training})}}
    (args.output_dir / "quarantine.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Fitting-only audit:", len(training), "clips; machine quarantine", manifest["counts"], flush=True)
    print("No Waleed, validation or test filtering; source labels/PCM unchanged.", flush=True)


if __name__ == "__main__":
    main()
