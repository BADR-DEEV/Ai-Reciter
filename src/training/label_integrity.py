"""Versioned canonical label overlays and hash-bound TRAINING-only quarantine.

Original datasets/audits/checkpoints are never rewritten. The overlay repairs
serialization, not incomplete audio or pronunciation, and is not teacher approval.
"""
import json
from pathlib import Path
from .reviewed_audio import sha256
from ..dataset_collection.qaloon_audio2text import (
    load_quran, normalize_quran_for_asr, normalize_quran_for_asr_v1, NORMALIZER_VERSION)


def corrected_label_overlay(rows):
    canonical = load_quran()
    corrected, changes = [], []
    for original in rows:
        row = dict(original)
        # Waleed Fatiha has different audio numbering/merged final segments.
        # Use each recording's actual source text, not a guessed display index.
        raw = row.get("text_raw_uthmani") or canonical[(row["surah"], row["ayah"])]["raw"]
        old, new = row["text_asr_normalized"], normalize_quran_for_asr(raw)
        if old not in {normalize_quran_for_asr_v1(raw), new}:
            raise ValueError(f"Stored label disagrees with its original source: {row['reciter_key']} {row['surah']}:{row['ayah']}")
        row.update(text_asr_normalized=new, original_text_asr_normalized=old,
                   normalizer_version=NORMALIZER_VERSION, text_raw_uthmani=raw)
        if old != new:
            changes.append({"reciter": row["reciter_key"], "surah": row["surah"], "ayah": row["ayah"],
                            "old": old, "new": new, "audio_sha256": sha256(row["path"])})
            row["normalization_reaudit_required"] = bool(row.get("experimental_admission"))
        corrected.append(row)
    return corrected, changes


def apply_training_quarantine(splits, manifest_path):
    """Only fitting rows may be excluded; validation/test/heldout stay intact."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if (manifest.get("scope") != "training-only-boundary-quarantine" or
            manifest.get("normalizer_version") != NORMALIZER_VERSION or not manifest.get("basis")):
        raise ValueError("Versioned, explicitly based training-only quarantine required")
    quarantined = {}
    if "training_inventory" in manifest:
        inventory = {(r["reciter"], r["surah"], r["ayah"]): (r["audio_sha256"], r["reference"])
                     for r in manifest["training_inventory"]}
        actual = {(r["reciter_key"], r["surah"], r["ayah"]): (sha256(r["path"]), r["text_asr_normalized"])
                  for r in splits["train"]}
        if inventory != actual or len(inventory) != len(manifest["training_inventory"]):
            raise ValueError("Complete fitting audit inventory differs from actual hashes/references")
    for row in manifest["clips"]:
        identity = (row["reciter"], int(row["surah"]), int(row["ayah"]))
        if identity in quarantined or not row.get("audio_sha256") or not row.get("reason"):
            raise ValueError("Quarantine identities must be unique, hash-bound and reasoned")
        quarantined[identity] = row
    matched, kept, excluded = set(), [], []
    for row in splits["train"]:
        identity = (row["reciter_key"], row["surah"], row["ayah"])
        if identity not in quarantined:
            kept.append(row)
            continue
        decision = quarantined[identity]
        if decision["audio_sha256"] != sha256(row["path"]):
            raise ValueError("Quarantined PCM changed")
        matched.add(identity)
        excluded.append(decision)
    if matched != quarantined.keys() or not kept:
        raise ValueError("Every quarantine identity must belong to fitting, never validation/test/heldout")
    return {**splits, "train": kept}, {"manifest_sha256": sha256(manifest_path), "excluded": excluded,
        "validation_test_and_heldout_unchanged": True, "not_teacher_approval": True}
