import json
from pathlib import Path
import tempfile
import unittest
from .label_integrity import corrected_label_overlay, apply_training_quarantine
from .reviewed_audio import sha256
from ..dataset_collection.qaloon_audio2text import NORMALIZER_VERSION


class LabelIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "synthetic.wav"
        self.path.write_bytes(b"synthetic-hash-fixture-not-training-audio")
        self.row = {"reciter_key": "dokali", "surah": 75, "ayah": 1, "path": str(self.path),
            "text_raw_uthmani": "ٱلْقِيَٰمَةِ", "text_asr_normalized": "القامة"}

    def test_overlay_changes_label_without_rewriting_original(self):
        corrected, changes = corrected_label_overlay([self.row])
        self.assertEqual(self.row["text_asr_normalized"], "القامة")
        self.assertEqual(corrected[0]["text_asr_normalized"], "القيامة")
        self.assertEqual(changes[0]["old"], "القامة")
        self.assertEqual(changes[0]["audio_sha256"], sha256(self.path))

    def test_waleed_fatiha_audio_number_is_not_mistaken_for_canonical_number(self):
        row = {**self.row, "reciter_key": "waleed", "surah": 1, "ayah": 4,
            "text_raw_uthmani": "مَلِكِ يَوْمِ الدِّينِ", "text_asr_normalized": "ملك يوم الدين"}
        corrected, _ = corrected_label_overlay([row])
        self.assertEqual(corrected[0]["text_asr_normalized"], "ملك يوم الدين")

    def test_training_quarantine_never_filters_holdouts_or_accepts_wrong_hash(self):
        path = Path(self.directory.name) / "quarantine.json"
        row2 = {**self.row, "ayah": 2}
        splits = {"train": [self.row, row2], "validation": [row2], "test": [row2]}
        decision = {"scope": "training-only-boundary-quarantine", "normalizer_version": NORMALIZER_VERSION,
            "basis": "synthetic-test", "clips": [{"reciter": "dokali", "surah": 75, "ayah": 1,
                "audio_sha256": sha256(self.path), "reason": "synthetic-test"}]}
        path.write_text(json.dumps(decision), encoding="utf-8")
        result, _ = apply_training_quarantine(splits, path)
        self.assertEqual(result["train"], [row2])
        self.assertIs(result["validation"], splits["validation"])
        self.assertIs(result["test"], splits["test"])
        decision["clips"][0]["audio_sha256"] = "wrong"
        path.write_text(json.dumps(decision), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "PCM"):
            apply_training_quarantine(splits, path)

    def test_wrong_source_cannot_be_silently_relabelled(self):
        with self.assertRaisesRegex(ValueError, "source"):
            corrected_label_overlay([{**self.row, "text_asr_normalized": "شيء آخر"}])
