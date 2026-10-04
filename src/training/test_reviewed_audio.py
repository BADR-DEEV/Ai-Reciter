import json
from pathlib import Path
import tempfile
import unittest
import wave

from .reviewed_audio import load_reviewed_trabulsi, sha256


class ReviewedAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "audio").mkdir()
        self.audio = self.root / "audio/093001.wav"
        self.source = self.root / "source.wav"
        for file in (self.audio, self.source):
            with wave.open(str(file), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"\x12\x01" * 16000)
        self.row = {"reciter_key": "trabulsi", "surah": 93, "ayah": 1,
                    "usable_for_training": True, "alignment_status": "approved", "reviewer": "test-reviewer",
                    "reviewed_at": "test-date", "listening_approved": True, "boundary_approved": True,
                    "authorization_status": "authorized", "source_file": str(self.source),
                    "source_sha256": sha256(self.source), "audio_sha256": sha256(self.audio),
                    "relative_audio_path": "audio/093001.wav", "sample_rate": 16000,
                    "start_frame": 0, "end_frame": 16000, "text_asr_normalized": "والضحى"}
        # Synthetic test receipt, never a real authorization/example grant.
        self.permission = {"reciter_key": "trabulsi", "training_allowed": True, "grant_reference": "synthetic-test-only",
            "rights_holder": "test-only", "reviewed_by": "test-only", "reviewed_at": "test-date",
            "supersedes_unauthorized_source_restriction": True, "authorized_source_sha256": [self.row["source_sha256"]]}
        self.manifest = self.root / "reviewed.jsonl"
        self.receipt = self.root / "permission.json"
        self.audit = self.root / "validation.json"

    def load(self, rows=None):
        self.manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in (rows or [self.row])), encoding="utf-8")
        self.receipt.write_text(json.dumps(self.permission), encoding="utf-8")
        original = self.root / "metadata.jsonl"
        original.write_text(self.manifest.read_text(encoding="utf-8"), encoding="utf-8")
        report = {"complete_audit": True, "strict_content_comparison": True, "technical_only": False,
                  "dataset_root": str(self.root), "metadata_sha256": sha256(original),
                  "clips": [{"surah": row["surah"], "ayah": row["ayah"], "audio_sha256": row["audio_sha256"],
                             "automatic_pass": True, "technical": {"passed": True}, "content": {"passed": True}, "issues": []} for row in (rows or [self.row])]}
        self.audit.write_text(json.dumps(report), encoding="utf-8")
        return load_reviewed_trabulsi(self.manifest, self.receipt, canonical={(93, 1): "والضحى", (93, 2): "والليل"}, validation_report=self.audit)

    def test_strict_validation_is_required_even_with_approval(self):
        self.load()
        with self.assertRaisesRegex(ValueError, "validation report"):
            load_reviewed_trabulsi(self.manifest, self.receipt)
        report = json.loads(self.audit.read_text(encoding="utf-8"))
        report["clips"][0]["automatic_pass"] = False
        self.audit.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "did not pass"):
            load_reviewed_trabulsi(self.manifest, self.receipt, canonical={(93, 1): "والضحى"}, validation_report=self.audit)

    def test_mp3quran_direct_does_not_inherit_hub_restriction(self):
        inventory = {"source_page": "https://www.mp3quran.net/eng/trabulsi-qalon", "policy_url": "https://www.mp3quran.net/eng/privacy",
                     "public_material_reuse_clause_found": True, "sources": [{"source_sha256": self.row["source_sha256"],
                        "publisher_sha256": self.row["source_sha256"], "byte_identical": True, "source_provider": "mp3quran-direct",
                        "riwayah": "Qaloon", "source_url": "https://cdn.mp3quran.net/audio/ahmad-tarabulsi/r1/093.mp3"}]}
        evidence = self.root / "source_inventory.json"
        evidence.write_text(json.dumps(inventory), encoding="utf-8")
        self.permission.update({"source_provider": "mp3quran-direct", "source_inventory": str(evidence), "source_inventory_sha256": sha256(evidence)})
        self.permission.pop("supersedes_unauthorized_source_restriction")
        self.row.update({"source_provider": "mp3quran-direct", "source_url": inventory["sources"][0]["source_url"]})
        self.assertEqual(sum(len(rows) for rows in self.load().values()), 1)
        self.row["source_url"] = "https://example.org/unrelated.mp3"
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            self.load()

    def test_both_permission_and_listening_approval_are_required(self):
        self.permission["training_allowed"] = False
        with self.assertRaisesRegex(ValueError, "rights-holder"):
            self.load()
        self.permission["training_allowed"] = True
        self.row["boundary_approved"] = False
        with self.assertRaisesRegex(ValueError, "listening/boundary"):
            self.load()

    def test_machine_proposal_does_not_become_training_data(self):
        self.row["alignment_status"] = "auto-checked-needs-listening-review"
        self.row["usable_for_training"] = False
        with self.assertRaises(ValueError):
            self.load()

    def test_receipt_must_cover_actual_immutable_recording(self):
        self.permission["authorized_source_sha256"] = ["f" * 64]
        with self.assertRaisesRegex(ValueError, "not covered"):
            self.load()
        self.permission["authorized_source_sha256"] = [self.row["source_sha256"]]
        self.source.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "source hash"):
            self.load()

    def test_mismatched_label_or_audio_is_rejected(self):
        self.row["text_asr_normalized"] = "different"
        with self.assertRaisesRegex(ValueError, "canonical"):
            self.load()
        self.row["text_asr_normalized"] = "والضحى"
        self.row["audio_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "audio hash"):
            self.load()

    def test_path_traversal_is_rejected(self):
        self.row["relative_audio_path"] = "../outside.wav"
        with self.assertRaisesRegex(ValueError, "path"):
            self.load()

    def test_frame_mismatch_is_rejected(self):
        self.row["end_frame"] += 1
        with self.assertRaisesRegex(ValueError, "frame boundaries"):
            self.load()

    def test_approved_cuts_of_one_recording_stay_in_one_partition(self):
        second = self.root / "audio/093002.wav"
        with wave.open(str(second), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\x13\x01" * 16000)
        row = {**self.row, "ayah": 2, "text_asr_normalized": "والليل", "relative_audio_path": "audio/093002.wav", "audio_sha256": sha256(second)}
        splits = self.load([self.row, row])
        self.assertEqual(sum(bool(rows) for rows in splits.values()), 1)
        self.assertEqual(sum(len(rows) for rows in splits.values()), 2)
        self.assertTrue(all(r["permission_receipt_sha256"] for rows in splits.values() for r in rows))

    def test_duplicate_audio_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.load([self.row, self.row])


if __name__ == "__main__":
    unittest.main()
