"""Synthetic admission fixtures, not actual recording grants or approvals."""
import json
from pathlib import Path
import tempfile
import unittest
import wave
from .experiment_data import experimental_candidates
from .reviewed_audio import sha256


class ExperimentalAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "audio").mkdir()
        self.audio = self.root / "audio/093001.wav"
        self.source = self.root / "source.wav"
        for path in (self.audio, self.source):
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"\x12\x01" * 16000)
        self.row = {"reciter_key": "trabulsi", "surah": 93, "ayah": 1,
            "source_file": str(self.source), "source_sha256": sha256(self.source), "audio_sha256": sha256(self.audio),
            "relative_audio_path": "audio/093001.wav", "sample_rate": 16000, "start_frame": 0, "end_frame": 16000,
            "text_asr_normalized": "والضحى", "usable_for_training": False, "listening_approved": False,
            "source_url": "https://cdn.mp3quran.net/audio/ahmad-tarabulsi/r1/093.mp3"}
        self.inventory = self.root / "inventory.json"
        self.inventory.write_text(json.dumps({"source_page": "https://www.mp3quran.net/eng/trabulsi-qalon",
            "policy_url": "https://www.mp3quran.net/eng/privacy", "public_material_reuse_clause_found": True,
            "sources": [{"source_sha256": self.row["source_sha256"], "publisher_sha256": self.row["source_sha256"],
                "source_provider": "mp3quran-direct", "source_url": self.row["source_url"], "riwayah": "Qaloon", "byte_identical": True}]}), encoding="utf-8")
        self.decision = {"private_experiment_only": True, "user_quality_acceptance": "synthetic unit fixture, not human review",
            "reciter_key": "trabulsi", "authorized_source_sha256": [self.row["source_sha256"]],
            "source_use_basis": "publisher-public-material-reuse-policy", "source_inventory": str(self.inventory),
            "source_inventory_sha256": sha256(self.inventory)}
        self.decision_path = self.root / "decision.json"

    def load(self, automatic_pass=True):
        metadata = self.root / "metadata.jsonl"
        metadata.write_text(json.dumps(self.row, ensure_ascii=False) + "\n", encoding="utf-8")
        (self.root / "validation.json").write_text(json.dumps({"dataset_root": str(self.root), "metadata_sha256": sha256(metadata),
            "complete_audit": True, "strict_content_comparison": True, "technical_only": False,
            "clips": [{"surah": 93, "ayah": 1, "audio_sha256": self.row["audio_sha256"], "automatic_pass": automatic_pass,
                "technical": {"passed": True}, "content": {"passed": True}, "issues": []}]}), encoding="utf-8")
        self.decision_path.write_text(json.dumps(self.decision), encoding="utf-8")
        return experimental_candidates(self.root, self.decision_path)

    def test_explicit_experimental_acceptance_does_not_manufacture_approval(self):
        rows = self.load()
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["usable_for_training"])
        self.assertFalse(rows[0]["listening_approved"])
        self.assertEqual(rows[0]["experimental_admission"], "user-accepted-machine-QA-not-certified")
        self.assertEqual(rows[0]["duration_seconds"], 1)

    def test_failed_candidate_never_enters_even_with_user_acceptance(self):
        with self.assertRaisesRegex(ValueError, "No exact passing"):
            self.load(automatic_pass=False)

    def test_source_and_inventory_changes_are_rejected(self):
        self.source.write_bytes(b"changed recording")
        with self.assertRaisesRegex(ValueError, "recording hash"):
            self.load()
        self.inventory.write_text("{}")
        with self.assertRaisesRegex(ValueError, "inventory hash"):
            self.load()

    def test_label_and_direct_source_identity_cannot_be_substituted(self):
        self.row["text_asr_normalized"] = "incorrect label"
        with self.assertRaisesRegex(ValueError, "canonical"):
            self.load()
        self.row["text_asr_normalized"] = "والضحى"
        self.row["source_url"] = "https://example.org/unrelated.mp3"
        with self.assertRaisesRegex(ValueError, "URL differs"):
            self.load()

    def test_taha_cannot_inherit_trabulsi_public_policy_basis(self):
        self.decision["reciter_key"] = "taha"
        with self.assertRaisesRegex(ValueError, "not Assabile/Taha"):
            self.load()

    def test_user_directed_research_keeps_uncertainty_and_cannot_enable_release(self):
        self.decision.update(source_use_basis="user-directed-private-research", reciter_key="taha",
            user_source_use_instruction="synthetic explicit private-use instruction", source_page="https://example.test/recordings",
            rights_status="unverified-user-asserted-open-access", production_release_allowed=False, explicit_ml_redistribution_grant=False)
        self.row["reciter_key"] = "taha"
        rows = self.load()
        self.assertEqual(rows[0]["rights_status"], "unverified-user-asserted-open-access")
        self.assertFalse(rows[0]["listening_approved"])
        self.decision["production_release_allowed"] = True
        with self.assertRaisesRegex(ValueError, "prohibit release"):
            self.load()


if __name__ == "__main__":
    unittest.main()
