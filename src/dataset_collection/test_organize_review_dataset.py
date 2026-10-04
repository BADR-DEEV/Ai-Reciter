import json
from pathlib import Path
import tempfile
import unittest
from .organize_review_dataset import digest, organize


class OrganizeDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.destination = self.root / "destination"
        for folder in ("audio", "audio_raw"):
            (self.source / folder).mkdir(parents=True)
            (self.source / folder / "clip.wav").write_bytes(b"synthetic-test-payload")
        row = {"relative_audio_path": "audio/clip.wav", "audio_sha256": digest(self.source / "audio/clip.wav"),
               "raw_audio_path": "audio_raw/clip.wav", "raw_audio_sha256": digest(self.source / "audio_raw/clip.wav"),
               "usable_for_training": False, "listening_approved": False}
        (self.source / "metadata.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    def test_hash_preserving_copy_and_audit_relocation_are_not_approval(self):
        audit = self.root / "audit.json"
        audit.write_text(json.dumps({"dataset_root": str(self.source), "metadata_sha256": digest(self.source / "metadata.jsonl")}), encoding="utf-8")
        self.assertEqual(organize(self.source, self.destination, audit), 1)
        self.assertEqual(digest(self.source / "metadata.jsonl"), digest(self.destination / "metadata.jsonl"))
        self.assertTrue((self.source / "audio/clip.wav").exists())
        moved = json.loads((self.destination / "validation.json").read_text())
        self.assertEqual(moved["dataset_root"], str(self.destination.resolve()))
        self.assertFalse(moved["relocation"]["new_listening_approval"])
        self.assertFalse(json.loads((self.destination / "metadata.jsonl").read_text())["usable_for_training"])

    def test_existing_destination_requires_explicit_nonexisting_archive(self):
        self.destination.mkdir()
        original = self.destination / "user-work.txt"
        original.write_text("preserve this")
        with self.assertRaisesRegex(ValueError, "Destination exists"):
            organize(self.source, self.destination)
        self.assertEqual(original.read_text(), "preserve this")
        archive = self.root / "archive"
        organize(self.source, self.destination, archive_existing=archive)
        self.assertEqual((archive / "user-work.txt").read_text(), "preserve this")

    def test_changed_source_and_nested_destinations_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-nested"):
            organize(self.source, self.source / "copy")
        (self.source / "audio/clip.wav").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "path/hash"):
            organize(self.source, self.destination)
        self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main()
