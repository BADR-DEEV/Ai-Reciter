from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .restore_local_full import restore


class RestoreTests(unittest.TestCase):
    def test_conflicting_file_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            target = destination / "model.safetensors"
            target.write_bytes(b"user-model")
            with patch("src.deployment.restore_local_full.hf_hub_download") as download:
                with self.assertRaisesRegex(ValueError, "Existing file differs"):
                    restore(destination)
                download.assert_not_called()
            self.assertEqual(target.read_bytes(), b"user-model")

    def test_checksum_mismatch_never_creates_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bad-download"
            source.write_bytes(b"not-published")
            with patch("src.deployment.restore_local_full.hf_hub_download", return_value=str(source)):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    restore(root / "model")
            self.assertEqual(list((root / "model").iterdir()), [])
