import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .pull_hf_assets import link_directory


class LinkDirectoryTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.target = Path(folder.name) / "data/dataset_qaloon_dokali"
        self.target.mkdir(parents=True)
        (self.target / "metadata.jsonl").write_text("{}\n", encoding="utf-8")
        self.link = Path(folder.name) / "src/dataset_qaloon_dokali"
        self.link.parent.mkdir()

    def test_link_reads_through(self):
        link_directory(self.link, self.target)
        self.assertEqual((self.link / "metadata.jsonl").read_text(encoding="utf-8"), "{}\n")

    def test_without_symlink_rights(self):
        # Windows without Developer Mode cannot create symlinks; a junction needs no rights.
        with patch.object(Path, "symlink_to", side_effect=OSError("A required privilege is not held by the client")):
            if os.name != "nt":
                with self.assertRaises(OSError):
                    link_directory(self.link, self.target)
                return
            link_directory(self.link, self.target)
        self.assertEqual((self.link / "metadata.jsonl").read_text(encoding="utf-8"), "{}\n")


if __name__ == "__main__":
    unittest.main()
