"""Validate local cache mappings and preservation of original metadata."""
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "web" / "public" / "quran"


@unittest.skipUnless((PUBLIC / "manifest.json").exists(), "Generate the local cache first")
class GeometryTests(unittest.TestCase):
    def test_all_surahs_and_unique_pages(self):
        manifest = json.loads((PUBLIC / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["surahs"]), 114)
        self.assertEqual(len(manifest["pages"]), 604)
        self.assertEqual(sum(row["trained"] for row in manifest["surahs"]), 38)
        for surah in manifest["surahs"]:
            data = json.loads((PUBLIC / "surahs" / f"{surah['id']:03d}.json").read_text(encoding="utf-8"))
            for ayah in data["ayahs"]:
                self.assertTrue(ayah["regions"], f"Unmapped {surah['id']}:{ayah['ayah']}")
                for region in ayah["regions"]:
                    self.assertTrue((ROOT / "web" / "public" / region["page"].lstrip("/")).exists())

    def test_fatiha_is_not_off_by_one(self):
        data = json.loads((PUBLIC / "surahs" / "001.json").read_text(encoding="utf-8"))
        self.assertEqual(data["ayahs"][0]["regions"][0]["display_ayah"], 2)
        self.assertEqual(data["ayahs"][5]["regions"][0]["display_ayah"], 7)
        self.assertEqual(data["ayahs"][6]["regions"][0]["display_ayah"], 7)
        self.assertFalse(data["preciseGeometry"])

    def test_original_labels_and_timestamps_are_preserved(self):
        paths = list((ROOT / "src" / "dataset_collection").glob("dataset_qaloon_*/metadata.jsonl"))
        paths += [ROOT / "walid_qaloon" / "metadata.jsonl"]
        for path in paths:
            backup = path.with_suffix(path.suffix + ".before-geometry.bak")
            if not backup.exists():
                continue
            before = [json.loads(line) for line in backup.read_text(encoding="utf-8").splitlines() if line.strip()]
            after = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(before), len(after))
            for original, enriched in zip(before, after):
                for key, value in original.items():
                    self.assertEqual(enriched[key], value, f"Changed {path.name}: {key}")
                if "waleed" in str(path) or "walid" in str(path):
                    self.assertEqual("start_time" in original, "start_time" in enriched)


if __name__ == "__main__":
    unittest.main()
