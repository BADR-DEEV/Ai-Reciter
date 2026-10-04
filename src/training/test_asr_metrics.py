import tempfile
import unittest
from pathlib import Path
from .asr_metrics import report_predictions, paired_comparison
from .experiment_data import recording_safe_splits, validate_partition_integrity
from .experiment_data import experimental_candidates
import json


class DiagnosticTests(unittest.TestCase):
    def rows(self):
        return [{"surah": 78, "ayah": 1, "reciter": "one", "reference": "قل هو الله", "prediction": "قل الله", "audio_sha256": "a", "duration_seconds": 4},
                {"surah": 112, "ayah": 1, "reciter": "two", "reference": "لم يلد", "prediction": "لم يلد زائد", "audio_sha256": "b", "duration_seconds": 16}]

    def test_micro_errors_macro_readers_and_exact_match_are_distinct(self):
        report = report_predictions(self.rows())
        self.assertEqual(report["overall"]["word_deletions"], 1)
        self.assertEqual(report["overall"]["word_insertions"], 1)
        self.assertEqual(report["overall"]["exact_ayah_accuracy"], 0)
        self.assertAlmostEqual(report["overall"]["wer"], 2 / 5)
        self.assertAlmostEqual(report["macro_reciter_wer"], (1 / 3 + 1 / 2) / 2)
        self.assertEqual(report["worst_reciter_wer"], .5)
        self.assertEqual(report["wer_interval"]["groups"], 2)
        self.assertIn("long_gt_15s", report)

    def test_paired_delta_requires_same_audio_and_labels(self):
        base = self.rows()
        candidate = [{**r, "prediction": r["reference"]} for r in base]
        self.assertAlmostEqual(paired_comparison(base, candidate)["candidate_minus_baseline_wer"], -.4)
        with self.assertRaises(ValueError):
            paired_comparison(base, [{**candidate[0], "audio_sha256": "different"}, candidate[1]])
        with self.assertRaises(ValueError):
            paired_comparison(base, candidate[:1])
        with self.assertRaises(ValueError):
            paired_comparison([{k: v for k, v in r.items() if k != "audio_sha256"} for r in base],
                              [{k: v for k, v in r.items() if k != "audio_sha256"} for r in candidate])

    def test_recording_groups_and_canonical_ids_never_cross_partitions(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = []
            for surah in range(78, 100):
                for ayah in (1, 2):
                    for reader in ("core", "trabulsi"):
                        path = Path(directory) / f"{surah}-{ayah}-{reader}"
                        path.write_text(path.name)
                        rows.append({"surah": surah, "ayah": ayah, "path": str(path), "source_sha256": f"{reader}-{surah}", "reciter_key": reader})
            split = recording_safe_splits(rows)
            self.assertEqual(split, recording_safe_splits(rows))
            self.assertTrue(validate_partition_integrity(split))
            split["test"].append(split["train"][0])
            with self.assertRaisesRegex(ValueError, "leakage"):
                validate_partition_integrity(split)

    def test_free_downloads_do_not_clear_taha_for_train_or_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            decision = Path(directory) / "decision.json"
            decision.write_text(json.dumps({"reciter_key": "taha", "private_experiment_only": True,
                "user_quality_acceptance": "Accept machine QA for this experiment", "authorized_source_sha256": ["a" * 64],
                "source_use_basis": "free-download"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Free downloads"):
                experimental_candidates(directory, decision)


if __name__ == "__main__":
    unittest.main()
