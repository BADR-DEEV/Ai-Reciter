import copy
import unittest
from .finalize_private_lora_run import verify_saved_predictions
from .gpu_evaluation import summarize_predictions


class EvaluationRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"reciter_key": "dokali", "surah": 112, "ayah": 1,
            "text_asr_normalized": "قل هو الله احد", "audio_sha256": "synthetic-hash", "text_seen_in_training": False}]
        self.details = [{"reciter": "dokali", "surah": 112, "ayah": 1,
            "reference": "قل هو الله احد", "prediction": "قل هو الله احد", "audio_sha256": "synthetic-hash",
            "text_seen_in_training": False, "duration_seconds": 2, "scorable": True, "eos_emitted": True, "decode_flags": []}]

    def test_only_complete_identical_frozen_predictions_are_reused(self):
        verify_saved_predictions(self.details, self.rows)
        with self.assertRaisesRegex(ValueError, "count"):
            verify_saved_predictions([], self.rows)
        for field, value in (("reciter", "waleed"), ("reference", "قل"), ("audio_sha256", "other"),
                             ("ayah", 2), ("text_seen_in_training", True)):
            changed = copy.deepcopy(self.details)
            changed[0][field] = value
            with self.assertRaisesRegex(ValueError, "differs"):
                verify_saved_predictions(changed, self.rows)

    def test_offline_summary_never_fabricates_inference_latency(self):
        report = summarize_predictions(self.details, "beam3")
        self.assertEqual(report["overall"]["wer"], 0)
        self.assertEqual(report["decoding_safety"]["coverage"], 1)
        self.assertNotIn("performance", report)

    def test_flagged_output_remains_in_official_error_denominator(self):
        self.details[0].update(prediction="د " * 40, scorable=False, eos_emitted=False,
            decode_flags=["sustained_token_cycle", "no_eos_or_length_limit"])
        report = summarize_predictions(self.details, "beam3")
        self.assertGreater(report["overall"]["wer"], 1)
        self.assertEqual(report["decoding_safety"]["flagged_samples"], 1)
        self.assertEqual(report["decoding_safety"]["coverage"], 0)
        self.assertEqual(report["diagnostics"]["overall"]["samples"], 1)


if __name__ == "__main__":
    unittest.main()
