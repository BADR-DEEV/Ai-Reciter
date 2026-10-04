import unittest
from .train_tarteel_lora import parse_args, canonical_training_targets


class ConservativeRecipeTests(unittest.TestCase):
    def test_recipe_preserves_repeats_and_uses_conservative_selection(self):
        args = parse_args(["--output-dir", "runs/synthetic-test-only"])
        self.assertEqual(args.learning_rate, 1e-5)
        self.assertEqual(args.epochs, 3)
        self.assertEqual(args.eval_steps, 39)
        self.assertEqual(args.decode_profile, "beam3")
        self.assertEqual(args.label_field, "normalized_with_harakat")

    def test_targets_come_from_canonical_source_and_preserve_qaloon_malik(self):
        rows = [{"reciter_key": "huthaify", "surah": 1, "ayah": 3, "text_asr_normalized": "ملك يوم الدين"}]
        # Qaloon canonical numbering is verified dynamically rather than assuming
        # this synthetic ayah happens to be the malik reference.
        from src.dataset_collection.qaloon_audio2text import load_quran, normalize_quran_for_asr
        canonical = load_quran()
        reference = next(k for k, v in canonical.items() if k[0] == 1 and normalize_quran_for_asr(v["raw"]) == "ملك يوم الدين")
        rows[0].update(surah=reference[0], ayah=reference[1])
        canonical_training_targets(rows)
        self.assertEqual(rows[0]["text_asr_normalized"], "ملك يوم الدين")
        self.assertIn("مَلِكِ", rows[0]["normalized_with_harakat"])
        self.assertNotIn("مَالِكِ", rows[0]["normalized_with_harakat"])
        rows[0]["text_asr_normalized"] = "مالك يوم الدين"
        with self.assertRaisesRegex(ValueError, "conflicts"):
            canonical_training_targets(rows)

    def test_noncommuting_uthmani_mark_normalization_does_not_rewrite_reference(self):
        from src.dataset_collection.qaloon_audio2text import load_quran, normalize_quran_for_asr
        raw = load_quran()[(82, 19)]["raw"]
        expected = normalize_quran_for_asr(raw)
        rows = [{"reciter_key": "dokali", "surah": 82, "ayah": 19, "text_asr_normalized": expected}]
        canonical_training_targets(rows)
        self.assertEqual(rows[0]["text_asr_normalized"], expected)
        self.assertTrue(rows[0]["normalized_with_harakat"])


if __name__ == "__main__":
    unittest.main()
