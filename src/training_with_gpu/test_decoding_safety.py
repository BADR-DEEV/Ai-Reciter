from types import SimpleNamespace
import unittest
from .decoding_safety import generation_diagnostics, repeated_cycle, DECODE_PROFILES


class DecodingSafetyTests(unittest.TestCase):
    tokenizer = SimpleNamespace(prefix_tokens=[10, 11, 12, 13], eos_token_id=9)

    def test_eos_is_not_confused_with_decoder_bos_or_padding(self):
        report = generation_diagnostics([10, 11, 12, 13, 1, 2, 9, 9, 9], self.tokenizer)
        self.assertTrue(report["eos_emitted"])
        self.assertTrue(report["scorable"])
        self.assertEqual(report["generated_tokens"], 2)
        empty = generation_diagnostics([10, 11, 12, 13, 9], self.tokenizer)
        self.assertFalse(empty["scorable"])
        self.assertIn("empty_decoding", empty["decode_flags"])

    def test_repeats_and_cap_require_abstention_not_silent_trim(self):
        sequence = [10, 11, 12, 13] + [1, 2] * 9
        original = sequence[:]
        report = generation_diagnostics(sequence, self.tokenizer, max_length=len(sequence))
        self.assertFalse(report["scorable"])
        self.assertTrue(report["hit_context_limit"])
        self.assertIn("sustained_token_cycle", report["decode_flags"])
        self.assertEqual(sequence, original)

    def test_legitimate_small_repeats_are_not_changed_or_flagged_as_errors(self):
        self.assertFalse(repeated_cycle([1, 2, 3] * 2))
        report = generation_diagnostics([10, 11, 12, 13] + [1, 2, 3] * 2 + [9], self.tokenizer)
        self.assertTrue(report["scorable"])
        self.assertEqual(DECODE_PROFILES["beam3"]["no_repeat_ngram_size"], 0)
        self.assertEqual(DECODE_PROFILES["beam3"]["repetition_penalty"], 1.0)
        self.assertTrue(all("max_new_tokens" not in p for p in DECODE_PROFILES.values()))

    def test_wrong_prefix_never_becomes_a_recitation_grade(self):
        self.assertFalse(generation_diagnostics([0, 1, 2, 9], self.tokenizer)["scorable"])

    def test_word_cycle_survives_different_tokenizations_but_must_abstain(self):
        tokenizer = SimpleNamespace(prefix_tokens=[10, 11, 12, 13], eos_token_id=9,
            decode=lambda ids, **kwargs: "وقال شرابا " * 8)
        # Distinct token IDs can still decode to a repeated word phrase.
        sequence = [10, 11, 12, 13] + list(range(100, 130)) + [9]
        original = sequence[:]
        result = generation_diagnostics(sequence, tokenizer)
        self.assertIn("sustained_word_cycle", result["decode_flags"])
        self.assertNotIn("sustained_token_cycle", result["decode_flags"])
        self.assertFalse(result["scorable"])
        self.assertEqual(sequence, original)


if __name__ == "__main__":
    unittest.main()
