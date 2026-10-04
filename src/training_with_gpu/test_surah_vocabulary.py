import unittest
from .surah_vocabulary import SurahVocabularyConstraint


class CharacterTokenizer:
    prefix_tokens = [1000, 1001]
    eos_token_id = 999
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]
    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)


class VocabularyTrieTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()
        self.trie = SurahVocabularyConstraint(self.tokenizer, {"a", "ab", "c"})
    def prefix(self, text):
        return self.tokenizer.prefix_tokens + self.tokenizer.encode(text)
    def test_terminal_prefix_word_can_extend_or_start_new_word(self):
        allowed = self.trie.allowed(self.prefix("a"))
        self.assertIn(ord("b"), allowed)
        self.assertIn(ord(" "), allowed)
        self.assertIn(999, allowed)
        self.assertNotIn(ord("z"), allowed)
    def test_skips_restarts_and_repetitions_remain_possible(self):
        self.assertIn(999, self.trie.allowed(self.prefix("c a a c")))
        self.assertIn(999, self.trie.allowed(self.prefix("")))
        self.assertIn(999, self.trie.allowed(self.prefix(" a c")))
    def test_prefix_mismatch_or_nonvocabulary_fragment_fails_closed(self):
        for sequence in ([0, 1], self.prefix("z"), self.prefix("a z"), self.prefix("z") + [999],
                         self.prefix("a") + [999, ord("c")]):
            with self.assertRaises(ValueError):
                self.trie.allowed(sequence)
    def test_no_word_completion_is_accepted_at_mid_word(self):
        trie = SurahVocabularyConstraint(self.tokenizer, {"ab"})
        self.assertNotIn(999, trie.allowed(self.prefix("a")))
        with self.assertRaisesRegex(ValueError, "inside"):
            trie.allowed(self.prefix("a") + [999])
    def test_assisted_vocabulary_can_still_invent_a_whole_permitted_word(self):
        # This is why an assisted decoder is never the official omission score.
        self.assertIn(999, self.trie.allowed(self.prefix("a c")))

    def test_hamza_carrier_surfaces_are_accepted_without_adding_a_hafs_alif(self):
        trie = SurahVocabularyConstraint.from_canonical(self.tokenizer, 78)
        self.assertIn(999, trie.allowed(self.prefix("عن النبإ العظيم")))
        self.assertIn(999, trie.allowed(self.prefix("ألم نجعل الأرض مهادا")))
        fatiha = SurahVocabularyConstraint.from_canonical(self.tokenizer, 1)
        self.assertIn("ملك", fatiha.canonical_normalized_vocabulary)
        self.assertNotIn("مالك", fatiha.canonical_normalized_vocabulary)
        self.assertNotIn("مالك", fatiha.vocabulary)

    def test_accepted_surfaces_always_reduce_to_an_active_surah_whole_word(self):
        from src.dataset_collection.qaloon_audio2text import normalize_quran_for_asr
        trie = SurahVocabularyConstraint.from_canonical(self.tokenizer, 1)
        self.assertTrue(all(normalize_quran_for_asr(word) in trie.canonical_normalized_vocabulary
                            for word in trie.vocabulary))
