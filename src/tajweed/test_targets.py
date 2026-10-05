import re
import unittest

from src.dataset_collection.qaloon_audio2text import normalize_quran_for_asr

from .feedback import compare
from .rules import MODEL_TAGS
from .targets import TOKENS_FILE, tagged_target, tagged_words

BAQARA_5 = "إِنَّ اَ۬لذِينَ كَفَرُواْ سَوَآءٌ عَلَيْهِمْ ءَٰا۬نذَرْتَهُمْ أَمْ لَمْ تُنذِرْهُمْ لَا يُؤْمِنُونَۖ ٥"
TAG = re.compile(r"<tj:[a-z_]+>")


class Targets(unittest.TestCase):
    def test_stripping_tags_gives_the_plain_target(self):
        for text in (BAQARA_5, "يَٰأَيُّهَا اَ۬لنَّاسُ اُ۟عْبُدُواْ رَبَّكُمُ", "أَلَٓمِّٓۖ ذَٰلِكَ اَ۬لْكِتَٰبُ لَا رَيْبَۖ فِيهِ هُدىٗ لِّلْمُتَّقِينَ"):
            self.assertEqual(TAG.sub("", tagged_target(text)), normalize_quran_for_asr(text))

    def test_audible_rules_are_tagged_on_their_word(self):
        words = {word: tags for _, word, tags in tagged_words(BAQARA_5)}
        self.assertEqual(words["ان"], ["ghunna"])
        self.assertIn("madd_muttasil", words["سواء"])
        self.assertTrue({"tasheel", "ikhfa"} <= set(words["اانذرتهم"]))
        self.assertEqual(words["الذين"], [])  # hamzat al-waṣl and silent letters are spelling, not tags

    def test_optional_ways_get_no_tag(self):
        self.assertNotIn("madd_munfasil", MODEL_TAGS)
        self.assertEqual(tagged_target("بِمَا أُنزِلَ"), "بما انزل<tj:ikhfa>")  # munfaṣil on بما: no tag

    def test_token_file_lists_every_tag(self):
        tokens = [line for line in TOKENS_FILE.read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
        self.assertEqual(tokens, [f"<tj:{t}>" for t in MODEL_TAGS])


class Feedback(unittest.TestCase):
    def test_missed_and_applied_rules_per_word(self):
        expected = tagged_words(BAQARA_5)
        hyp_words = [w for _, w, _ in expected]
        hyp_tags = [list(tags) for _, _, tags in expected]
        hyp_tags[[w for _, w, _ in expected].index("سواء")] = []
        result = compare(BAQARA_5, hyp_words, hyp_tags)
        sawa = next(w for w in result["words"] if w["text"] == "سواء")
        self.assertEqual(sawa["missed"], ["madd_muttasil"])
        self.assertEqual(result["rules"]["madd_muttasil"]["applied"], 0)
        self.assertEqual(result["rules"]["ghunna"], {**result["rules"]["ghunna"], "expected": 1, "applied": 1})
        self.assertLess(result["score"], 1)

    def test_unrecognised_words_are_not_judged(self):
        result = compare(BAQARA_5, ["ان", "الذين"], [["ghunna"], []])
        self.assertEqual([w["text"] for w in result["words"]], ["ان"])
        self.assertEqual(result["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
