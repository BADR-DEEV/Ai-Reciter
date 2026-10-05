import re
import unittest

from src.dataset_collection.qaloon_audio2text import normalize_quran_for_asr

from .feedback import compare
from .rules import MODEL_TAGS
from .targets import TOKENS_FILE, tagged_target, tagged_words

BAQARA_5 = "إِنَّ اَ۬لذِينَ كَفَرُواْ سَوَآءٌ عَلَيْهِمْ ءَٰا۬نذَرْتَهُمْ أَمْ لَمْ تُنذِرْهُمْ لَا يُؤْمِنُونَۖ ٥"
TAG = re.compile(r"<[a-z_]+>")


class Targets(unittest.TestCase):
    def test_stripping_tags_gives_the_plain_target(self):
        for text in (BAQARA_5, "يَٰأَيُّهَا اَ۬لنَّاسُ اُ۟عْبُدُواْ رَبَّكُمُ", "أَلَٓمِّٓۖ ذَٰلِكَ اَ۬لْكِتَٰبُ لَا رَيْبَۖ فِيهِ هُدىٗ لِّلْمُتَّقِينَ"):
            self.assertEqual(TAG.sub("", tagged_target(text)), normalize_quran_for_asr(text))

    def test_audible_rules_are_tagged_on_their_word(self):
        words = {word: tags for _, word, tags in tagged_words(BAQARA_5)}
        self.assertEqual(words["ان"], ["n_ghunna"])
        self.assertEqual(words["سواء"], ["mad"])
        self.assertEqual(words["اانذرتهم"], ["tasheel", "n_ikhfa"])
        self.assertEqual(words["الذين"], [])  # hamzat al-waṣl and silent letters are spelling, not tags

    def test_optional_ways_get_no_tag(self):
        self.assertNotIn("madd_munfasil", MODEL_TAGS)
        self.assertEqual(tagged_target("بِمَا أُنزِلَ"), "بما انزل<n_ikhfa>")  # munfaṣil is 2 or 4 for Qālūn: no <mad>

    def test_token_file_lists_every_tag(self):
        tokens = [line for line in TOKENS_FILE.read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
        self.assertEqual(tokens, [f"<{t}>" for t in MODEL_TAGS])

    def test_acoustic_tokens_follow_the_report_mapping(self):
        self.assertEqual(tagged_target("هُدىٗ مِّن رَّبِّهِمْ"), "هدى من<m_ghunna> ربهم")  # idghām into mīm: on the doubled mīm
        self.assertEqual(tagged_target("وَمَا هُم بِمُؤْمِنِينَ"), "وما هم<m_ikhfa> بمؤمنين")
        self.assertEqual(tagged_target("عَذَابٌ أَلِيمُۢ بِمَا"), "عذاب اليم<m_ikhfa> بما")
        self.assertEqual(tagged_target("مَنْ يَّقُولُ"), "من<idgham_ghunna> يقول")
        self.assertEqual(tagged_target("مِن رَّبِّهِمْ"), "من ربهم")  # idghām without ghunna: no acoustic token
        self.assertEqual(tagged_target("إِنَّ اَ۬لذِينَ", plain=True), "ان الذين")  # plain-reading negative


class Feedback(unittest.TestCase):
    def test_missed_and_applied_rules_per_word(self):
        expected = tagged_words(BAQARA_5)
        hyp_words = [w for _, w, _ in expected]
        hyp_tags = [list(tags) for _, _, tags in expected]
        hyp_tags[[w for _, w, _ in expected].index("سواء")] = []
        result = compare(BAQARA_5, hyp_words, hyp_tags)
        sawa = next(w for w in result["words"] if w["text"] == "سواء")
        self.assertEqual(sawa["missed"], ["mad"])
        self.assertEqual(result["rules"]["mad"]["applied"], 0)
        self.assertEqual(result["rules"]["n_ghunna"], {**result["rules"]["n_ghunna"], "expected": 1, "applied": 1})
        self.assertEqual([(m["text"], m["tag"]) for m in result["mistakes"]], [("سواء", "mad")])
        self.assertIn("4 counts", result["mistakes"][0]["fix_en"])
        self.assertLess(result["score"], 1)

    def test_unrecognised_words_are_not_judged(self):
        result = compare(BAQARA_5, ["ان", "الذين"], [["n_ghunna"], []])
        self.assertEqual([w["text"] for w in result["words"]], ["ان"])
        self.assertEqual(result["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
