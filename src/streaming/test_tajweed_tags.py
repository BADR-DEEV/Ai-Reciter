import unittest

from .matcher import words
from .tajweed_tags import carry_tags, split_tags, strip_tags, tag_list, tagged_words


class TajweedTagTests(unittest.TestCase):
    def test_tags_attach_to_preceding_word(self):
        self.assertEqual(split_tags("سواء<tj:madd_muttasil> عليهم اانذرتهم<tj:tasheel>"),
                         ("سواء عليهم اانذرتهم", [(0, ["madd_muttasil"]), (2, ["tasheel"])]))

    def test_stray_spaces_around_and_inside_tags(self):
        expected = ("سواء عليهم", [(0, ["madd_muttasil"])])
        for text in ("سواء <tj:madd_muttasil> عليهم", "سواء<tj:madd_muttasil>عليهم", "  سواء  < tj : madd_muttasil >  عليهم "):
            self.assertEqual(split_tags(text), expected)

    def test_leading_tags_are_dropped_and_multiple_tags_kept_in_order(self):
        self.assertEqual(split_tags("<tj:ghunna> <tj:idgham>من<tj:ikhfa> <tj:ghunna><tj:ikhfa> ربهم"),
                         ("من ربهم", [(0, ["ikhfa", "ghunna"])]))
        self.assertEqual(split_tags("<tj:ghunna>"), ("", []))
        self.assertEqual(split_tags(""), ("", []))

    def test_strip_tags_matches_split_and_leaves_plain_text(self):
        self.assertEqual(strip_tags("قل<tj:qalqala> هو"), "قل هو")
        self.assertEqual(strip_tags(" قل  هو "), "قل هو")
        self.assertEqual(strip_tags(None), "")

    def test_uppercase_or_malformed_tags_are_not_tags(self):
        self.assertEqual(split_tags("قل<tj:Qalqala>")[1], [])
        self.assertEqual(words("قل<tj:Qalqala> هو"), ["قل", "هو"])  # leftovers normalize away

    def test_tagged_words_follow_matcher_normalization(self):
        heard, tags = tagged_words("سَوَآءٌ<tj:madd_muttasil> 123 <tj:x> عليهم", words)
        self.assertEqual(heard, words("سَوَآءٌ 123 عليهم"))
        self.assertEqual(tags, [["madd_muttasil", "x"], []])
        self.assertEqual(tag_list(tags), [{"word_index": 0, "tags": ["madd_muttasil", "x"]}])

    def test_carry_tags_keeps_prefix_tags_and_newest_window(self):
        previous, previous_tags = ["قل", "هو", "الله"], [["a"], [], ["b"]]
        self.assertEqual(carry_tags(previous, previous_tags, ["قل", "هو"], ["الله", "احد"], [["c"], ["d"]]),
                         [["a"], [], ["c"], ["d"]])
        # A truncated prefix is found inside the previous words; unknown prefixes get no tags.
        self.assertEqual(carry_tags(previous, previous_tags, ["هو", "الله"], ["احد"], [[]]), [[], ["b"], []])
        self.assertEqual(carry_tags(previous, previous_tags, ["احد"], [], []), [[]])
        self.assertEqual(carry_tags([], [], [], ["قل"], [["x"]]), [["x"]])


if __name__ == "__main__":
    unittest.main()
