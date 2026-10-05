import unittest

from .matcher import words
from .tajweed_tags import carry_tags, split_tags, strip_tags, tag_list, tagged_words, transfer_tags


class TajweedTagTests(unittest.TestCase):
    def test_tags_attach_to_preceding_word(self):
        self.assertEqual(split_tags("سواء<madd_muttasil> عليهم اانذرتهم<tasheel>"),
                         ("سواء عليهم اانذرتهم", [(0, ["madd_muttasil"]), (2, ["tasheel"])]))

    def test_stray_spaces_around_and_inside_tags(self):
        expected = ("سواء عليهم", [(0, ["madd_muttasil"])])
        for text in ("سواء <madd_muttasil> عليهم", "  سواء  < madd_muttasil >  عليهم "):
            self.assertEqual(split_tags(text), expected)

    def test_first_model_tags_are_renamed(self):
        self.assertEqual(split_tags("سواء<tj:madd_muttasil> عليهم<tj:madd_arid> اانذرتهم<tj:tasheel><tj:ikhfa>"),
                         ("سواء عليهم اانذرتهم", [(0, ["mad"]), (2, ["tasheel", "n_ikhfa"])]))

    def test_a_tag_inside_a_word_does_not_split_it(self):
        # Targets always have a space after tags, so text right after a tag continues the same word.
        self.assertEqual(split_tags("اان<tasheel>ذرتهم ام"), ("اانذرتهم ام", [(0, ["tasheel"])]))
        self.assertEqual(strip_tags("ان<ghunna>ا الذين"), "انا الذين")

    def test_leading_tags_are_dropped_and_multiple_tags_kept_in_order(self):
        self.assertEqual(split_tags("<ghunna> <idgham>من<ikhfa> <ghunna><ikhfa> ربهم"),
                         ("من ربهم", [(0, ["ikhfa", "ghunna"])]))
        self.assertEqual(split_tags("<ghunna>"), ("", []))
        self.assertEqual(split_tags(""), ("", []))

    def test_strip_tags_matches_split_and_leaves_plain_text(self):
        self.assertEqual(strip_tags("قل<qalqala> هو"), "قل هو")
        self.assertEqual(strip_tags(" قل  هو "), "قل هو")
        self.assertEqual(strip_tags(None), "")

    def test_uppercase_or_malformed_tags_are_not_tags(self):
        self.assertEqual(split_tags("قل<Qalqala>")[1], [])
        self.assertEqual(words("قل<Qalqala> هو"), ["قل", "هو"])  # leftovers normalize away

    def test_tagged_words_follow_matcher_normalization(self):
        heard, tags = tagged_words("سَوَآءٌ<madd_muttasil> 123 <x> عليهم", words)
        self.assertEqual(heard, words("سَوَآءٌ 123 عليهم"))
        self.assertEqual(tags, [["madd_muttasil", "x"], []])
        self.assertEqual(tag_list(tags), [{"word_index": 0, "tags": ["madd_muttasil", "x"]}])

    def test_transfer_tags_keeps_the_plain_words(self):
        # Ahkam mode: words come from the plain model, tags from the tajweed model.
        self.assertEqual(transfer_tags("في لوح محفوظ", "في لوح<idgham_ghunna> محسن", words), "في لوح<idgham_ghunna> محفوظ")
        self.assertEqual(transfer_tags("قل هو الله احد", "قل<qalqala> الله احد<qalqala>", words), "قل<qalqala> هو الله احد<qalqala>")
        self.assertEqual(transfer_tags("قل هو", "", words), "قل هو")
        self.assertEqual(transfer_tags("", "قل<qalqala>", words), "")

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
