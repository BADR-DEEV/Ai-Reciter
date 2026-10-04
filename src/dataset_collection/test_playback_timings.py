import unittest
from .build_playback_timings import checked_timings


class PlaybackTimingTests(unittest.TestCase):
    def test_exact_words_and_real_pauses_only(self):
        words = [{"clean_word": "قل", "start": .1, "end": .5}, {"clean_word": "هو", "start": .8, "end": 1.1}]
        result = checked_timings(words, "قُلْ هُوَ", 2)
        self.assertEqual([w["index"] for w in result], [0, 1])
        self.assertEqual(result[1]["start"], .8)

    def test_disagreement_groups_overlap_invalid_times_never_get_guessed_timings(self):
        for words in ([{"clean_word": "قل هو", "start": 0, "end": 1}],
                      [{"clean_word": "قل", "start": 0, "end": 1}, {"clean_word": "هو", "start": .5, "end": 2}],
                      [{"clean_word": "قل", "start": 0, "end": 1}, {"clean_word": "هي", "start": 1, "end": 2}],
                      [{"clean_word": "قل", "start": float("nan"), "end": 1}, {"clean_word": "هو", "start": 1, "end": 2}]):
            self.assertIsNone(checked_timings(words, "قُلْ هُوَ", 2))


if __name__ == "__main__":
    unittest.main()
