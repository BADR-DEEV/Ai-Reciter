import unittest

import numpy as np

from src.streaming import practice


class PracticeTests(unittest.TestCase):
    def test_plain_removes_vowels_and_quranic_marks(self):
        self.assertEqual(practice.plain("حَا"), "حا")
        self.assertEqual(practice.plain("رَبِّ"), "رب")

    def test_silence_detection(self):
        self.assertTrue(practice.is_silent(np.zeros(16000, dtype=np.float32)))
        tone = 0.2 * np.sin(np.linspace(0, 400 * np.pi, 16000)).astype(np.float32)
        self.assertFalse(practice.is_silent(tone))
        self.assertTrue(practice.is_silent(np.zeros(0, dtype=np.float32)))

    def test_sound_verdicts(self):
        clear = practice.assess_sound(["حا", "ها"], 0, [-1.0, -6.0], "حا")
        self.assertEqual((clear["verdict"], clear["heard"]), ("correct", 0))
        unsure = practice.assess_sound(["حا", "ها"], 0, [-1.0, -1.2], "")
        self.assertEqual(unsure["verdict"], "close")
        wrong = practice.assess_sound(["حا", "ها"], 0, [-6.0, -1.0], "ها")
        self.assertEqual((wrong["verdict"], wrong["heard"]), ("other", 1))
        self.assertAlmostEqual(sum(wrong["probabilities"]), 1.0, places=2)
        disagree = practice.assess_sound(["ها", "حا"], 0, [-1.0, -6.0], " حا")
        self.assertEqual(disagree["verdict"], "close")

    def test_reading_marks_each_word(self):
        result = practice.assess_reading("قُلْ هُوَ اَ۬للَّهُ أَحَدٌۖ", "قل هو الله احد")
        self.assertEqual(result["verdict"], "correct")
        partial = practice.assess_reading("قُلْ هُوَ اَ۬للَّهُ أَحَدٌۖ", "قل احد")
        self.assertEqual([w["status"] for w in partial["words"]], ["correct", "missed", "missed", "correct"])
        self.assertEqual(partial["verdict"], "close")
        self.assertEqual(practice.assess_reading("قُلْ", "")["verdict"], "other")


    def test_reading_carries_heard_tajweed_tags(self):
        result = practice.assess_reading("قُلْ هُوَ اَ۬للَّهُ أَحَدٌۖ", "قل هو احد", [["qalqala"], [], ["qalqala"]])
        self.assertEqual([w.get("tags") for w in result["words"]], [["qalqala"], None, None, ["qalqala"]])
        self.assertNotIn("tags", practice.assess_reading("قُلْ", "قل", [["x"], []])["words"][0])

if __name__ == "__main__":
    unittest.main()
