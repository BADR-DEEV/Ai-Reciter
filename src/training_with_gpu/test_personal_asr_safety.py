import tempfile
from pathlib import Path
import unittest
import numpy as np
import soundfile as sf
from .test_my_audio import analyze_omissions, load_and_preprocess_audio


class PersonalAudioTests(unittest.TestCase):
    def test_sequence_comparison_does_not_reuse_one_word_for_two_reference_words(self):
        result = analyze_omissions("الله الله", "الله")
        self.assertEqual(result["apparent_deletions"], 1)
        self.assertFalse(result["exact_normalized_text_match"])
        self.assertNotIn("true_omissions", result)
        self.assertNotIn("slips", result)

    def test_personal_audio_is_never_silently_truncated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "long.wav"
            sf.write(path, np.ones(31 * 16000, dtype=np.float32) * .01, 16000, subtype="PCM_16")
            with self.assertRaisesRegex(ValueError, "Never silently truncate"):
                load_and_preprocess_audio(path)
            sf.write(path, np.ones(16000, dtype=np.float32) * .01, 16000, subtype="PCM_16")
            self.assertEqual(len(load_and_preprocess_audio(path)), 16000)

    def test_empty_expected_text_cannot_produce_a_success_grade(self):
        with self.assertRaises(ValueError):
            analyze_omissions("", "الله")


if __name__ == "__main__":
    unittest.main()
