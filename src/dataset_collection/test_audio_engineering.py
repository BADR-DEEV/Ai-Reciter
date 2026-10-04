import unittest
import numpy as np
from .audit_core_audio_engineering import acoustic_facts


class PCMEngineeringTests(unittest.TestCase):
    def test_energy_flags_are_observations_and_do_not_modify_samples(self):
        audio = np.ones(16000, dtype=np.float32) * .1
        original = audio.copy()
        facts = acoustic_facts(audio, 16000)
        self.assertEqual(facts["duration_seconds"], 1)
        self.assertIn("energetic_end_review_only", facts["flags"])
        self.assertTrue(np.array_equal(original, audio))
        self.assertNotIn("boundary_approved", facts)

    def test_invalid_pcm_and_silence(self):
        for audio, sr in ((np.array([]), 16000), (np.ones((2, 3)), 16000), (np.array([np.nan]), 16000), (np.ones(5), 8000)):
            with self.assertRaises(ValueError):
                acoustic_facts(audio, sr)
        self.assertIn("near_silent", acoustic_facts(np.zeros(16000), 16000)["flags"])


if __name__ == "__main__":
    unittest.main()
