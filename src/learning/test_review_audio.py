import unittest
from .review_audio_segments import propose_segments


def timestamped(text):
    return [{"word": word, "start": i + .1, "end": i + .9} for i, word in enumerate(text.split())]


class SegmentationTests(unittest.TestCase):
    def setUp(self):
        self.refs = [{"ayah": 0, "text": "بسم الله الرحمن الرحيم"},
                     {"ayah": 1, "text": "الحمد لله رب العالمين"},
                     {"ayah": 2, "text": "الرحمن الرحيم"},
                     {"ayah": 3, "text": "ملك يوم الدين"}]

    def test_missing_reference_is_not_forced(self):
        result = propose_segments(timestamped("الحمد لله رب العالمين ملك يوم الدين"), self.refs)
        self.assertEqual([r["ayah"] for r in result["candidates"]], [1, 3])
        self.assertEqual(result["unmatched_reference_ayahs"], [2])

    def test_repeat_is_a_separate_occurrence(self):
        result = propose_segments(timestamped("الحمد لله رب العالمين الحمد لله رب العالمين"), self.refs)
        self.assertEqual([r["occurrence"] for r in result["candidates"]], [1, 2])
        self.assertFalse(any(r["usable_for_training"] for r in result["candidates"]))

    def test_fused_clip_yields_multiple_proposals(self):
        result = propose_segments(timestamped("الحمد لله رب العالمين الرحمن الرحيم"), self.refs)
        self.assertEqual([r["ayah"] for r in result["candidates"]], [1, 2])

    def test_basmalah_is_explicit_and_overlaps_are_flagged(self):
        result = propose_segments(timestamped("بسم الله الرحمن الرحيم الحمد لله رب العالمين"), self.refs)
        self.assertEqual(result["candidates"][0]["ayah"], 0)
        self.assertIn(2, result["candidates"][0]["ambiguous_with"])

    def test_partial_restart_not_invented_as_whole_ayah(self):
        result = propose_segments(timestamped("الحمد لله الحمد لله رب"), self.refs)
        self.assertEqual(result["candidates"], [])
