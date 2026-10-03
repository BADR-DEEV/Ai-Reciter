import unittest
import numpy as np
from .buffer import after_advance


class BufferTests(unittest.TestCase):
    def test_next_ayah_prefix_is_retained(self):
        pcm = np.arange(32000, dtype=np.float32)
        update = {"current": 2, "results": {2: {"words": [{"status": "correct"}]}}}
        self.assertIs(after_advance(pcm, 40000, 32000, update), pcm)

    def test_pending_frames_and_overlap_not_erased(self):
        pcm = np.arange(32000, dtype=np.float32)
        result = after_advance(pcm, 40000, 32000, {"current": 2, "results": {}}, overlap=4800)
        self.assertEqual(len(result), 12800)
        np.testing.assert_equal(result, pcm[-12800:])

    def test_retained_tail_is_bounded(self):
        pcm = np.arange(300, dtype=np.float32)
        self.assertEqual(len(after_advance(pcm, 300, 300, {}, overlap=4800)), 300)

    def test_early_transition_keeps_context_until_boundary(self):
        pcm = np.arange(32000, dtype=np.float32)
        self.assertIs(after_advance(pcm, 32000, 32000, {}, final=False), pcm)
        self.assertEqual(len(after_advance(pcm, 32000, 32000, {}, final=True)), 0)

    def test_tentative_next_prefix_not_discarded_at_pause(self):
        pcm = np.arange(32000, dtype=np.float32)
        self.assertIs(after_advance(pcm, 32000, 32000, {"tentative_prefix": True}, final=True), pcm)
