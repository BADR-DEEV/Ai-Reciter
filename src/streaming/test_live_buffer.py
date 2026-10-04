import unittest
import numpy as np
import torch
from types import SimpleNamespace
from .live_buffer import AudioQueue, TranscriptOverlap
from .loop_stop import LoopStop
from .matcher import RecitationTracker


class LiveBufferTests(unittest.TestCase):
    def test_ring_wrap_keeps_every_unacknowledged_sample(self):
        queue = AudioQueue(sample_rate=10, capacity_seconds=3)
        queue.append(np.arange(20, dtype=np.float32))
        queue.discard_before(15)
        queue.append(np.arange(20, 40, dtype=np.float32))
        np.testing.assert_equal(queue.read(15, 40), np.arange(15, 40))
        with self.assertRaisesRegex(ValueError, "silently discarded"):
            queue.append(np.arange(10, dtype=np.float32))
        np.testing.assert_equal(queue.read(15, 40), np.arange(15, 40))

    def test_same_audio_origin_revises_instead_of_duplicating(self):
        join = TranscriptOverlap()
        join.accept(0, "قل هو الله")
        self.assertEqual(join.accept(0, "قل هو الله احد"), "قل هو الله احد")

    def test_exact_overlap_retains_long_ayah_prefix_and_next_ayah(self):
        join = TranscriptOverlap()
        join.accept(0, "قل هو الله احد الله الصمد")
        text = join.accept(4, "الله الصمد لم يلد ولم يولد")
        self.assertEqual(text, "قل هو الله احد الله الصمد لم يلد ولم يولد")
        before = join.raw[:]
        self.assertIsNone(join.accept(8, "كلام مختلف"))
        self.assertEqual(join.raw, before)

    def test_repeated_overlap_is_ambiguous_not_silently_deduplicated(self):
        join = TranscriptOverlap()
        join.accept(0, "الله الصمد الله الصمد")
        self.assertIsNone(join.accept(4, "الله الصمد الله الصمد"))
        # Larger same-origin acoustic re-decode preserves BOTH repetitions.
        self.assertEqual(join.accept(0, "الله الصمد الله الصمد قل"), "الله الصمد الله الصمد قل")

    def test_cut_word_can_revise_only_after_three_exact_blind_anchors(self):
        join = TranscriptOverlap()
        join.accept(0, "ملك يوم الدين اياك نعبد واياك نستعمل")
        self.assertEqual(join.accept(4, "اياك نعبد واياك نستعين اهدنا"),
                         "ملك يوم الدين اياك نعبد واياك نستعين اهدنا")

    def test_internal_waqf_carries_prefix_and_keeps_actual_repeat(self):
        join = TranscriptOverlap()
        join.accept(0, "اياك نعبد")
        join.commit_boundary(4)
        self.assertEqual(join.accept(4, "اياك نعبد واياك نستعين"), "اياك نعبد اياك نعبد واياك نستعين")

    def test_corroborated_prefix_advances_several_fast_ayahs_in_one_update(self):
        texts = ["قل هو الله احد", "الله الصمد", "لم يلد ولم يولد"]
        tracker = RecitationTracker([{"ayah": i + 1, "normalized": text} for i, text in enumerate(texts)])
        transcript = " ".join(texts)
        self.assertFalse(tracker.feed(transcript, continuous=True)["complete"])
        self.assertTrue(tracker.feed(transcript, continuous=True, stable_prefix=len(transcript.split()))["complete"])

    def test_decoder_gap_cannot_be_graded_as_learner_omissions(self):
        tracker = RecitationTracker([{"ayah": 1, "normalized": "قل هو الله احد"}, {"ayah": 2, "normalized": "الله الصمد"}])
        tracker.uncertain_audio = True
        result = tracker.feed("الله الصمد", final=True)
        self.assertFalse(any(w["status"] == "missed" for r in result["results"].values() for w in r["words"]))

    def test_sustained_loops_stop_early_but_two_legitimate_repeats_do_not(self):
        tokenizer = SimpleNamespace(prefix_tokens=[10, 11], decode=lambda ids, **kw: "")
        stopping = LoopStop(tokenizer)
        self.assertTrue(stopping(torch.tensor([[10, 11] + [1, 2] * 8]), None).item())
        self.assertFalse(stopping(torch.tensor([[10, 11] + [1, 2] * 2]), None).item())
