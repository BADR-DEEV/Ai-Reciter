import unittest
from .qaloon_data import speaker_disjoint_splits


class SpeakerSplitTests(unittest.TestCase):
    def setUp(self):
        self.splits = {name: [{"reciter_key": reader, "text_asr_normalized": text}
                             for reader in ("husary", "huthaify", "dokali", "waleed")]
                       for name, text in (("train", "known"), ("validation", "new-validation"), ("test", "new-test"))}

    def test_no_heldout_voice_enters_training(self):
        result = speaker_disjoint_splits(self.splits, "dokali", "waleed")
        self.assertEqual({r["reciter_key"] for r in result["train"]}, {"husary", "huthaify"})
        self.assertEqual({r["reciter_key"] for r in result["validation"]}, {"dokali"})
        self.assertEqual({r["reciter_key"] for r in result["test"]}, {"waleed"})
        self.assertEqual([r["text_seen_in_training"] for r in result["test"]], [True, False, False])

    def test_rejects_same_validation_test_voice(self):
        with self.assertRaises(ValueError):
            speaker_disjoint_splits(self.splits, "waleed", "waleed")

    def test_does_not_mutate_original(self):
        speaker_disjoint_splits(self.splits, "dokali", "waleed")
        self.assertNotIn("text_seen_in_training", self.splits["train"][0])
