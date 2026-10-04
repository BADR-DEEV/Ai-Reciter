from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
import torch
from .server import Engine


class EngineSafetyTests(unittest.TestCase):
    def engine(self, ids):
        engine = Engine.__new__(Engine)
        engine.device, engine.num_beams = "cpu", 3
        engine.model = SimpleNamespace(dtype=torch.float32, config=SimpleNamespace(max_target_positions=448),
            generate=Mock(return_value=SimpleNamespace(sequences=torch.tensor([ids]))))
        class Processor:
            tokenizer = SimpleNamespace(prefix_tokens=[10, 11, 12, 13], eos_token_id=9)
            def __call__(self, audio, **kwargs):
                return SimpleNamespace(input_features=torch.zeros(1, 80, 3000), attention_mask=torch.ones(1, 3000))
            def batch_decode(self, tokens, **kwargs):
                return ["قل هو الله احد"]
        engine.processor = Processor()
        return engine
    def test_silence_does_not_decode_or_force_candidate_likelihoods(self):
        engine = self.engine([10, 11, 12, 13, 1, 9])
        with patch("src.streaming.server.lessons.candidate_logprobs") as closed_set:
            self.assertEqual(engine.check(np.zeros(16000, dtype=np.float32), ["قل", "قال"]), ("", []))
            engine.model.generate.assert_not_called()
            closed_set.assert_not_called()
    def test_model_loop_cannot_be_passed_to_learner_matcher(self):
        engine = self.engine([10, 11, 12, 13] + [1] * 20)
        self.assertEqual(engine.transcribe(np.ones(16000, dtype=np.float32) * .1), "")
        self.assertFalse(engine.last_diagnostics["scorable"])
    def test_complete_blind_decode_has_no_short_token_cap(self):
        engine = self.engine([10, 11, 12, 13, 1, 2, 9])
        self.assertEqual(engine.transcribe(np.ones(16000, dtype=np.float32) * .1), "قل هو الله احد")
        kwargs = engine.model.generate.call_args.kwargs
        self.assertEqual(kwargs["num_beams"], 3)
        self.assertEqual(kwargs["max_length"], 448)
        self.assertNotIn("max_new_tokens", kwargs)
        self.assertNotIn("prefix_allowed_tokens_fn", kwargs)
        self.assertNotIn("prompt_ids", kwargs)
    def test_long_audio_cannot_be_silently_cut(self):
        engine = self.engine([10, 11, 12, 13, 1, 9])
        with self.assertRaisesRegex(ValueError, "never silently truncate"):
            engine.transcribe(np.ones(31 * 16000, dtype=np.float32))
