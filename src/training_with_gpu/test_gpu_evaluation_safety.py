import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import soundfile as sf
import torch
from .gpu_evaluation import evaluate_rows


class EvaluationAudioSafetyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "audio.wav"
        sf.write(self.path, np.ones(16000, dtype=np.float32) * .1, 16000, subtype="PCM_16")
        self.rows = [{"path": str(self.path), "surah": 112, "ayah": 1, "reciter_key": "synthetic",
                      "text_asr_normalized": "قل هو الله احد", "text_seen_in_training": False}]
        class Model(torch.nn.Module):
            config = SimpleNamespace(max_target_positions=448)
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
            def generate(self, features, **kwargs):
                if kwargs.get("return_dict_in_generate") is not True:
                    raise AssertionError("Must preserve prefix/EOS for safety audit")
                return SimpleNamespace(sequences=torch.tensor([[10, 11, 12, 13, 1, 9]] * len(features)))
        class Tokenizer:
            prefix_tokens = [10, 11, 12, 13]
            eos_token_id = 9
            def batch_decode(self, tokens, **kwargs):
                return ["قل هو الله احد"] * len(tokens)
        def features(signals, **kwargs):
            return SimpleNamespace(input_features=torch.zeros(len(signals), 80, 3000),
                attention_mask=torch.ones(len(signals), 3000, dtype=torch.long))
        self.processor = SimpleNamespace(tokenizer=Tokenizer(), feature_extractor=features)
        self.model = Model()

    def test_structured_decode_is_streamed_without_overwriting_prior_evidence(self):
        path = Path(self.directory.name) / "predictions.jsonl"
        scores, details = evaluate_rows(self.model, self.processor, self.rows, decode_profile="beam3", predictions_path=path)
        self.assertEqual(scores["overall"]["wer"], 0)
        self.assertEqual(scores["decoding_safety"]["flagged_samples"], 0)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), details[0])
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            evaluate_rows(self.model, self.processor, self.rows, predictions_path=path)
        self.assertEqual(path.read_bytes(), before)

    def test_long_and_empty_audio_fail_rather_than_truncate(self):
        for audio in (np.ones(31 * 16000, dtype=np.float32), np.array([], dtype=np.float32)):
            sf.write(self.path, audio, 16000, subtype="PCM_16")
            with self.assertRaisesRegex(ValueError, "never silently truncate"):
                evaluate_rows(self.model, self.processor, self.rows)

    def test_missing_decoder_output_cannot_silently_remove_evaluation_sample(self):
        self.processor.tokenizer.batch_decode = lambda *args, **kwargs: []
        with self.assertRaisesRegex(ValueError, "output count"):
            evaluate_rows(self.model, self.processor, self.rows)

    def test_nonspeech_has_empty_truthful_reference_not_a_manufactured_ayah(self):
        rows = [{**self.rows[0], "text_asr_normalized": ""}]
        scores, details = evaluate_rows(self.model, self.processor, rows, decode_profile="beam3")
        self.assertEqual(details[0]["reference"], "")
        self.assertEqual(scores["diagnostics"]["overall"]["reference_words"], 0)
        self.assertEqual(scores["diagnostics"]["overall"]["word_insertions"], 4)
        self.assertIsNone(scores["qaloon_lexical_fidelity"]["lexical_fidelity"])

    def test_constraints_cannot_mix_historical_normalizer_or_vowelled_labels(self):
        for kwargs in ({"normalizer_version": "qaloon-asr-v1"}, {"label_field": "normalized_with_harakat"}):
            with self.assertRaisesRegex(ValueError, "version-matched"):
                evaluate_rows(self.model, self.processor, self.rows, surah_constraint=True, **kwargs)

    def test_legacy_feature_loader_also_rejects_implicit_whisper_truncation(self):
        from src.training.qaloon_data import audio_features
        for audio in (np.ones(31 * 16000, dtype=np.float32), np.array([], dtype=np.float32)):
            sf.write(self.path, audio, 16000, subtype="PCM_16")
            with self.assertRaisesRegex(ValueError, "never silently truncate"):
                audio_features(self.processor, self.path)


if __name__ == "__main__":
    unittest.main()
