from types import SimpleNamespace
from contextlib import redirect_stderr
import io
import unittest

from .train_base_full import configure_generation, parse_args, TARTEEL_MODEL, TARTEEL_TINY_MODEL, MODEL_REVISIONS


class TarteelSetupTests(unittest.TestCase):
    def test_tiny_initialization_is_pinned_and_uses_same_three_readers(self):
        args = parse_args([], default_model=TARTEEL_TINY_MODEL, default_reciters=["dokali", "huthaify", "husary"], default_dokali_weight=1.0)
        self.assertEqual(args.init_model, TARTEEL_TINY_MODEL)
        self.assertEqual(len(MODEL_REVISIONS[args.init_model]), 40)
        self.assertEqual(set(args.reciters), {"dokali", "huthaify", "husary"})

    def test_default_three_reader_recipe_and_no_implicit_trabulsi(self):
        args = parse_args([], default_model=TARTEEL_MODEL, default_reciters=["dokali", "huthaify", "husary"], default_dokali_weight=1.0)
        self.assertEqual(args.init_model, TARTEEL_MODEL)
        self.assertEqual(set(args.reciters), {"dokali", "huthaify", "husary"})
        self.assertEqual(args.dokali_weight, 1)
        self.assertIsNone(args.trabulsi_reviewed_manifest)

    def test_one_sided_trabulsi_option_is_rejected(self):
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            parse_args(["--trabulsi-reviewed-manifest", "unreviewed.jsonl"])

    def test_only_missing_timestamp_tail_is_supported(self):
        class Tokenizer:
            def get_vocab(self):
                return {"<|ar|>": 0, "<|transcribe|>": 1, "<|notimestamps|>": 2}
            def convert_tokens_to_ids(self, token):
                return self.get_vocab()[token]
        template = SimpleNamespace(lang_to_id={"<|ar|>": 0}, task_to_id={"transcribe": 1}, no_timestamps_token_id=2, suppress_tokens=[])
        model = SimpleNamespace(config=SimpleNamespace(vocab_size=1504, max_target_positions=448))
        configure_generation(model, SimpleNamespace(tokenizer=Tokenizer()), template)
        self.assertEqual(model.generation_config.suppress_tokens, list(range(3, 1504)))
        model.config.vocab_size = 1505
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            configure_generation(model, SimpleNamespace(tokenizer=Tokenizer()), template)

    def test_legacy_generation_metadata_upgraded_without_touching_weights(self):
        class Tokenizer:
            def __len__(self):
                return 3
            def convert_tokens_to_ids(self, token):
                return {"<|ar|>": 0, "<|transcribe|>": 1, "<|notimestamps|>": 2}[token]
            def get_vocab(self):
                return {"<|ar|>": 0, "<|transcribe|>": 1, "<|notimestamps|>": 2}
        template = SimpleNamespace(lang_to_id={"<|ar|>": 0}, task_to_id={"transcribe": 1}, no_timestamps_token_id=2)
        weights = object()
        model = SimpleNamespace(config=SimpleNamespace(vocab_size=3, max_target_positions=448, max_length=1024), weights=weights)
        configure_generation(model, SimpleNamespace(tokenizer=Tokenizer()), template)
        self.assertEqual(model.generation_config.max_length, 448)
        self.assertEqual(model.config.max_length, 20)
        self.assertEqual(model.generation_config.language, "arabic")
        self.assertIsNone(model.generation_config.forced_decoder_ids)
        self.assertIs(model.weights, weights)


if __name__ == "__main__":
    unittest.main()
