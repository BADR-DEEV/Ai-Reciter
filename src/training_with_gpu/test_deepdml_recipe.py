import unittest
import contextlib
import io
import numpy as np
from .train_deepdml_lora import RECIPE
from .train_tarteel_lora import parse_args
from .train_base_full import DEEPDML_BASE, DEEPDML_SMALL, MODEL_REVISIONS, tempo_perturb, add_snr_noise
from unittest.mock import patch


class DeepDMLRecipeTests(unittest.TestCase):
    def test_requested_train_and_evaluate_entrypoints_dispatch_to_safe_recipes(self):
        from . import train, evaluate
        argv = ["--output-dir", "runs/synthetic", "--quarantine-manifest", "synthetic.json"]
        self.assertEqual(train.parse_args(argv).init_model, DEEPDML_BASE)
        with patch("src.training_with_gpu.train_deepdml_lora.main") as fitting:
            train.main(argv)
            fitting.assert_called_once_with(argv)
        with patch("src.training_with_gpu.benchmark_reciter.main") as benchmark:
            evaluate.main()
            benchmark.assert_called_once_with()
    def test_recipe_rejects_openai_and_tarteel_before_loading_data(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for model in ("openai/whisper-base", "tarteel-ai/whisper-base-ar-quran"):
                with self.assertRaises(SystemExit):
                    parse_args(["--output-dir", "runs/synthetic", "--quarantine-manifest", "synthetic.json",
                                "--init-model", model], RECIPE, allowed_models=(DEEPDML_BASE, DEEPDML_SMALL))
    def test_pinned_models_all_six_linear_families_and_12gb_recipe(self):
        args = parse_args(["--output-dir", "runs/synthetic", "--quarantine-manifest", "synthetic.json"], RECIPE)
        self.assertEqual(args.init_model, DEEPDML_BASE)
        self.assertEqual(args.rank, 32)
        self.assertEqual(args.alpha, 64)
        self.assertEqual(set(args.target_modules), {"q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"})
        self.assertEqual(args.noise_prob, .12)
        self.assertEqual((args.tempo_min, args.tempo_max), (.95, 1.05))
        self.assertEqual(args.precision, "bf16")
        self.assertTrue(args.skip_heldout_eval)
        self.assertTrue(args.repair_normalization)
        self.assertEqual(args.label_field, "text_asr_normalized")
        self.assertEqual(len(MODEL_REVISIONS[DEEPDML_SMALL]), 40)

    def test_noise_has_measured_requested_snr(self):
        t = np.arange(16000) / 16000
        audio = (.1 * np.sin(2 * np.pi * 220 * t)).astype("float32")
        noisy = add_snr_noise(audio, 30)
        snr = 10 * np.log10(np.mean(audio ** 2) / np.mean((noisy - audio) ** 2))
        self.assertAlmostEqual(float(snr), 30, places=2)
        self.assertTrue(np.isfinite(noisy).all())

    def test_tempo_preserves_pitch_and_rejects_overlength_without_cropping(self):
        t = np.arange(32000) / 16000
        audio = (.1 * np.sin(2 * np.pi * 220 * t)).astype("float32")
        stretched = tempo_perturb(audio, 16000, .95)
        self.assertEqual(len(stretched), round(len(audio) / .95))
        frequency = np.fft.rfftfreq(len(stretched), 1 / 16000)[np.argmax(np.abs(np.fft.rfft(stretched)))]
        self.assertLess(abs(frequency - 220), 2)
        long = np.ones(30 * 16000, dtype=np.float32) * .1
        self.assertTrue(np.array_equal(tempo_perturb(long, 16000, .95), long))
