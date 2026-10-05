import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from .model_options import MODEL_PRESETS, ModelSpec, check_model, models_from_env, parse_models, select_model, resolve_local_model, with_tajweed_slot
from .serve import main, parse_args

# The tajweed slot takes v2 when its folder exists on this machine, else v1.
NEWEST_TAJWEED = "rattil-tajweed-v2" if MODEL_PRESETS["rattil-tajweed-v2"].is_dir() else "rattil-tajweed-v1"


class ModelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.adapter = self.root / "run/adapter"
        self.adapter.mkdir(parents=True)
        (self.adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "deepdml/whisper-base-ar-quran-mix-norm"}), encoding="utf-8")
        (self.adapter / "adapter_model.safetensors").write_bytes(b"synthetic-fixture-not-weights")
        self.full = self.root / "gpu_base_full"
        self.full.mkdir()
        (self.full / "config.json").write_text('{}', encoding="utf-8")

    def test_deepdml_accepts_run_parent_or_exact_adapter(self):
        self.assertEqual(select_model("deepdml", self.adapter.parent), self.adapter)
        self.assertEqual(select_model("deepdml", self.adapter), self.adapter)
        self.assertEqual(resolve_local_model(self.adapter.parent), self.adapter)

    def test_presets_cannot_silently_load_the_other_model_family(self):
        self.assertEqual(select_model("gpu-full-base", self.full), self.full)
        with self.assertRaises(ValueError):
            select_model("gpu-full-base", self.adapter)
        with self.assertRaises(ValueError):
            select_model("deepdml", self.full)
        with self.assertRaisesRegex(ValueError, "missing"):
            select_model("deepdml", self.root / "missing")
        self.assertEqual(select_model("rattil-v3", self.full), self.full)
        with self.assertRaises(ValueError):
            select_model("rattil-v3", self.adapter)

    def test_launcher_sets_explicit_selection_before_single_worker_start(self):
        with patch.dict(os.environ, {"RECITER_NUM_BEAMS": "5"}), patch("uvicorn.run") as run:
            main(["--model", "deepdml", "--model-path", str(self.adapter.parent)])
            self.assertEqual(os.environ["RECITER_MODEL_PATH"], str(self.adapter))
            self.assertTrue(os.environ["RECITER_MODELS"].startswith(f"plain=deepdml:{self.adapter},tajweed={NEWEST_TAJWEED}:"))
            self.assertEqual(os.environ["RECITER_ALLOW_EXPERIMENTAL_ADAPTER"], "1")
            self.assertNotIn("RECITER_NUM_BEAMS", os.environ)
            run.assert_called_once_with("src.streaming.server:app", host="127.0.0.1", port=8000, workers=1, ws_max_size=16384)

    def test_full_selection_does_not_inherit_adapter_permission_or_decoder_override(self):
        with patch.dict(os.environ, {"RECITER_ALLOW_EXPERIMENTAL_ADAPTER": "1"}), patch("uvicorn.run"):
            main(["--model", "gpu-full-base", "--model-path", str(self.full), "--beams", "1", "--dtype", "fp16"])
            self.assertEqual(os.environ["RECITER_ALLOW_EXPERIMENTAL_ADAPTER"], "0")
            self.assertEqual(os.environ["RECITER_NUM_BEAMS"], "1")
            self.assertEqual(os.environ["RECITER_DTYPE"], "fp16")

    def test_selection_failure_never_starts_server(self):
        with patch("uvicorn.run") as run:
            with self.assertRaises(ValueError):
                main(["--model", "deepdml", "--model-path", str(self.root / "missing")])
            run.assert_not_called()

    def test_runtime_defaults_preserve_saved_decoder_and_choose_device_automatically(self):
        args = parse_args(["--model", "gpu-full-base"])
        self.assertIsNone(args.beams)
        self.assertEqual((args.device, args.dtype), ("auto", "auto"))

    def test_parse_models_names_presets_paths_and_overrides(self):
        specs = parse_models(["rattil-v3", f"tajweed={self.full}", "old=gpu-full-base:" + str(self.full)])
        self.assertEqual(list(specs), ["plain", "tajweed", "old"])
        self.assertEqual(specs["plain"], ModelSpec("plain", MODEL_PRESETS["rattil-v3"], "rattil-v3"))
        self.assertEqual((specs["tajweed"].path, specs["tajweed"].preset, specs["tajweed"].kind), (self.full, None, "tajweed"))
        self.assertEqual((specs["old"].path, specs["old"].label, specs["old"].kind), (self.full, "gpu-full-base", "plain"))
        self.assertEqual(list(parse_models(["plain=rattil-v3, tajweed=rattil-tajweed-v1"])), ["plain", "tajweed"])
        bare = parse_models(["rattil-tajweed-v1", "rattil-v3", "gpu-full-base"])
        self.assertEqual({n: s.preset for n, s in bare.items()}, {"tajweed": "rattil-tajweed-v1", "plain": "rattil-v3", "gpu-full-base": "gpu-full-base"})
        self.assertEqual(bare["tajweed"].kind, "tajweed")
        for bad in (["nope"], ["Plain=rattil-v3"], ["plain="], ["plain=rattil-v3", "plain=gpu-full-base"]):
            with self.assertRaises(ValueError):
                parse_models(bad)

    def test_tajweed_slot_and_entries_round_trip(self):
        specs = with_tajweed_slot(parse_models([f"plain={self.full}"]))
        self.assertEqual(specs["tajweed"].preset, NEWEST_TAJWEED)
        self.assertEqual(parse_models([",".join(s.entry() for s in specs.values())]), specs)
        self.assertEqual(list(with_tajweed_slot(parse_models(["mine=rattil-tajweed-v1", "rattil-v3"]))), ["mine", "plain"])

    def test_env_registry_and_legacy_single_model(self):
        self.assertEqual(list(models_from_env({"RECITER_MODELS": f"plain={self.full},x={self.full}"})), ["plain", "x", "tajweed"])
        legacy = models_from_env({"RECITER_MODEL_PATH": "openai/whisper-base"})
        self.assertFalse(legacy["plain"].local)
        self.assertTrue(legacy["plain"].present())
        self.assertEqual(models_from_env({"RECITER_MODEL_PATH": str(self.full), "RECITER_MODEL_PRESET": "gpu-full-base"})["plain"].label, "gpu-full-base")
        with self.assertRaises(ValueError):
            models_from_env({"RECITER_MODELS": " , "})

    def test_missing_tajweed_model_is_optional_but_plain_is_not(self):
        missing = ModelSpec("tajweed", self.root / "missing", "rattil-tajweed-v1")
        self.assertIsNone(check_model(missing, required=False))
        self.assertIsNone(check_model(ModelSpec("tajweed", self.adapter, "rattil-tajweed-v1"), required=False))
        with self.assertRaises(ValueError):
            check_model(missing)
        with self.assertRaises(ValueError):
            check_model(ModelSpec("extra", self.root / "missing"), required=False)
        self.assertEqual(check_model(ModelSpec("tajweed", self.full, "rattil-tajweed-v1")), self.full)

    def test_launcher_serves_several_named_models(self):
        with patch.dict(os.environ, {}), patch("uvicorn.run") as run:
            main(["--model", f"plain=rattil-v3:{self.full}", "--models", f"tajweed={self.root / 'missing'},extra=gpu-full-base:{self.full}"])
            specs = parse_models([os.environ["RECITER_MODELS"]])
            self.assertEqual(list(specs), ["plain", "tajweed", "extra"])
            self.assertEqual((specs["plain"].path, specs["plain"].preset), (self.full, "rattil-v3"))
            self.assertEqual(specs["tajweed"].path, self.root / "missing")
            self.assertEqual(os.environ["RECITER_MODEL_PRESET"], "rattil-v3")
            self.assertEqual(os.environ["RECITER_ALLOW_EXPERIMENTAL_ADAPTER"], "0")
            run.assert_called_once()

    def test_launcher_reads_env_registry_and_rejects_missing_plain_models(self):
        with patch.dict(os.environ, {"RECITER_MODELS": f"plain=gpu-full-base:{self.full}"}), patch("uvicorn.run") as run:
            self.assertEqual(list(parse_args([]).specs), ["plain", "tajweed"])
            with self.assertRaises(ValueError):
                main(["--models", f"plain=gpu-full-base:{self.full},extra={self.root / 'missing'}"])
            with self.assertRaises(SystemExit), patch("sys.stderr"):
                parse_args(["--model", "rattil-v3", "--model", "gpu-full-base", "--model-path", str(self.full)])
            with self.assertRaises(SystemExit), patch("sys.stderr"):
                parse_args(["--model", "unknown-preset"])
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
