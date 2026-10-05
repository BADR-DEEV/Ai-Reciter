import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from .model_options import select_model, resolve_local_model
from .serve import main, parse_args


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
