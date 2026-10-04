import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import soundfile as sf
from src.training.reviewed_audio import sha256
from .benchmark_reciter import build_skip_probes, build_repeat_probes, build_nonspeech_probes


class BenchmarkProbeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.rows = []
        for index, text in enumerate(("قل هو الله احد", "الله الصمد", "لم يلد ولم يولد"), 1):
            path = self.root / f"source_{index}.wav"
            sf.write(path, np.full(16000, index * .1, dtype="float32"), 16000, subtype="PCM_16")
            self.rows.append({"reciter_key": "synthetic", "surah": 112, "ayah": index, "path": str(path),
                              "text_asr_normalized": text, "audio_sha256": sha256(path)})
    def test_whole_skip_keeps_actual_speech_label_separate_from_absent_reference(self):
        probes, skips = build_skip_probes(self.rows, self.root)
        self.assertEqual(len(probes), 1)
        self.assertEqual(skips, ["الله الصمد"])
        self.assertEqual(probes[0]["text_asr_normalized"], "قل هو الله احد لم يلد ولم يولد")
        audio, sr = sf.read(probes[0]["path"], dtype="float32")
        self.assertEqual((sr, len(audio)), (16000, 40000))
        first = sf.read(self.rows[0]["path"], dtype="float32")[0]
        last = sf.read(self.rows[-1]["path"], dtype="float32")[0]
        self.assertTrue(np.array_equal(audio[:16000], first))
        self.assertTrue(np.array_equal(audio[-16000:], last))
    def test_true_repeated_content_is_not_deduplicated_in_reference(self):
        probes = build_repeat_probes(self.rows[:1], self.root)
        self.assertEqual(probes[0]["text_asr_normalized"], "قل هو الله احد قل هو الله احد")
        self.assertEqual(sf.info(probes[0]["path"]).duration, 2.5)
    def test_silence_and_noise_references_are_truthfully_empty(self):
        probes = build_nonspeech_probes(self.root, 112)
        self.assertEqual(len(probes), 4)
        self.assertTrue(all(p["text_asr_normalized"] == "" for p in probes))
        self.assertTrue(all(sf.info(p["path"]).duration == 28 for p in probes))
        self.assertFalse(np.any(sf.read(probes[0]["path"])[0]))
        provenance = json.loads((self.root / "nonspeech_probe_provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["seed"], 42)
