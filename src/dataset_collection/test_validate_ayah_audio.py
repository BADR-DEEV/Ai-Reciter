from pathlib import Path
import tempfile
import unittest
import wave
from unittest.mock import patch

import numpy as np
import soundfile as sf

from .validate_ayah_audio import exact_text_check, classify_content, inspect_wav, coverage_check, metadata_schema_issues, valid_previous_terminal_evidence, file_hash
from .approve_ayah_audio import selected_reviews
from .segment_and_slice import load_source_audio, previous_terminal_evidence, VERSION


class ContentValidationTests(unittest.TestCase):
    def test_training_schema_cannot_hide_null_audio_or_wrong_filename(self):
        row = {"surah": 79, "ayah": 16, "audio_filename": "079016.wav", "relative_audio_path": "audio/079016.wav",
               "text": "نص", "text_asr_normalized": "نص", "text_raw_uthmani": "نص", "reciter": "Trabulsi",
               "normalized_with_harakat": "نص", "source_ayahs": [16]}
        self.assertEqual(metadata_schema_issues(row), [])
        self.assertTrue(metadata_schema_issues({**row, "relative_audio_path": None}))
        self.assertIn("canonical-ID-audio-filename-mismatch", metadata_schema_issues({**row, "audio_filename": "079017.wav"}))
        self.assertIn("invalid-surah-integer-ID", metadata_schema_issues({**row, "surah": True}))

    def test_coverage_detects_entirely_absent_surah_and_duplicate_padding(self):
        canonical = {(1, 1): "نص", (79, 15): "نص", (79, 16): "نص", (92, 1): "نص"}
        rows = [{"surah": 79, "ayah": 15}, {"surah": 79, "ayah": 16}]
        report = coverage_check(canonical, rows)
        self.assertIn([92, 1], report["missing_reference_ayahs"])
        self.assertIn([1, 1], report["missing_reference_ayahs"])
        padded = rows + [{"surah": 79, "ayah": 15}, {"surah": 79, "ayah": 16}]
        report = coverage_check(canonical, padded)
        self.assertEqual(len(padded), report["expected_row_count"])
        self.assertFalse(report["reference_coverage_complete"])
        self.assertEqual(report["duplicate_reference_ayahs"], [[79, 15], [79, 16]])

    def test_partial_lam_yalid_cannot_pass_full_ayah(self):
        check = exact_text_check("لم يلد ولم يولد", "لم يلد")
        self.assertFalse(check["passed"])
        self.assertEqual(check["expected_words"], 4)
        self.assertEqual(classify_content(check, 112, 3, {}), "partial-ayah-missing-words")
        self.assertEqual([op["expected"] for op in check["word_operations"] if op["operation"] == "delete"], [["ولم", "يولد"]])

    def test_next_ayah_with_previous_tail_is_not_exact(self):
        self.assertFalse(exact_text_check("ولم يكن له كفؤا احد", "ولم يولد ولم يكن له كفؤا احد")["passed"])

    def test_exact_normalized_tokens_no_fuzzy_or_wrong_reading(self):
        self.assertTrue(exact_text_check("لَمْ يَلِدْ وَلَمْ يُولَدْ", "لم يلد ولم يولد")["passed"])
        for ref, heard in (("ملك يوم الدين", "مالك يوم الدين"), ("سعى", "سعا"), ("والضحى", "والضحى والضحى")):
            self.assertFalse(exact_text_check(ref, heard)["passed"])

    def test_another_label_and_merged_ayahs_are_classified(self):
        canonical = {(112, 1): "قل هو الله احد", (112, 2): "الله الصمد", (112, 3): "لم يلد ولم يولد"}
        self.assertTrue(classify_content(exact_text_check(canonical[112, 3], canonical[112, 2]), 112, 3, canonical).startswith("matches-another-ayah"))
        merged = exact_text_check(canonical[112, 1], canonical[112, 1] + " " + canonical[112, 2])
        self.assertEqual(classify_content(merged, 112, 1, canonical), "merged-ayahs-or-leakage")

    def test_automatic_pass_and_blank_review_never_human_approve(self):
        decision = {"surah": "112", "ayah": "3", "audio_sha256": "x", "automatic_pass": "True"}
        self.assertEqual(selected_reviews([decision], {(112, 3, "x")}), {})
        decision.update({"listening_approved": "true", "boundary_approved": "true", "reviewer": "test", "reviewed_at": "test"})
        with self.assertRaises(ValueError):
            selected_reviews([decision], set())
        self.assertIn((112, 3), selected_reviews([decision], {(112, 3, "x")}))


class WavValidationTests(unittest.TestCase):
    def test_start_quarantine_does_not_destroy_independently_verified_end(self):
        sf.write(self.path, self.audio, 16000, subtype="PCM_16")
        check = exact_text_check("اذهب الى فرعون انه طغى", "اذهب الى فرعون انه طغى")
        report = {"version": VERSION, "source_sha256": "source-hash", "cleaning": {"mode": "none"},
                  "clip_verification": [{"ayah": 17, "passed": False, "own_content_passed": True,
                      "reason": "shared-start-depends-on-rejected-previous-ayah", "exact_check": check}],
                  "review_candidates": [{"ayah": 17, "path": self.path.name, "start_frame": 1000, "end_frame": 33000}],
                  "joint_boundary_events": [{"after_ayah": 17, "resolved": True, "left_end": 33000 / 16000}]}
        evidence = previous_terminal_evidence(report, 17, self.path.parent)
        self.assertIsNotNone(evidence)
        row = {"surah": 79, "ayah": 18, "start_frame": 33000, "source_sha256": "source-hash",
               "previous_boundary_ayah": 17, "previous_terminal_evidence": evidence}
        canonical = {(79, 17): check["expected"]}
        self.assertTrue(valid_previous_terminal_evidence(row, canonical))
        self.assertFalse(valid_previous_terminal_evidence({**row, "start_frame": 32999}, canonical))
        evidence["raw_audio_sha256"] = "changed"
        self.assertFalse(valid_previous_terminal_evidence(row, canonical))
        report["clip_verification"][0]["reason"] = "non-exact-raw-clip-transcript"
        self.assertIsNone(previous_terminal_evidence(report, 17, self.path.parent))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "test.wav"
        self.audio = np.zeros(16000 * 2, dtype=np.float32)
        self.audio[1600:-1600] = .2 * np.sin(2 * np.pi * 170 * np.arange(len(self.audio) - 3200) / 16000)

    def test_good_engineering_and_frames_pass_without_content_claim(self):
        sf.write(self.path, self.audio, 16000, subtype="PCM_16")
        _, _, result = inspect_wav(self.path, {"sample_rate": 16000, "start_frame": 10, "end_frame": 32010})
        self.assertTrue(result["passed"])
        self.assertEqual(result["frames"], 32000)

    def test_ffmpeg_fallback_is_explicit_and_preserves_source_duration(self):
        sf.write(self.path, self.audio, 16000, subtype="PCM_16")
        with patch("src.dataset_collection.segment_and_slice.sf.read", side_effect=RuntimeError("test decoder rejection")):
            audio, sr, report = load_source_audio(self.path)
        self.assertEqual(sr, 16000)
        self.assertEqual(len(audio), len(self.audio))
        self.assertEqual(report["decoder"], "PyAV-FFmpeg-s16-resampler")
        self.assertIn("test decoder rejection", report["fallback_reason"])

    def test_wrong_format_and_frame_count_fail(self):
        sf.write(self.path, np.column_stack([self.audio, self.audio]), 22050, subtype="PCM_16")
        _, _, result = inspect_wav(self.path, {"sample_rate": 16000, "start_frame": 0, "end_frame": 1})
        self.assertIn("require-16000Hz-mono-PCM16-WAV", result["issues"])
        self.assertIn("metadata-frame-count-mismatch", result["issues"])

    def test_silence_clipping_and_edge_truncation_fail(self):
        for audio, issue in ((np.zeros(16000), "silent-or-near-silent"), (np.ones(16000), "excessive-full-scale-clipping"),
                              (.2 * np.sin(2 * np.pi * 170 * np.arange(16000) / 16000), "voiced-end-edge-tail-needs-review")):
            sf.write(self.path, audio, 16000, subtype="PCM_16")
            _, _, result = inspect_wav(self.path)
            self.assertIn(issue, result["issues"])


if __name__ == "__main__":
    unittest.main()
