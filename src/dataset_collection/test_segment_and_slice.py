import unittest

import numpy as np

from .segment_and_slice import align_references, validate_clip_transcript, refine_ayah_boundaries, clean_audio, word_similarity
from .segment_and_slice import align_surah_with_hard_walls, slice_surah_with_waveform_tracking, find_dynamic_speech_end
from .segment_and_slice import propose_pause_boundary, quarantine_neighbor_cuts, coverage_manifest, recover_missing_references, pcm16_audio, apply_boundary_overrides
from unittest.mock import patch


def refs(*texts):
    return [{"ayah": i + 1, "text_asr": text, "text": text, "raw": text} for i, text in enumerate(texts)]


def words(text, start=0, step=.5):
    return [{"clean_word": word, "start": start + i * step, "end": start + i * step + .3} for i, word in enumerate(text.split())]


class AlignmentTests(unittest.TestCase):
    def test_final_long_a_spelling_is_comparison_only_not_hafs_substitution(self):
        result = validate_clip_transcript("والليل اذا سجا", "والليل اذا سجى")
        self.assertTrue(result["passed"])
        self.assertEqual(result["comparison_only_spelling_variants"], [{"reference": "سجى", "observed": "سجا"}])
        self.assertEqual(word_similarity("سعى", "سعا"), 1)
        self.assertNotEqual(word_similarity("ملك", "مالك"), 1)
        self.assertNotEqual(word_similarity("سعى", "سعي"), 1)
    def test_fused_audio_still_produces_every_ayah_in_order(self):
        reference = refs("الحمد لله رب العالمين", "الرحمن الرحيم", "ملك يوم الدين")
        heard = words("الحمد لله رب العالمين الرحمن الرحيم ملك يوم الدين")
        bounds, _, report = align_references(heard, reference, 10)
        self.assertEqual(list(bounds), [1, 2, 3])
        self.assertLessEqual(bounds[1]["end"], bounds[2]["start"])
        self.assertLessEqual(bounds[2]["end"], bounds[3]["start"])
        self.assertEqual(report["unmatched_ayahs"], [])

    def test_missing_terminal_never_consumes_next_ayah_with_same_terminal(self):
        reference = refs("اول كلام طويل خاتمة", "ثاني كلام مختلف خاتمة", "ثالث نص صحيح هنا")
        heard = words("اول كلام طويل ثاني كلام مختلف خاتمة ثالث نص صحيح هنا")
        bounds, _, report = align_references(heard, reference, 10)
        self.assertNotIn(1, bounds)
        self.assertEqual(list(bounds), [2, 3])
        self.assertEqual(report["unmatched_ayahs"], [1])

    def test_missing_ayah_start_is_not_guessed_from_previous_end(self):
        bounds, _, report = align_references(words("الحمد لله رب العالمين ملك يوم الدين"),
            refs("الحمد لله رب العالمين", "الرحمن الرحيم", "ملك يوم الدين"), 10)
        self.assertEqual(list(bounds), [1, 3])
        self.assertEqual(report["unmatched_ayahs"], [2])

    def test_loose_long_match_containing_next_complete_ayah_is_filtered_before_path(self):
        opening = "الحمد لله " + "كلمة " * 15
        reference = refs(opening + "العظيم الحكيم", "العظيم الحكيم", "الرحمن الرحيم")
        # First ayah's terminal is absent; next ayah repeats its entire terminal
        # phrase. A loose long match could otherwise beat/consume the short next.
        heard = words(opening + "العظيم العظيم الحكيم الرحمن الرحيم")
        bounds, _, report = align_references(heard, reference, 30)
        self.assertNotIn(1, bounds)
        self.assertEqual(list(bounds), [2, 3])
        self.assertEqual(report["unmatched_ayahs"], [1])

    def test_more_than_old_fourteen_word_window_does_not_skip_later_ayah(self):
        heard = words("الحمد لله رب العالمين " + "خطأ " * 35 + "الرحمن الرحيم ملك يوم الدين")
        bounds, _, _ = align_references(heard, refs("الحمد لله رب العالمين", "الرحمن الرحيم", "ملك يوم الدين"), 30)
        self.assertEqual(list(bounds), [1, 2, 3])

    def test_fatiha_rahim_is_not_a_preamble_keyword_wall(self):
        reference = refs("الحمد لله رب العالمين", "الرحمن الرحيم", "ملك يوم الدين")
        reference.insert(0, {"ayah": -1, "text_asr": "بسم الله الرحمن الرحيم"})
        heard = words("بسم الله الرحمن الرحيم الحمد لله رب العالمين الرحمن الرحيم ملك يوم الدين")
        bounds, preamble, report = align_surah_with_hard_walls(heard, reference, 10, 3, True, True)
        self.assertAlmostEqual(preamble, heard[3]["end"])
        self.assertEqual(list(bounds), [1, 2, 3])
        self.assertEqual(len(report["preambles"]), 1)

    def test_without_preamble_nothing_is_trimmed_by_rahim(self):
        reference = refs("الحمد لله رب العالمين", "الرحمن الرحيم")
        reference.insert(0, {"ayah": -1, "text_asr": "بسم الله الرحمن الرحيم"})
        bounds, preamble, _ = align_references(words("الحمد لله رب العالمين الرحمن الرحيم"), reference, 10)
        self.assertEqual(preamble, 0)
        self.assertEqual(list(bounds), [1, 2])

    def test_repeated_start_keeps_a_later_complete_occurrence(self):
        heard = words("الحمد لله الحمد لله رب العالمين الرحمن الرحيم")
        bounds, _, _ = align_references(heard, refs("الحمد لله رب العالمين", "الرحمن الرحيم"), 10)
        self.assertEqual(list(bounds), [1, 2])
        self.assertEqual(bounds[1]["word_start"], 2)

    def test_same_full_reference_twice_maps_two_distinct_occurrences(self):
        heard = words("كلا سوف تعلمون ثم كلام آخر كلا سوف تعلمون")
        bounds, _, _ = align_references(heard, refs("كلا سوف تعلمون", "ثم كلام آخر", "كلا سوف تعلمون"), 10)
        self.assertEqual(list(bounds), [1, 2, 3])

    def test_final_ayah_never_forced_to_eof(self):
        bounds, _, _ = align_references(words("ملك يوم الدين كلام زائد بعد النهاية"), refs("ملك يوم الدين"), 10)
        self.assertAlmostEqual(bounds[1]["end"], 1.3)
        self.assertAlmostEqual(bounds[1]["speech_wall"], 1.5)

    def test_shared_timestamp_group_cannot_be_sliced_between_ayahs(self):
        heard = [{"word": "الحمد لله رب العالمين الرحمن الرحيم", "start": 0, "end": 5}]
        bounds, _, report = align_references(heard, refs("الحمد لله رب العالمين", "الرحمن الرحيم"), 10)
        self.assertEqual(bounds, {})
        self.assertEqual(report["unmatched_ayahs"], [1, 2])

    def test_invalid_timestamps_are_reported(self):
        heard = words("الرحمن الرحيم") + [{"word": "نص", "start": float("nan"), "end": 4}]
        bounds, _, report = align_references(heard, refs("الرحمن الرحيم"), 10)
        self.assertIn(1, bounds)
        self.assertEqual(report["invalid_word_indices"], [2])

    def test_second_pass_detects_an_extra_complete_ayah(self):
        result = validate_clip_transcript("الحمد لله رب العالمين الرحمن الرحيم", "الحمد لله رب العالمين", refs("الرحمن الرحيم"))
        self.assertFalse(result["passed"])
        self.assertEqual(result["contains_later_ayahs"], [1])
        self.assertTrue(validate_clip_transcript("الحمد لله رب العالمين", "الحمد لله رب العالمين")["passed"])


class SlicingTests(unittest.TestCase):
    def setUp(self):
        self.sr = 1000
        self.audio = np.zeros(12000, dtype=np.float32)
        self.audio[:1000] = .2
        self.audio[2000:3000] = .2
        self.audio[4000:5000] = .2

    def bounds(self, intervals):
        return {number: {"start": start, "end": end, "data": {"text": str(number)}} for number, start, end in intervals}

    def test_crossing_end_is_rejected_without_shifting_later_ayahs(self):
        bounds = self.bounds([(1, 0, 4.5), (2, 2, 3), (3, 4, 5)])
        clips, report = slice_surah_with_waveform_tracking(self.audio, self.sr, bounds, 0, 3, True)
        self.assertEqual([c["ayah"] for c in clips], [2, 3])
        self.assertLess(clips[0]["start"], 2.01)
        self.assertEqual(report["rejected"][0]["ayah"], 1)

    def test_rejected_long_ayah_does_not_consume_next_onset(self):
        audio = np.ones(60000, dtype=np.float32) * .2
        bounds = self.bounds([(1, 0, 40), (2, 41, 42)])
        bounds[2]["speech_wall"] = 42.2
        clips, report = slice_surah_with_waveform_tracking(audio, self.sr, bounds, 0, 2, True)
        self.assertEqual([c["ayah"] for c in clips], [2])
        self.assertLess(clips[0]["start"], 41)
        self.assertEqual(report["rejected"][0]["reason"], "duration-out-of-range")

    def test_speech_wall_before_an_unmatched_ayah_still_blocks_tail_growth(self):
        audio = np.ones(10000, dtype=np.float32) * .2
        bounds = self.bounds([(1, 0, 1), (3, 4, 5)])
        bounds[1]["speech_wall"] = 1.5
        clips = slice_surah_with_waveform_tracking(audio, self.sr, bounds, 0, 3)
        self.assertEqual(clips[0]["end"], 1.5)

    def test_terminal_decay_does_not_scan_backwards_over_closing_speech(self):
        audio = np.zeros(10000, dtype=np.float32)
        audio[:2000] = .2; audio[5000:8000] = .2
        clips = slice_surah_with_waveform_tracking(audio, self.sr, self.bounds([(1, 0, 1)]), 0, 1)
        self.assertLess(clips[0]["end"], 2.3)

    def test_madd_tail_kept_until_decay_but_never_past_next_onset(self):
        audio = np.zeros(10000, dtype=np.float32); audio[:3500] = .2
        end = find_dynamic_speech_end(audio, self.sr, 1, 4)
        self.assertGreaterEqual(end, 3.5)
        self.assertLess(end, 3.7)
        end = find_dynamic_speech_end(np.ones(10000, dtype=np.float32) * .2, self.sr, 1, 2)
        self.assertEqual(end, 2)


class JointBoundaryAndCleaningTests(unittest.TestCase):
    def test_late_terminal_timestamp_inside_pause_does_not_erase_full_pause(self):
        sr = 16000
        audio = .2 * np.sin(2 * np.pi * 170 * np.arange(sr * 4) / sr)
        audio[round(1.8 * sr):round(2.1 * sr)] = 0
        event = propose_pause_boundary(audio, sr, 1.2, 2.05, 2.05, 3)
        self.assertTrue(event["resolved"])
        self.assertGreaterEqual(event["left_end"], 2.05 - .06)
        self.assertLess(event["left_end"], 2.1)

    def test_tuwa_internal_pause_cannot_be_selected_before_terminal_end(self):
        sr = 16000
        audio = .2 * np.sin(2 * np.pi * 170 * np.arange(sr * 5) / sr)
        # A short INTERNAL pause before طوى, then the real ayah pause.
        audio[round(1.25 * sr):round(1.6 * sr)] = 0
        audio[round(2.6 * sr):round(2.9 * sr)] = 0
        result = propose_pause_boundary(audio, sr, 1, 1.7, 1.7, 3.5)
        self.assertTrue(result["resolved"])
        self.assertGreater(result["left_end"], 2.6)
        self.assertEqual(result["left_end"], result["right_start"])

    def test_left_failure_quarantines_right_even_when_asr_omits_leaked_tuwa(self):
        entries = [{"candidate": {"ayah": 16}, "verification": {"passed": False}},
                   {"candidate": {"ayah": 17}, "verification": {"passed": True}},
                   {"candidate": {"ayah": 18}, "verification": {"passed": True}}]
        result = quarantine_neighbor_cuts(entries, {16: {}, 17: {"previous_boundary_ayah": 16}, 18: {"previous_boundary_ayah": 17}})
        self.assertFalse(result[1]["verification"]["passed"])
        self.assertTrue(result[1]["verification"]["own_content_passed"])
        self.assertTrue(result[2]["verification"]["passed"])  # no cascading skips

    def test_unsliced_previous_reference_also_quarantines_shared_start(self):
        result = quarantine_neighbor_cuts([{"candidate": {"ayah": 17}, "verification": {"passed": True}}],
                                           {16: {"flags": ["unsafe"]}, 17: {"previous_boundary_ayah": 16}})
        self.assertFalse(result[0]["verification"]["passed"])

    def test_pcm_verification_is_same_integer_sample_grid(self):
        audio = np.linspace(-.8, .8, 16000, dtype=np.float32)
        result = pcm16_audio(audio, 16000)
        self.assertEqual(len(result), len(audio))
        self.assertLessEqual(np.max(np.abs(result - audio)), 1 / 32768)

    def test_coverage_accounts_for_missing_audio_without_faking_training_rows(self):
        database = {(79, 15): {"raw": "هل اتاك حديث موسى", "text": "هل اتاك حديث موسى"},
                    (79, 16): {"raw": "اذ ناداه ربه بالواد المقدس طوى", "text": "اذ ناداه ربه بالواد المقدس طوى"}}
        with patch("src.dataset_collection.segment_and_slice.geometry_fields", return_value={}):
            result = coverage_manifest(database, {79}, [{"surah": 79, "ayah": 15, "relative_audio_path": "audio/079015.wav", "usable_for_training": False}],
                                       [{"surah": 79, "ayah_decisions": [{"ayah": 16, "reason": "terminal-check-failed"}]}], "Trabulsi", "trabulsi")
        self.assertEqual(len(result), 2)
        self.assertIsNone(result[1]["relative_audio_path"])
        self.assertEqual(result[1]["missing_reason"], "terminal-check-failed")
        self.assertFalse(result[1]["usable_for_training"])

    def test_regional_recovery_never_overlaps_an_existing_ayah(self):
        reference = refs("الحمد لله", "الرحمن الرحيم", "ملك يوم الدين")
        bounds = {1: {"start": 0, "end": 1}, 3: {"start": 4, "end": 5}}
        with patch("src.dataset_collection.segment_and_slice.transcribe_words", return_value=words("الرحمن الرحيم", start=2)):
            result, attempts = recover_missing_references(None, np.zeros(6000), 1000, reference, bounds, 6)
        self.assertIn(2, result)
        self.assertEqual(attempts[0]["recovered"], [2])
        with patch("src.dataset_collection.segment_and_slice.transcribe_words", return_value=words("الرحمن الرحيم", start=.1)):
            result, attempts = recover_missing_references(None, np.zeros(6000), 1000, reference, bounds, 6)
        self.assertNotIn(2, result)

    def test_manual_cuts_are_hashed_audited_and_do_not_approve_content(self):
        override = {"surah": 79, "ayah": 1, "source_sha256": "hash", "start_frame": 100, "end_frame": 20100,
                    "sample_rate": 16000, "reviewed_by": "test-reviewer", "reviewed_at": "test-date"}
        result = apply_boundary_overrides({}, refs("نص صحيح"), [override], 79, "hash", 32000, 16000)
        self.assertFalse(result[1]["usable_for_training"])
        clips = slice_surah_with_waveform_tracking(np.zeros(32000), 16000, result, 0, 1)
        self.assertEqual(clips[0]["start_frame"], 100)
        self.assertEqual(clips[0]["end_frame"], 20100)
        for bad in ({**override, "source_sha256": "different"}, {**override, "reviewed_by": ""},
                    {**override, "end_frame": 50000}, {**override, "ayah": 99}):
            with self.assertRaises(ValueError):
                apply_boundary_overrides({}, refs("نص صحيح"), [bad], 79, "hash", 32000, 16000)

    def test_duplicate_and_overlapping_manual_cuts_are_rejected(self):
        first = {"surah": 79, "ayah": 1, "source_sha256": "hash", "start_frame": 100, "end_frame": 20100,
                 "sample_rate": 16000, "reviewed_by": "test", "reviewed_at": "test"}
        second = {**first, "ayah": 2, "start_frame": 20000, "end_frame": 30000}
        for overrides in ([first, first], [first, second]):
            with self.assertRaises(ValueError):
                apply_boundary_overrides({}, refs("اول صحيح", "ثاني صحيح"), overrides, 79, "hash", 32000, 16000)

    def test_shared_sample_grid_never_duplicates_roundtrip_boundary_sample(self):
        sr = 16000
        audio = np.zeros(sr * 66, dtype=np.float32)
        nominal = (1028360 + .1) / sr
        bounds = {
            1: {"start": 63, "end": nominal, "speech_wall": nominal, "acoustic_end": True, "data": {"text": "اول"}},
            2: {"start": nominal, "end": 65, "data": {"text": "تال"}},
        }
        clips = slice_surah_with_waveform_tracking(audio, sr, bounds, 0, 2)
        self.assertEqual(len(clips), 2)
        self.assertEqual(clips[0]["end_frame"], 1028360)
        self.assertGreaterEqual(clips[1]["start_frame"], clips[0]["end_frame"])

    def make_audio(self, continuous=False):
        sr = 16000
        audio = np.zeros(sr * 6, dtype=np.float32)
        wave = .2 * np.sin(2 * np.pi * 170 * np.arange(len(audio)) / sr)
        audio[:round(2.3 * sr)] = wave[:round(2.3 * sr)]
        start = 2.3 if continuous else 2.65
        audio[round(start * sr):round(3.8 * sr)] = wave[round(start * sr):round(3.8 * sr)]
        return audio, sr

    def test_dhuha_saa_tail_and_next_start_move_together_to_real_pause(self):
        for ending in ("والضحى", "وما سعى"):
            with self.subTest(ending=ending):
                audio, sr = self.make_audio()
                bounds = {
                    1: {"start": 0, "end": 1.5, "last_word_start": .6, "speech_wall": 1.5,
                        "next_word_end": 3.8, "flags": [], "data": {"text": ending}},
                    2: {"start": 1.5, "end": 3.8, "last_word_start": 2.65, "flags": [], "data": {"text": "كلام تال"}},
                }
                fixed, _, events = refine_ayah_boundaries(audio, sr, bounds, 0)
                self.assertTrue(events[0]["resolved"])
                self.assertGreater(fixed[1]["end"], 2.3)
                self.assertGreater(fixed[2]["start"], 2.3)
                self.assertLess(fixed[2]["start"], 2.65)
                clips = slice_surah_with_waveform_tracking(audio, sr, fixed, 0, 2)
                self.assertEqual([c["ayah"] for c in clips], [1, 2])
                self.assertGreater(clips[0]["end"], 2.3)
                self.assertGreater(clips[1]["start"], 2.3)
                # Do not mutate the original timestamp audit.
                self.assertEqual(bounds[1]["end"], 1.5)

    def test_continuous_madd_join_is_withheld_not_guessed_or_leaked(self):
        audio, sr = self.make_audio(continuous=True)
        bounds = {
            1: {"start": 0, "end": 1.5, "last_word_start": .6, "speech_wall": 1.5,
                "next_word_end": 3.8, "flags": [], "data": {"text": "والضحى"}},
            2: {"start": 1.5, "end": 3.8, "last_word_start": 2.65, "flags": [], "data": {"text": "تال"}},
        }
        fixed, _, events = refine_ayah_boundaries(audio, sr, bounds, 0)
        self.assertFalse(events[0]["resolved"])
        self.assertIn("unresolved-previous-madd-or-join", fixed[2]["flags"])
        clips = slice_surah_with_waveform_tracking(audio, sr, fixed, 0, 2)
        self.assertEqual(clips, [])

    def test_one_word_ayah_cannot_reuse_pause_before_its_repaired_start(self):
        sr = 16000
        audio = np.zeros(sr * 6, dtype=np.float32)
        wave = (.2 * np.sin(2 * np.pi * 170 * np.arange(len(audio)) / sr)).astype(np.float32)
        audio[:round(1.4 * sr)] = wave[:round(1.4 * sr)]  # Preamble
        audio[round(1.8 * sr):round(3.3 * sr)] = wave[round(1.8 * sr):round(3.3 * sr)]
        audio[round(3.6 * sr):round(4.8 * sr)] = wave[round(3.6 * sr):round(4.8 * sr)]
        bounds = {
            1: {"start": .9, "first_word_end": 2.2, "last_word_start": .9, "end": 2.2, "speech_wall": 2.2,
                "next_word_end": 4.8, "flags": [], "data": {"text": "والضحى"}},
            2: {"start": 2.2, "last_word_start": 3.6, "end": 4.8, "flags": [], "data": {"text": "تال"}},
        }
        fixed, _, events = refine_ayah_boundaries(audio, sr, bounds, .9)
        self.assertTrue(events[0]["resolved"])
        self.assertTrue(events[1]["resolved"])
        self.assertGreater(fixed[1]["end"], fixed[1]["start"])
        self.assertGreater(fixed[1]["end"], 3.3)
        self.assertGreater(fixed[2]["start"], 3.3)

    def test_cleaner_preserves_frames_vowel_tail_and_raw_input(self):
        rng = np.random.default_rng(42)
        sr = 16000
        audio = rng.normal(0, .001, sr * 4).astype(np.float32) + .02
        t = np.arange(sr * 2) / sr
        audio[sr:3 * sr] += (.15 * np.sin(2 * np.pi * 170 * t)).astype(np.float32)
        original = audio.copy()
        cleaned, report = clean_audio(audio, sr)
        self.assertEqual(len(cleaned), len(audio))
        np.testing.assert_array_equal(audio, original)
        self.assertTrue(np.isfinite(cleaned).all())
        self.assertTrue(report["noise_reduction_applied"])
        self.assertLess(np.std(cleaned[2000:10000]), np.std(audio[2000:10000]))
        self.assertGreater(np.sqrt(np.mean(cleaned[round(2.7 * sr):round(2.95 * sr)] ** 2)), .09)

    def test_no_noise_profile_does_not_learn_voice_as_noise(self):
        sr = 16000
        audio = (.2 * np.sin(2 * np.pi * 170 * np.arange(sr) / sr)).astype(np.float32)
        cleaned, report = clean_audio(audio, sr)
        self.assertFalse(report["noise_reduction_applied"])
        self.assertEqual(len(cleaned), len(audio))

    def test_disabled_cleaning_is_bit_exact_and_invalid_input_fails(self):
        audio = np.array([.1, .2, .3], dtype=np.float32)
        cleaned, _ = clean_audio(audio, 16000, "none")
        np.testing.assert_array_equal(cleaned, audio)
        with self.assertRaises(ValueError):
            clean_audio(np.array([np.nan]), 16000)


if __name__ == "__main__":
    unittest.main()
