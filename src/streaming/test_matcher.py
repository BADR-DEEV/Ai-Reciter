import unittest

from .matcher import RecitationTracker, alignment


def tracker(*texts):
    return RecitationTracker([{"ayah": i + 1, "normalized": text} for i, text in enumerate(texts)])


class MatcherTests(unittest.TestCase):
    def test_hafs_malik_cannot_be_colored_as_qaloon_malik(self):
        self.assertEqual(alignment(["ملك"], ["مالك"]), [])
        result = tracker("ملك يوم الدين").feed("مالك يوم الدين", final=True)
        self.assertNotEqual(result["results"][1]["words"][0]["status"], "correct")

    def test_stale_normalized_cache_is_rebuilt_from_source_text(self):
        t = RecitationTracker([{"ayah": 1, "text": "ٱلْقِيَٰمَةِ", "normalized": "القامة"}])
        self.assertEqual(t.ayahs[0]["normalized"], "القيامة")

    def test_retained_completed_context_does_not_mark_future_ayah(self):
        t = tracker("الحمد لله رب العالمين", "الرحمن الرحيم", "ملك يوم الدين")
        t.feed("الحمد لله رب العالمين", final=True, continuous=True)
        update = t.feed("الحمد لله رب العالمين", final=True, continuous=True)
        self.assertEqual(update["current"], 2)
        self.assertNotIn(2, update["results"])
        update = t.feed("الحمد لله رب العالمين ملك يوم الدين", final=True, continuous=True)
        self.assertTrue(update["complete"])
        self.assertTrue(all(w["status"] == "missed" for w in update["results"][2]["words"]))

    def test_retained_next_ayah_prefix_and_final_can_complete(self):
        t = tracker("قل هو الله احد", "الله الصمد")
        update = t.feed("قل هو الله احد الله", final=True, continuous=True)
        self.assertEqual(update["current"], 2)
        self.assertTrue(update["tentative_prefix"])
        self.assertNotIn(2, update["results"])
        self.assertTrue(t.feed("قل هو الله احد الله الصمد", final=True, continuous=True)["complete"])

    def test_uncorroborated_tail_not_preserved_as_a_green_omitted_word(self):
        t = tracker("الحمد لله رب العالمين", "الرحمن الرحيم", "ملك يوم الدين")
        update = t.feed("الحمد لله رب العالمين الرحمن", final=True, continuous=True)
        self.assertTrue(update["tentative_prefix"])
        self.assertNotIn(2, update["results"])
        update = t.feed("الحمد لله رب العالمين الرحمن ملك يوم الدين", final=True, continuous=True)
        self.assertTrue(update["complete"])
        self.assertEqual([w["status"] for w in update["results"][2]["words"]], ["missed", "missed"])

    def test_new_utterance_single_word_requires_corroboration_after_transition(self):
        t = tracker("الحمد لله رب العالمين", "الرحمن الرحيم", "ملك يوم الدين")
        t.feed("الحمد لله رب العالمين", final=True, continuous=True)
        t.clear_context()
        update = t.feed("الرحمن", final=True, continuous=True)
        self.assertTrue(update["tentative_prefix"])
        self.assertNotIn(2, update["results"])
        update = t.feed("الرحمن الرحيم", final=True, continuous=True)
        self.assertEqual(update["current"], 3)
        self.assertEqual(update["results"][2]["status"], "correct")

    def test_high_partial_match_can_recover_to_later_ayah(self):
        t = tracker("الحمد لله رب العالمين", "الرحمن الرحيم")
        t.feed("الحمد لله رب")
        result = t.feed("الحمد لله رب الرحمن الرحيم", final=True)
        self.assertTrue(result["complete"])
        self.assertTrue(result["results"][1]["final"])
        self.assertEqual(result["results"][1]["words"][-1]["status"], "missed")

    def test_fast_reading_jump_credits_ayahs_heard_in_the_same_window(self):
        # A fast reader's misheard final word (حسابا -> حسبا) made the matcher jump to the
        # furthest matching ayah and mark 78:37 missed although every word of it was heard.
        t = tracker("جزاء من ربك عطاء حسابا",
                    "رب السماوات والارض وما بينهما الرحمن لا يملكون منه خطابا",
                    "يوم يقوم الروح والملائكة صفا لا يتكلمون الا من اذن له الرحمن وقال صوابا")
        update = t.feed("جزاء من ربك عطاء حسبا رب السماوات والارض وما بينهما الرحمن لا يملكون منه خطابا "
                        "يوم يقوم الروح والملائكة صفا لا يتكلمون الا من اذن له الرحمن وقال صوابا",
                        final=True, continuous=True)
        self.assertEqual(update["results"][2]["status"], "correct")
        self.assertTrue(all(w["status"] == "correct" for w in update["results"][2]["words"]))
        # حسبا is close enough to حسابا for the fuzzy word match, as before the fix.
        self.assertEqual(update["results"][1]["status"], "correct")

    def test_fast_reading_jump_still_marks_a_really_skipped_ayah(self):
        t = tracker("جزاء من ربك عطاء حسابا",
                    "رب السماوات والارض وما بينهما الرحمن لا يملكون منه خطابا",
                    "يوم يقوم الروح والملائكة صفا لا يتكلمون الا من اذن له الرحمن وقال صوابا")
        update = t.feed("جزاء من ربك عطاء حسبا يوم يقوم الروح والملائكة صفا لا يتكلمون الا من اذن له الرحمن وقال صوابا",
                        final=True, continuous=True)
        self.assertEqual(update["results"][2]["status"], "missed")
        self.assertTrue(all(w["status"] == "missed" for w in update["results"][2]["words"]))

    def test_high_partial_does_not_advance_without_later_evidence(self):
        t = tracker("الحمد لله رب العالمين", "الرحمن الرحيم")
        self.assertEqual(t.feed("الحمد لله رب العالمون", final=True)["current"], 1)

    def test_order_and_duplicate_words(self):
        self.assertEqual(len(alignment(["الله", "الله"], ["الله"])), 1)
        self.assertLess(len(alignment(["قل", "هو", "الله"], ["الله", "هو", "قل"])), 3)

    def test_partial_green_does_not_advance(self):
        t = tracker("الحمد لله رب العالمين", "الرحمن الرحيم")
        result = t.feed("الحمد لله رب")
        self.assertEqual(result["results"][1]["status"], "correct")
        self.assertEqual(result["current"], 1)

    def test_terminal_requires_stability(self):
        t = tracker("الحمد لله رب العالمين", "الرحمن الرحيم")
        self.assertEqual(t.feed("الحمد لله رب العالمين")["current"], 1)
        self.assertEqual(t.feed("الحمد لله رب العالمين")["current"], 2)

    def test_terminal_word_alone_not_enough(self):
        t = tracker("الحمد لله رب العالمين")
        self.assertFalse(t.feed("العالمين", final=True)["complete"])

    def test_incorrect_terminal_does_not_advance(self):
        t = tracker("الحمد لله رب العالمين")
        self.assertFalse(t.feed("الحمد لله رب العالمون", final=True)["complete"])

    def test_skip_red_then_automatic_continue(self):
        t = tracker("الحمد لله رب العالمين", "الرحمن الرحيم", "ملك يوم الدين")
        t.feed("الرحمن الرحيم")
        result = t.feed("الرحمن الرحيم")
        self.assertEqual(result["results"][1]["status"], "missed")
        self.assertTrue(result["results"][1]["final"])
        self.assertEqual(result["results"][2]["status"], "correct")
        self.assertEqual(result["current"], 3)

    def test_silence_does_not_mark_omissions(self):
        t = tracker("الحمد لله رب العالمين")
        self.assertEqual(t.feed("", final=True)["results"], {})

    def test_fatiha_intro_does_not_skip_to_rahman(self):
        t = RecitationTracker([{"ayah": 1, "normalized": "الحمد لله رب العالمين"},
                               {"ayah": 2, "normalized": "الرحمن الرحيم"}], surah=1)
        self.assertEqual(t.feed("بسم الله الرحمن الرحيم", final=True)["results"], {})
        self.assertEqual(t.feed("بسم الله الرحمن الرحيم الحمد لله رب العالمين", final=True)["current"], 2)

    def test_multiple_ayahs_in_one_decode(self):
        t = tracker("قل هو الله احد", "الله الصمد")
        result = t.feed("قل هو الله احد الله الصمد", final=True)
        self.assertTrue(result["complete"])
        self.assertEqual(len(result["results"]), 2)

    def test_low_score_but_correct_last_word_advances_red(self):
        t = tracker("قل هو الله احد", "الله الصمد")
        result = t.feed("قل احد", final=True)
        self.assertEqual(result["current"], 2)
        self.assertEqual(result["results"][1]["status"], "missed")

    def test_final_mismatch_does_not_stop_session(self):
        t = tracker("الحمد لله رب العالمين", "الرحمن الرحيم")
        result = t.feed("كلام مختلف تماما", final=True)
        self.assertEqual(result["results"][1]["status"], "missed")
        self.assertEqual(result["current"], 1)
        result = t.feed("الرحمن الرحيم", final=True)
        self.assertTrue(result["complete"])

    def test_short_unrelated_tail_does_not_mark_future_ayah(self):
        t = tracker("الحمد لله رب العالمين", "الرحمن الرحيم", "ملك يوم الدين")
        t.feed("الحمد لله رب العالمين", final=True)
        result = t.feed("العالمين", final=True)
        self.assertNotIn(2, result["results"])

    def test_word_prefix_green_future_gray(self):
        t = tracker("الحمد لله رب العالمين")
        result = t.feed("الحمد لله", final=True)
        self.assertEqual([word["status"] for word in result["results"][1]["words"]],
                         ["correct", "correct", "pending", "pending"])

    def test_internal_omission_red_only_after_confirmation(self):
        t = tracker("الحمد لله رب العالمين")
        result = t.feed("الحمد رب العالمين")
        self.assertEqual(result["results"][1]["words"][1]["status"], "pending")
        result = t.feed("الحمد رب العالمين")
        self.assertEqual([word["status"] for word in result["results"][1]["words"]],
                         ["correct", "missed", "correct", "correct"])
        self.assertTrue(result["complete"])

    def test_partial_gap_confirmed_without_finishing_ayah(self):
        t = tracker("الحمد لله رب العالمين")
        t.feed("الحمد رب")
        result = t.feed("الحمد رب")
        self.assertEqual([word["status"] for word in result["results"][1]["words"]],
                         ["correct", "missed", "correct", "pending"])
        self.assertFalse(result["complete"])

    def test_skipped_tail_keeps_already_heard_words_green(self):
        t = tracker("الحمد لله رب العالمين", "الرحمن الرحيم")
        t.feed("الحمد لله")
        result = t.feed("الرحمن الرحيم", final=True)
        self.assertEqual([word["status"] for word in result["results"][1]["words"]],
                         ["correct", "correct", "missed", "missed"])

    def test_skipped_whole_ayah_marks_every_word_red(self):
        t = tracker("الحمد لله رب العالمين", "الرحمن الرحيم")
        result = t.feed("الرحمن الرحيم", final=True)
        self.assertTrue(all(word["status"] == "missed" for word in result["results"][1]["words"]))

    def test_duplicate_word_matches_first_occurrence(self):
        self.assertEqual(alignment(["الله", "الله", "الصمد"], ["الله"]), [(0, 0)])
        t = tracker("الله الله الصمد")
        result = t.feed("الله الصمد", final=True)
        self.assertEqual([word["status"] for word in result["results"][1]["words"]],
                         ["correct", "missed", "correct"])

    def test_corrected_decoder_hypothesis_clears_omission(self):
        t = tracker("الحمد لله رب العالمين")
        t.feed("الحمد رب")
        t.feed("الحمد رب")
        result = t.feed("الحمد لله رب")
        self.assertEqual(result["results"][1]["words"][1]["status"], "correct")


if __name__ == "__main__":
    unittest.main()
