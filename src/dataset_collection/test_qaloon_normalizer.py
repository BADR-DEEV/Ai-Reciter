import unittest
from .qaloon_audio2text import normalize_quran_for_asr as normalize, normalize_quran_for_asr_v1, load_quran, aligned_text, NORMALIZER_VERSION


class QaloonNormalizerTests(unittest.TestCase):
    def test_vocative_is_preserved_and_separated(self):
        for raw, expected in (("يَٰٓأَيُّهَا النَّاسُ", "يا ايها الناس"), ("يَا أَيُّهَا", "يا ايها"),
            ("يَٰبَنِيٓ", "يا بني"), ("وَيَٰقَوْمِ", "ويا قوم"), ("يَٰمُوسَىٰ", "يا موسى")):
            self.assertEqual(normalize(raw), expected)

    def test_consonantal_yaa_is_not_a_silent_seat(self):
        for raw, expected in (("ٱلْقِيَٰمَةِ", "القيامة"), ("ءَايَٰتِ", "ايات"),
            ("آيَات", "ايات"), ("دِيَٰرِهِمْ", "ديارهم"), ("هَلْ أَتَيٰكَ", "هل اتاك"),
            ("إِذْ نَادَيٰهُ", "اذ ناداه"), ("مُوسَىٰ", "موسى")):
            self.assertEqual(normalize(raw), expected)

    def test_medial_dagger_alif_cannot_create_a_false_vocative_boundary(self):
        for raw, expected in (("خَطَٰيَٰكُمْ", "خطاياكم"), ("خَطَٰيَٰنَا", "خطايانا"),
                              ("خَطَٰيَٰهُم", "خطاياهم")):
            self.assertEqual(normalize(raw), expected)

    def test_qaloon_hafs_and_terminal_forms_stay_distinct(self):
        self.assertNotEqual(normalize("مَلِكِ"), normalize("مَالِكِ"))
        self.assertNotEqual(normalize("طوى"), normalize("طوا"))
        self.assertNotEqual(normalize("مؤمن"), normalize("مومن"))

    def test_historical_bug_remains_auditable_not_retroactively_erased(self):
        self.assertEqual(normalize_quran_for_asr_v1("ٱلْقِيَٰمَةِ"), "القامة")
        self.assertEqual(normalize("ٱلْقِيَٰمَةِ"), "القيامة")

    def test_canonical_source_and_idempotence(self):
        for reference, row in load_quran().items():
            text = normalize(row["raw"])
            self.assertTrue(text, reference)
            self.assertEqual(normalize(text), text, reference)
            self.assertEqual(row["text_asr"], text, reference)
            self.assertEqual(row["normalizer_version"], NORMALIZER_VERSION)

    def test_merged_fatiha_labels_come_from_intact_sources(self):
        quran = load_quran()
        last = aligned_text(quran, 1, 7)
        self.assertEqual(last["text_asr"], normalize(last["raw"]))
        self.assertEqual(last["normalizer_version"], NORMALIZER_VERSION)


if __name__ == "__main__":
    unittest.main()
