import json
import unittest
from pathlib import Path

from qaloon_audio2text import _test_normalize_quran_for_asr, normalize_quran_for_asr

# Qaloon (Madani) ayah numbering, which differs from Hafs in several surahs.
QALOON = json.loads((Path(__file__).with_name("QaloonData_v10(1).json")).read_text(encoding="utf-8"))
AYAH = {(row["sura_no"], row["aya_no"]): row["aya_text"] for row in QALOON}


class NormalizeQuranForAsrTest(unittest.TestCase):
    def test_existing_examples(self):
        _test_normalize_quran_for_asr()

    def test_vowelled_ya_before_dagger_alif_is_kept(self):
        self.assertEqual(normalize_quran_for_asr(AYAH[(75, 1)]), "لا اقسم بيوم القيامة")
        self.assertIn("الشياطين", normalize_quran_for_asr(AYAH[(2, 101)]))
        self.assertIn("ديارهم", normalize_quran_for_asr(AYAH[(2, 84)]))

    def test_vocative_ya_is_separated_from_its_noun(self):
        self.assertEqual(normalize_quran_for_asr(AYAH[(84, 6)]), "يا ايها الانسان انك كادح الى ربك كدحا فملاقيه")
        self.assertEqual(normalize_quran_for_asr(AYAH[(89, 30)]), "يا ايتها النفس المطمئنة")
        self.assertTrue(normalize_quran_for_asr(AYAH[(78, 40)]).endswith("يا ليتني كنت ترابا"))

    def test_unvowelled_ya_is_a_seat_for_the_alif(self):
        self.assertEqual(normalize_quran_for_asr(AYAH[(79, 15)]), "هل اتاك حديث موسى")
        self.assertIn("ادراك", normalize_quran_for_asr(AYAH[(82, 17)]))

    def test_idempotent(self):
        for key in ((84, 6), (75, 1), (79, 15)):
            once = normalize_quran_for_asr(AYAH[key])
            self.assertEqual(normalize_quran_for_asr(once), once)


if __name__ == "__main__":
    unittest.main()
