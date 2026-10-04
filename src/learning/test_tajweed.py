import unittest
import json
import re
from .build_qalon_tajweed import annotate, RULES, ROOT


class TajweedTests(unittest.TestCase):
    def rules(self, text):
        return [s["rule"] for s in annotate(text)["spans"]]

    def test_natural_and_necessary_madd_not_seconds(self):
        self.assertIn("madd_tabii", self.rules("قَالَ لَهُ"))
        self.assertIn("madd_lazim", self.rules("الضَّآلِّينَ"))
        self.assertIn("madd_lazim", self.rules("اَلضَّآلِّينَ"))
        self.assertIn("madd_tabii", self.rules("فِيْ بَيْتٍ"))
        self.assertEqual(RULES["madd_tabii"]["harakat"], [2])
        self.assertEqual(RULES["madd_lazim"]["harakat"], [6])
        self.assertIsNone(RULES["madd_munfasil"]["harakat"])

    def test_idgham_across_words_not_dunya(self):
        self.assertIn("idgham", self.rules("مِنْ رَبِّهِمْ"))
        self.assertNotIn("idgham", self.rules("الدُّنْيَا"))

    def test_ikhfa_and_iqlab(self):
        self.assertIn("ikhfa", self.rules("مِنْ شَرِّ"))
        self.assertIn("iqlab", self.rules("مِنْ بَعْدِ"))

    def test_qalqala_and_contextual_heaviness_limits(self):
        self.assertIn("qalqala", self.rules("أَحَدٌ"))
        self.assertIn("tafkhim", self.rules("قُلْ"))
        self.assertNotIn("tafkhim", self.rules("بِسْمِ اللَّهِ"))

    def test_exact_text_and_offsets_retained_not_approved(self):
        text = "قَالَ رَبِّي"
        result = annotate(text)
        self.assertEqual(result["text"], text)
        self.assertEqual(result["status"], "needs-review")
        for span in result["spans"]:
            self.assertTrue(0 <= span["start"] < span["end"] <= len(text))

    def test_disjoint_letters_not_guessed_as_natural_word(self):
        self.assertEqual(annotate("أَلَٓمِّٓ", disjoint=True)["spans"], [])
        self.assertFalse(any("Disjoint" in w for w in annotate("أَلَمْ")["warnings"]))

    def test_generated_whole_quran_preserves_display_and_original_source(self):
        root = ROOT / "web/public/quran"
        generated = root / "qalon_majwad_mushaf.json"
        if not generated.exists():
            self.skipTest("Generate the local whole-Quran cache/tajweed draft first")
        data = json.loads(generated.read_text(encoding="utf-8"))
        self.assertEqual(data["approved_ayahs"], 0)
        self.assertEqual(len(data["surahs"]), 114)
        count = 0
        for surah in data["surahs"]:
            original = json.loads((root / "surahs" / f"{surah['id']:03d}.json").read_text(encoding="utf-8"))
            self.assertEqual(len(surah["ayahs"]), len(original["ayahs"]))
            for ayah, source in zip(surah["ayahs"], original["ayahs"]):
                text = re.sub(r"[\d\u0660-\u0669]+", "", source.get("displayText") or source["text"]).strip()
                self.assertEqual(ayah["text"], text)
                self.assertEqual(ayah["source_text"], source["text"])
                for span in ayah["spans"]:
                    self.assertTrue(0 <= span["start"] < span["end"] <= len(text.encode("utf-16-le")) // 2)
                    self.assertEqual(span["status"], "needs-review")
                count += 1
        self.assertEqual(count, 6210)
