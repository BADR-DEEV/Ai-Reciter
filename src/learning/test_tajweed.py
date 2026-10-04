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
        self.assertEqual(RULES["madd_munfasil"]["harakat"], [2, 4])
        self.assertEqual(RULES["madd_muttasil"]["harakat"], [4])
        self.assertIsNone(RULES["madd_changed_hamza"]["harakat"])

    def test_qaloon_madd_options_and_changed_adjacent_hamza(self):
        self.assertIn("madd_muttasil", self.rules("جَاءَكُمْ"))
        self.assertIn("madd_munfasil", self.rules("بِمَا أُنْزِلَ"))
        self.assertIn("madd_changed_hamza", self.rules("جَاءَ أَمْرُنَا"))
        self.assertNotIn("madd_muttasil", self.rules("جَاءَ أَمْرُنَا"))

    def test_source_silent_marks_are_used_without_changing_display(self):
        display = "قَالُوا"
        result = annotate(display, source_text="قَالُوا۟")
        silent = [s for s in result["spans"] if s["rule"] == "silent"]
        self.assertEqual([display[s["start"]:s["end"]] for s in silent], ["ا"])
        self.assertEqual(result["text"], display)
        self.assertFalse(any(s["rule"].startswith("madd") and s["start"] == 6 for s in result["spans"]))

    def test_nasal_and_merged_letters_distinguish_nasal_idgham(self):
        self.assertIn("ghunna", self.rules("مِنْ مَالٍ"))
        self.assertNotIn("ghunna", self.rules("مِنْ رَبِّهِمْ"))
        self.assertEqual(RULES["idgham"]["group"], "merged")
        self.assertEqual(RULES["ikhfa"]["group"], "nasal")

    def test_supported_ra_and_allah_lam_contexts(self):
        self.assertIn("tafkhim", self.rules("رَبِّ"))
        self.assertNotIn("tafkhim", self.rules("فِرْعَوْنَ"))
        self.assertIn("tafkhim", self.rules("قَالَ اللَّهُ"))
        self.assertNotIn("tafkhim", self.rules("بِسْمِ اللَّهِ"))

    def test_final_shadda_keeps_necessary_madd_in_wasl_and_waqf(self):
        self.assertIn("madd_lazim", self.rules("وَلَا جَانٌّ"))
        self.assertNotIn("madd_arid", self.rules("وَلَا جَانٌّ"))

    def test_idgham_across_words_not_dunya(self):
        self.assertIn("idgham", self.rules("مِنْ رَبِّهِمْ"))
        self.assertNotIn("idgham", self.rules("الدُّنْيَا"))

    def test_ikhfa_and_iqlab(self):
        self.assertIn("ikhfa", self.rules("مِنْ شَرِّ"))
        self.assertIn("iqlab", self.rules("مِنْ بَعْدِ"))

    def test_mim_sakin_rules_are_conditional_not_universal_mim_al_jam(self):
        self.assertIn("ikhfa_shafawi", self.rules("تَرْمِيهِمْ بِحِجَارَةٍ"))
        self.assertIn("idgham_shafawi", self.rules("لَهُمْ مَا"))
        self.assertNotIn("ikhfa_shafawi", self.rules("لَهُمُ بَيْتٌ"))
        self.assertIn("conditional", RULES["ikhfa_shafawi"]["en"])

    def test_lin_and_iwad_stop_rules_not_arid_or_qalqala(self):
        self.assertIn("madd_lin", self.rules("خَوْفٍ"))
        self.assertNotIn("madd_lin", self.rules("خَوْفٍ عَلَيْهِمْ"))
        self.assertIn("madd_iwad", self.rules("أَحَدًا"))
        self.assertIn("madd_iwad", self.rules("أَفْوَاجاٗ"))
        self.assertNotIn("qalqala", self.rules("أَحَدًا"))
        self.assertNotIn("madd_iwad", self.rules("رَحْمَةً"))
        self.assertEqual(RULES["madd_iwad"]["harakat"], [2])
        self.assertEqual(RULES["madd_lin"]["harakat"], [2, 4, 6])

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
