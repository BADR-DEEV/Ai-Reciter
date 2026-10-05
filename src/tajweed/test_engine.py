import glob
import json
from pathlib import Path
import unittest

from .engine import annotate, segments
from .hafs import classify, hafs_words
from .rules import GROUPS, PRIORITY, RULES
from .text import ayah_words, display_text

QURAN = Path(__file__).resolve().parents[2] / "web/public/quran"


def rules_on(text, letters, surah=None, ayah=None, hafs=None, word=None):
    """Rules on the first cluster whose text starts with `letters`."""
    result = annotate(text, surah, ayah, hafs)
    for w, (_, clusters) in enumerate(result.words):
        if word is not None and w != word:
            continue
        for c in clusters:
            if c.text.startswith(letters):
                return set(result.rules_at(w, c.index))
    raise AssertionError(f"{letters!r} not found in {text}")


class GeneralTajweed(unittest.TestCase):
    def test_fatiha_opening(self):
        text = "اِ۬لْحَمْدُ لِلهِ رَبِّ اِ۬لْعَٰلَمِينَ"
        self.assertIn("hamzat_wasl", rules_on(text, "اِ۬"))
        self.assertIn("izhar_qamari", rules_on(text, "لْ"))
        self.assertIn("lam_tarqiq", rules_on(text, "ل", word=1) | rules_on(text, "لِ", word=1) | set(annotate(text).rules_at(1, 1)))
        self.assertIn("ra_tafkhim", rules_on(text, "رَ"))
        self.assertIn("madd_arid", rules_on(text, "ي", word=3))

    def test_baqara_5_connected_madd_and_tas_hil(self):
        text = "إِنَّ اَ۬لذِينَ كَفَرُواْ سَوَآءٌ عَلَيْهِمْ ءَٰا۬نذَرْتَهُمْ أَمْ لَمْ تُنذِرْهُمْ لَا يُؤْمِنُونَۖ"
        self.assertIn("ghunna", rules_on(text, "نَّ"))
        self.assertIn("silent", rules_on(text, "اْ"))
        self.assertIn("madd_muttasil", rules_on(text, "ا", word=3))
        self.assertIn("mim_jam", rules_on(text, "مْ", word=4))
        self.assertIn("tasheel", rules_on(text, "ا۬"))
        self.assertIn("ikhfa", rules_on(text, "ن", word=5))
        self.assertIn("ra_tarqiq", rules_on(text, "رْ", word=8))

    def test_nun_and_tanwin(self):
        self.assertIn("iqlab", rules_on("عَذَابٌ أَلِيمُۢ بِمَا كَانُواْ", "مُۢ"))
        self.assertIn("iqlab", rules_on("مِنۢ بَعْدِ ذَٰلِكَ", "نۢ"))
        self.assertIn("idgham_no_ghunna", rules_on("هُدىٗ لِّلْمُتَّقِينَ", "ى"))
        self.assertIn("idgham_ghunna", rules_on("وَمِنَ اَ۬لنَّاسِ مَنْ يَّقُولُ", "نْ"))
        self.assertIn("izhar_halqi", rules_on("سَوَآءٌ عَلَيْهِمْ", "ءٌ"))
        self.assertIn("izhar_mutlaq", rules_on("فِے اِ۬لدُّنْيَا وَالْأٓخِرَةِ", "نْ"))

    def test_mim_sakinah(self):
        text = "وَمَا هُم بِمُؤْمِنِينَ"
        self.assertTrue({"ikhfa_shafawi", "mim_jam"} <= rules_on(text, "م", word=1))
        self.assertIn("idgham_shafawi", rules_on("فِے قُلُوبِهِم مَّرَضٞ", "م", word=1))
        self.assertIn("izhar_shafawi", rules_on("عَلَيْهِمْ وَلَا", "مْ"))

    def test_qalqala(self):
        self.assertIn("qalqala", rules_on("وَمِمَّا رَزَقْنَٰهُمْ يُنفِقُونَ", "قْ"))
        self.assertIn("qalqala_waqf", rules_on("وَالرُّكَّعِ اِ۬لسُّجُودِ", "دِ"))

    def test_ra(self):
        self.assertIn("ra_tarqiq", rules_on("إِلَىٰ فِرْعَوْنَ", "رْ"))
        self.assertIn("ra_tafkhim", rules_on("كِتَٰباً فِے قِرْطَاسٖ", "رْ"))
        self.assertIn("ra_tafkhim", rules_on("اِ۪رْجِعِ اِ۪لَيْهِمْ", "رْ"))
        self.assertIn("ra_tarqiq", rules_on("وَاللَّهُ عِندَهُۥ أَجْرٌ عَظِيمٌ وَهُوَ خَيْرٌ", "رٌ", word=5))
        self.assertIn("ra_tafkhim", rules_on("وَالْفَجْرِ", "رِ"))
        self.assertIn("ra_tarqiq", rules_on("إِنْ هُوَ إِلَّا ذِكْرٌ", "رٌ"))
        self.assertIn("ra_both", rules_on("فَكَانَ كُلُّ فِرْقٖ كَالطَّوْدِ", "رْ"))

    def test_lam_of_allah(self):
        self.assertIn("lam_tafkhim", rules_on("قَالَ اَ۬للَّهُ إِنِّے", "لَّ"))
        self.assertIn("lam_tarqiq", rules_on("بِسْمِ اِ۬للَّهِ", "لَّ"))
        self.assertIn("lam_tafkhim", rules_on("اَ۬للَّهُ لَا إِلَٰهَ إِلَّا هُوَ", "لَّ"))

    def test_madd_kinds(self):
        self.assertIn("madd_lazim", rules_on("وَلَا اَ۬لضَّآلِّينَ", "آ"))
        self.assertIn("madd_munfasil", rules_on("بِمَا أُنزِلَ إِلَيْكَ", "ا", word=0))
        self.assertIn("madd_munfasil", rules_on("يَٰأَيُّهَا اَ۬لنَّاسُ", "يَٰ"))
        self.assertIn("madd_silah", rules_on("مَا حَوْلَهُۥ ذَهَبَ", "هُۥ"))
        self.assertIn("madd_munfasil", rules_on("وَرُسُلِهِۦ أَحَدٍ", "هِۦ"))
        self.assertIn("madd_badal", rules_on("ءَامَنُواْ بِاللَّهِ", "ا"))
        self.assertIn("wasl_drop", rules_on("فِے اِ۬لْأَرْضِ", "ے"))
        self.assertIn("madd_iwad", rules_on("إِنَّ اَ۬للَّهَ كَانَ عَلَيْكُمْ رَقِيباٗ", "اࣰ"))
        self.assertIn("madd_lin", rules_on("لِإِيلَٰفِ قُرَيْشٍ", "يْ"))

    def test_cases_from_the_first_draft(self):
        self.assertIn("madd_lazim", {a.rule for a in annotate("وَلَا جَانٌّ").annotations})
        self.assertNotIn("madd_arid", {a.rule for a in annotate("وَلَا جَانٌّ").annotations})
        self.assertIn("madd_lin", {a.rule for a in annotate("خَوْفٍ").annotations})
        self.assertNotIn("madd_lin", {a.rule for a in annotate("خَوْفٍ عَلَيْهِمْ").annotations})
        self.assertIn("madd_iwad", {a.rule for a in annotate("أَحَدًا").annotations})
        self.assertNotIn("qalqala_waqf", {a.rule for a in annotate("أَحَدًا").annotations})
        self.assertNotIn("madd_iwad", {a.rule for a in annotate("رَحْمَةً").annotations})
        self.assertIn("qalqala_waqf", {a.rule for a in annotate("أَحَدٌ").annotations})
        self.assertIn("ikhfa_shafawi", {a.rule for a in annotate("تَرْمِيهِمْ بِحِجَارَةٍ").annotations})

    def test_opening_letters(self):
        text = "أَلَٓمِّٓۖ ذَٰلِكَ اَ۬لْكِتَٰبُ"
        self.assertIn("madd_lazim_harfi", rules_on(text, "لَٓ", 2, 1))
        self.assertTrue({"madd_lazim_harfi", "ghunna"} <= rules_on(text, "مِّٓ", 2, 1))
        self.assertIn("madd_ayn", rules_on("كَٓهَيَعَٓصَٓۖ ذِكْرُ", "عَٓ", 19, 1))
        self.assertIn("madd_ayn", rules_on("حَمِٓ عَٓسِٓقَٓۖ كَذَٰلِكَ", "عَٓ", 42, 1))
        # Not opening letters outside the first ayah of those surahs.
        self.assertNotIn("madd_lazim_harfi", rules_on("أَلَمْ تَرَ", "لَ", 2, 5))


class QalunRiwayah(unittest.TestCase):
    def test_two_hamzas_across_words(self):
        text = "يَهْدِے مَنْ يَّشَآءُ اِ۪لَىٰ صِرَٰطٖ مُّسْتَقِيمٖ"
        self.assertIn("ibdal", rules_on(text, "اِ۪"))
        self.assertIn("ibdal", rules_on("اَ۬لسُّفَهَآءُۖ اَ۬لَا إِنَّهُمْ", "اَ۬", word=1))
        self.assertIn("tasheel", rules_on("شُهَدَآءَ ا۪ذْ حَضَرَ", "ا۪"))
        first = rules_on("بِأَسْمَآءِ هَٰؤُلَآ۟ إِن كُنتُمْ", "آ۟")
        self.assertTrue({"tasheel", "madd_changed_hamza"} <= first)
        # The article after a hamza-final word is still hamzat al-waṣl.
        self.assertIn("hamzat_wasl", rules_on("جَآءَ اَ۬لْحَقُّ", "اَ۬"))

    def test_isqat_needs_hafs(self):
        text = "حَتَّىٰ إِذَا جَا أَمْرُنَا"
        hafs = hafs_words("حَتَّىٰٓ إِذَا جَآءَ أَمۡرُنَا")
        with_hafs = rules_on(text, "ا", hafs=hafs, word=2)
        self.assertIn("isqat", with_hafs)
        self.assertNotIn("madd_munfasil", with_hafs)

    def test_mark_based_points(self):
        self.assertIn("imala", rules_on("عَلَىٰ شَفَا جُرُفٍ ه۪ارٖ فَانْهَارَ", "ه۪"))
        self.assertIn("ishmam", rules_on("لُوطاٗ س۬ےٓءَ بِهِمْ", "س۬"))
        self.assertIn("ikhtilas", rules_on("لَا تَع۬دُّواْ فِے اِ۬لسَّبْتِ", "ع۬"))
        self.assertIn("tasheel", rules_on("قُلْ أَرَٰ۬يْتَكُمْ", "رَٰ۬"))
        self.assertIn("tasheel", rules_on("فَقَٰتِلُواْ أَي۪مَّةَ اَ۬لْكُفْرِ", "ي۪"))
        self.assertIn("ha_sukun", rules_on("وَهْوَ بِكُلِّ شَےْءٍ عَلِيمٞ", "هْ"))
        self.assertIn("mim_jam", rules_on("عَلَيْهِمْ ءَٰا۬نذَرْتَهُمْ", "مْ"))

    def test_lexical_points(self):
        self.assertIn("ibdal", rules_on("إِنَّ يَاجُوجَ وَمَاجُوجَ", "ا", word=1))
        self.assertIn("ha_kinaya_qasr", rules_on("مَنْ إِن تَأْمَنْهُ بِقِنطَارٖ يُؤَدِّهِ إِلَيْكَ", "هِ", 3, 74, word=4))
        self.assertNotIn("ha_kinaya_qasr", rules_on("وَلَا يَـُٔودُهُۥ حِفْظُهُمَا", "هُۥ", 2, 254))
        self.assertIn("taqlil", rules_on("وَأَنزَلَ اَ۬لتَّوْرَيٰةَ", "يٰ"))
        self.assertIn("naql", rules_on("ءَآلَٰنَ وَقَدْ عَصَيْتَ", "ءَ", 10, 91))
        self.assertIn("hadhf", rules_on("وَالنَّصَٰرَىٰ وَالصَّٰبِينَ", "وَ", word=1))
        self.assertIn("idgham_riwaya", rules_on("اَ۪تَّخَذتُّمُ اُ۬لْعِجْلَ", "ذ"))
        self.assertIn("idgham_riwaya", rules_on("قَالَ كَمْ لَبِثْتَ", "ثْ"))
        self.assertIn("no_sakt", rules_on("كَلَّا بَلْ رَانَ عَلَىٰ قُلُوبِهِم", "لْ", 83, 14))

    def test_hafs_differences(self):
        self.assertEqual(classify("تَرَنِۦ", ayah_words("تَرَنِۦ")[0][1], hafs_words("تَرَنِ")[0], None)[0], "ya_zaida")
        self.assertEqual(classify("إِنِّيَ", ayah_words("إِنِّيَ")[0][1], hafs_words("إِنِّيٓ")[0], None)[0], "ya_idafa")
        self.assertEqual(classify("مَلِكِ", ayah_words("مَلِكِ")[0][1], hafs_words("مَٰلِكِ")[0], None)[0], "riwaya_diff")
        for qaloon, hafs in [("اَ۬لذِينَ", "ٱلَّذِينَ"), ("شَئْاٗ", "شَيْـٔٗا"), ("فَانفَجَرَتْ", "فَٱنفَجَرَتْ"),
                             ("فِصَالاً", "فِصَالًا"), ("إِبْرَٰهِيمَ", "إِبْرَٰهِـۧمَ"), ("وَهُدىٗ", "وَهُدٗى")]:
            words = ayah_words(qaloon)
            self.assertIsNone(classify(words[0][0], words[0][1], hafs_words(hafs)[0], None), qaloon)


class Rendering(unittest.TestCase):
    def test_segments_rebuild_words_and_keep_lam_alif(self):
        text = "وَلَا اَ۬لضَّآلِّينَۖ قَالَ اَ۬للَّهُ"
        result = annotate(text)
        for (display, _), parts in zip(result.words, segments(result)):
            self.assertEqual("".join(p for p, _ in parts), display)
        for parts in segments(result):
            for i in range(1, len(parts)):
                self.assertFalse(parts[i - 1][0].endswith(("ل", "لَ", "لْ")) and parts[i][0].startswith("ا"))

    def test_open_tanwin_display(self):
        self.assertEqual(display_text("هُدىٗ عَظِيمٞ ظُلُمَٰتٖ ١"), "هُدىࣰ عَظِيمࣱ ظُلُمَٰتࣲ")

    def test_registry_is_consistent(self):
        self.assertEqual(set(PRIORITY), set(RULES))
        for rule in RULES.values():
            self.assertIn(rule["group"], GROUPS)


@unittest.skipUnless((QURAN / "surahs").is_dir(), "Quran cache not generated")
class WholeQuran(unittest.TestCase):
    def test_every_ayah(self):
        for path in sorted(glob.glob(str(QURAN / "surahs/*.json"))):
            surah = json.loads(Path(path).read_text(encoding="utf-8"))
            for ayah in surah["ayahs"]:
                result = annotate(ayah["text"], surah["id"], ayah["ayah"])
                display = ayah.get("displayText") or ayah["text"]
                self.assertEqual(len(result.words), len(display.split()), (surah["id"], ayah["ayah"]))
                for (word, _), parts in zip(result.words, segments(result)):
                    self.assertEqual("".join(p for p, _ in parts), word)


if __name__ == "__main__":
    unittest.main()
