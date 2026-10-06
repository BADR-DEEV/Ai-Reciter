"""Quran search by recitation: openings, juzʾ table, alignment and ranking on a tiny text."""
import json
from pathlib import Path
import re
import unittest

from .matcher import words
from .search import JUZ_STARTS, QuranIndex, fit, juz_of, strip_openings

ROOT = Path(__file__).resolve().parents[2]
SURAHS = [
    {"id": 1, "name": "Al-Fātiḥah", "arabic": "الفاتحة", "trained": True, "ayahs": [
        {"ayah": 1, "text": "الحمد لله رب العالمين"},
        {"ayah": 2, "text": "الرحمن الرحيم"},
        {"ayah": 3, "text": "ملك يوم الدين"}]},
    {"id": 2, "name": "Al-Baqarah", "arabic": "البقرة", "trained": False, "ayahs": [
        {"ayah": 140, "text": "تلك امة قد خلت لها ما كسبت ولكم ما كسبتم"},
        {"ayah": 141, "text": "سيقول السفهاء من الناس ما ولىهم عن قبلتهم"}]},
    {"id": 11, "name": "Hūd", "arabic": "هود", "trained": False, "ayahs": [
        {"ayah": 107, "text": "خالدين فيها ما دامت السماوات والارض الا ما شاء ربك ان ربك فعال لما يريد"}]},
    {"id": 112, "name": "Al-Ikhlāṣ", "arabic": "الإخلاص", "trained": True, "ayahs": [
        {"ayah": 1, "text": "قل هو الله احد"},
        {"ayah": 2, "text": "الله الصمد"},
        {"ayah": 3, "text": "لم يلد ولم يولد"},
        {"ayah": 4, "text": "ولم يكن له كفؤا احد"}]},
    {"id": 85, "name": "Al-Burūj", "arabic": "البروج", "trained": True, "ayahs": [
        {"ayah": 15, "text": "ذو العرش المجيد"},
        {"ayah": 16, "text": "فعال لما يريد"}]},
    {"id": 114, "name": "An-Nās", "arabic": "الناس", "trained": True, "ayahs": [
        {"ayah": 1, "text": "قل اعوذ برب الناس"},
        {"ayah": 2, "text": "ملك الناس"},
        {"ayah": 3, "text": "اله الناس"}]},
]


def where(result):
    return result["start"]["surah"], result["start"]["ayah"], result["end"]["ayah"]


class JuzTests(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(juz_of(1, 1), 1)
        self.assertEqual(juz_of(2, 140), 1)
        self.assertEqual(juz_of(2, 141), 2)
        self.assertEqual(juz_of(77, 49), 29)
        self.assertEqual(juz_of(78, 1), 30)
        self.assertEqual(juz_of(114, 6), 30)

    def test_web_table_mirrors_python(self):
        source = (ROOT / "web/lib/juz.ts").read_text(encoding="utf-8")
        starts = tuple((int(s), int(a)) for s, a in re.findall(r"\[(\d+), (\d+)\]", source))
        self.assertEqual(starts, JUZ_STARTS)

    @unittest.skipUnless((ROOT / "web/public/quran/text-mapping.json").is_file(), "Quran cache not built")
    def test_starts_are_the_hafs_boundaries(self):
        hafs = [(1, 1), (2, 142), (2, 253), (3, 93), (4, 24), (4, 148), (5, 82), (6, 111), (7, 88), (8, 41),
                (9, 93), (11, 6), (12, 53), (15, 1), (17, 1), (18, 75), (21, 1), (23, 1), (25, 21), (27, 56),
                (29, 46), (33, 31), (36, 28), (39, 32), (41, 47), (46, 1), (51, 31), (58, 1), (67, 1), (78, 1)]
        mapping = json.loads((ROOT / "web/public/quran/text-mapping.json").read_text(encoding="utf-8"))
        for (surah, ayah), (_, hafs_ayah) in zip(JUZ_STARTS[1:], hafs[1:]):
            self.assertEqual(mapping[f"{surah}:{ayah}"][0], hafs_ayah, (surah, ayah))


class OpeningTests(unittest.TestCase):
    def test_taawwudh_and_basmala_are_removed(self):
        heard = words("أعوذ بالله من الشيطان الرجيم بسم الله الرحمن الرحيم قل هو الله احد")
        self.assertEqual(strip_openings(heard), (["قل", "هو", "الله", "احد"], ["taawwudh", "basmala"]))

    def test_longer_taawwudh(self):
        heard = words("أعوذ بالله السميع العليم من الشيطان الرجيم الله الصمد")
        self.assertEqual(strip_openings(heard)[0], ["الله", "الصمد"])

    def test_basmala_alone_leaves_nothing(self):
        self.assertEqual(strip_openings(words("بسم الله الرحمن الرحيم")), ([], ["basmala"]))

    def test_ayahs_that_merely_share_words_are_kept(self):
        for text in ("فاذا قرات القران فاستعذ بالله من الشيطان الرجيم", "الرحمن علم القران", "الله لا اله الا هو"):
            self.assertEqual(strip_openings(words(text)), (words(text), []))


class FitTests(unittest.TestCase):
    def test_free_ends_in_the_text_only(self):
        score, pairs = fit(["b", "c"], ["x", "a", "b", "c", "d"])
        self.assertEqual([j for _, j, _ in pairs], [2, 3])
        self.assertEqual(score, 2.0)

    def test_disallowed_words_never_match(self):
        _, pairs = fit(["a", "b"], ["a", "b"], allowed=[False, True])
        self.assertEqual([j for _, j, _ in pairs], [1])


class SearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = QuranIndex(SURAHS)

    def test_exact_passage_is_first_with_high_confidence(self):
        best = self.index.search(words("لم يلد ولم يولد"))[0]
        self.assertEqual(where(best), (112, 3, 3))
        self.assertEqual((best["confidence"], best["coverage"], best["rank"]), ("high", 1.0, 1))
        self.assertEqual(best["name"], "Al-Ikhlāṣ")
        self.assertEqual(best["juz"], [30])
        self.assertTrue(all(w["status"] == "correct" for w in best["ayahs"][0]["words"]))

    def test_a_whole_short_ayah_with_no_rival_is_a_strong_match(self):
        best = self.index.search(words("الله الصمد"))[0]
        self.assertEqual((where(best), best["confidence"]), ((112, 2, 2), "high"))
        self.assertEqual(self.index.search(words("قل هو"))[0]["confidence"], "medium")  # half an ayah only

    def test_clip_across_ayahs_reports_the_span_and_matched_words(self):
        best = self.index.search(words("الله احد الله الصمد"))[0]
        self.assertEqual(where(best), (112, 1, 2))
        self.assertEqual([w["status"] for w in best["ayahs"][0]["words"]], ["pending", "pending", "correct", "correct"])

    def test_mis_heard_words_still_find_the_passage(self):
        best = self.index.search(words("سيقول السفهاء من الناص ما ولاهم عن قبلتهم"))[0]
        self.assertEqual(where(best), (2, 141, 141))
        self.assertEqual(best["confidence"], "high")
        self.assertLess(best["coverage"], 1)

    def test_up_to_three_distinct_places_best_first(self):
        results = self.index.search(words("ملك الناس"))
        self.assertEqual(where(results[0]), (114, 2, 2))
        self.assertLessEqual(len(results), 3)
        self.assertEqual([r["rank"] for r in results], list(range(1, len(results) + 1)))
        ayahs = [(a["surah"], a["ayah"]) for r in results for a in r["ayahs"]]
        self.assertEqual(len(ayahs), len(set(ayahs)))  # no ayah shown twice
        self.assertTrue(all(a["score"] >= b["score"] for a, b in zip(results, results[1:])))

    def test_juz_scope_keeps_results_inside_it(self):
        everywhere = self.index.search(words("الحمد لله رب العالمين"))
        self.assertEqual(where(everywhere[0]), (1, 1, 1))
        in_juz_30 = self.index.search(words("الحمد لله رب العالمين"), juz={30})
        self.assertTrue(all(r["juz"] == [30] for r in in_juz_30))
        self.assertEqual(self.index.search(words("سيقول السفهاء"), juz={2})[0]["start"], {"surah": 2, "ayah": 141})
        self.assertEqual(self.index.search(words("سيقول السفهاء"), juz={1}), [])

    def test_identical_wording_prefers_the_whole_ayah(self):
        results = self.index.search(words("فعال لما يريد"))
        self.assertEqual([where(r) for r in results[:2]], [(85, 16, 16), (11, 107, 107)])
        self.assertEqual(results[0]["score"], results[1]["score"])

    def test_places_sharing_only_a_common_word_are_not_matches(self):
        self.assertEqual(self.index.search(words("الله الصمد"), juz={1}), [])
        self.assertEqual(self.index.search(words("قل هو الله احد"), juz={1}), [])

    def test_one_rare_word_is_enough_evidence(self):
        best = self.index.search(words("سيقول نسمع"))[0]  # second word mis-heard
        self.assertEqual(where(best), (2, 141, 141))

    @unittest.skipUnless((ROOT / "web/public/quran/surahs/085.json").is_file(), "Quran cache not built")
    def test_rare_word_beats_earlier_common_word_in_the_whole_quran(self):
        # A real 3 s fragment of 85:3 «وشاهد ومشهود», heard as «وشاهد وما». Hundreds of earlier
        # places match وما equally well; the rare word must still decide.
        from .search import load_index
        best = load_index(ROOT / "web/public/quran/surahs").search(words("وشاهد وما"))[0]
        self.assertEqual(where(best), (85, 3, 3))

    @unittest.skipUnless((ROOT / "web/public/quran/surahs/002.json").is_file(), "Quran cache not built")
    def test_two_weak_matches_inside_one_long_ayah_are_one_place(self):
        # rattil-v3 heard Al-Ikhlāṣ 1 as «هو الله احد»; searched in juzʾ 1, both halves land in 2:101.
        from .search import load_index
        results = load_index(ROOT / "web/public/quran/surahs").search(words("هو الله احد"), {1})
        ayahs = [(a["surah"], a["ayah"]) for r in results for a in r["ayahs"]]
        self.assertEqual(len(ayahs), len(set(ayahs)))

    def test_unknown_words_find_nothing(self):
        self.assertEqual(self.index.search(["كلمة", "غريبة"]), [])
        self.assertEqual(self.index.search([]), [])


if __name__ == "__main__":
    unittest.main()
