import unittest
from .recitation_diagnostics import lexical_fidelity, skipped_ayah_completion


class RecitationDiagnosticTests(unittest.TestCase):
    def row(self, reference, prediction, surah=1):
        return {"reciter": "synthetic", "surah": surah, "ayah": 4, "reference": reference, "prediction": prediction}
    def test_fidelity_uses_exact_one_to_one_occurrences_and_reports_denominator(self):
        report = lexical_fidelity([self.row("ملك ملك يوم الدين", "ملك مالك يوم الدين")])
        self.assertEqual(report["eligible_occurrences"], 2)
        self.assertEqual(report["lexical_fidelity"], .5)
        self.assertEqual(report["hafs_occurrences"], 1)
        self.assertIsNone(lexical_fidelity([self.row("الله", "الله")])["lexical_fidelity"])
    def test_skipped_reference_is_not_the_spoken_asr_label(self):
        row = self.row("قل هو الله احد لم يلد ولم يولد", "قل هو الله احد الله الصمد لم يلد ولم يولد", 112)
        report = skipped_ayah_completion([row], ["الله الصمد"])
        self.assertEqual(report["invented_skipped_ayah_rate"], 1)
        row["prediction"] = row["reference"]
        self.assertEqual(skipped_ayah_completion([row], ["الله الصمد"])["invented_skipped_ayah_rate"], 0)
    def test_ambiguous_skip_probe_must_not_claim_hallucination(self):
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            skipped_ayah_completion([self.row("الله الصمد", "الله الصمد", 112)], ["الله الصمد"])
