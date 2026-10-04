"""Transparent lexical Qaloon contrasts and synthetic skipped-ayah diagnostics.

Not a full riwayah, madd, vowel or tajweed assessment. Always report denominator.
"""
import jiwer

# Scope by surah, not audio ayah number: Waleed Fatiha numbering differs.
LEXICAL_CONTRASTS = ((1, "ملك", "مالك"), (112, "كفؤا", "كفوا"))


def lexical_fidelity(details, contrasts=LEXICAL_CONTRASTS):
    outcomes = []
    for row in details:
        reference, hypothesis = row["reference"].split(), row["prediction"].split()
        alignment = jiwer.process_words(row["reference"], row["prediction"])
        mapped = {}
        for chunk in alignment.alignments[0]:
            if chunk.type in {"equal", "substitute"}:
                for i, j in zip(range(chunk.ref_start_idx, chunk.ref_end_idx), range(chunk.hyp_start_idx, chunk.hyp_end_idx)):
                    mapped[i] = j
        for surah, qaloon, hafs in contrasts:
            if row["surah"] != surah:
                continue
            for index, word in enumerate(reference):
                if word == qaloon:
                    observed = hypothesis[mapped[index]] if index in mapped else None
                    outcomes.append({"reciter": row["reciter"], "surah": surah, "ayah": row["ayah"],
                        "reference_token_index": index, "qaloon": qaloon, "hafs": hafs,
                        "observed": observed, "outcome": "qaloon" if observed == qaloon else "hafs" if observed == hafs else "other-or-deleted"})
    count = len(outcomes)
    return {"eligible_occurrences": count, "exact_qaloon_occurrences": sum(r["outcome"] == "qaloon" for r in outcomes),
        "hafs_occurrences": sum(r["outcome"] == "hafs" for r in outcomes),
        "lexical_fidelity": sum(r["outcome"] == "qaloon" for r in outcomes) / count if count else None,
        "contrast_inventory": [list(c) for c in contrasts], "cases": outcomes,
        "scope": "two-explicit-lexical-contrast-probe; not-full-Qaloon-certification; normalized-vowels-unassessed"}


def phrase_present(phrase, transcript):
    needle, words = phrase.split(), transcript.split()
    return bool(needle) and any(words[i:i + len(needle)] == needle for i in range(len(words) - len(needle) + 1))


def skipped_ayah_completion(details, skipped_references):
    if len(details) != len(skipped_references) or not details:
        raise ValueError("Nonempty paired synthetic skipped-ayah references required")
    cases = []
    for row, skipped in zip(details, skipped_references):
        if phrase_present(skipped, row["reference"]):
            raise ValueError("Skipped phrase occurs in actually spoken text; completion probe is ambiguous")
        cases.append({"surah": row["surah"], "ayah": row["ayah"], "reciter": row["reciter"],
            "skipped_reference": skipped, "invented_full_skipped_phrase": phrase_present(skipped, row["prediction"])})
    return {"probes": len(cases), "invented_skipped_ayahs": sum(r["invented_full_skipped_phrase"] for r in cases),
        "invented_skipped_ayah_rate": sum(r["invented_full_skipped_phrase"] for r in cases) / len(cases),
        "cases": cases, "scope": "synthetic-whole-ayah-skip; not-real-learner-word-omission-sensitivity"}
