# Qaloon model: from the team model to v2

Details of v1: `QALOON_MODEL_V1.md`.
Test voice everywhere: Sheikh Waleed, never used in training.

## Results

| | Team model | v1 | v2 |
|---|---|---|---|
| WER, normal speed | ~25% | 2.6% | **2.0%** |
| WER, 1.25x / 1.5x speed | - | 19.9% / 30.4% | **16.9% / 21.5%** |
| Qaloon words (e.g. ملك, شا, كفؤا) | - | 100% | 100% |
| Invents a skipped ayah | 0% | 0% | 0% |
| Fast An-Naba (1.4x) in the app, words marked missing | - | 44 | 28, **14 with the matcher fix** |

## Team model -> v1

- Start from `deepdml/whisper-base-ar-quran-mix-norm` (Quran-pretrained) instead of plain Whisper.
- Train only on the approved sheikhs: Huthaify, Husary, Dokali.
- Fixed a normalizer bug that corrupted 840 words (القيامة -> القامة, يا أيها -> اايها).
- Excluded 28 clips whose audio and label disagree (mostly Dokali boundary bleed).

## v1 -> v2

- Added 6 Qaloon sheikhs from mp3quran: Daawob, Abu Snaina, Qeniwa, Akri, Deeban, Kshidan (1,675 -> 4,661 clips).
  - Full-surah MP3s cut ayah by ayah with forced alignment against QaloonData (`segment_by_pauses.py`).
  - Each clip kept only if 3 independent models agree it matches its ayah; rejects kept for review.
  - Converted to the team's ayah numbering (`apply_team_merges.py`).
- Pitch-preserving tempo augmentation (0.85x-1.45x) for fast readers.

## App fix (branch `fix-fast-reading-skips`)

With fast reading, a misheard last word made the matcher jump ahead and mark the ayahs in between as missed, even when they were heard.
Skipped ayahs now get credit for the words actually heard (`matcher.py`, 2 new tests).

## Open items

- Trabulsi (533 clips ready) not used: the dataset card marks him unauthorized.
- The 4 ayah merges in `AYAH_MERGES` and the Qaloon word list need a Qaloon teacher's review.
- Waleed's al-Fatiha labels are shifted by one ayah.
- Code changes to existing files are in `docs/patches/qaloon_model_v2.patch` (not applied).

## Files

- Models (private): `Mathani-Ayat/rattil-qaloon-v1`, `Mathani-Ayat/rattil-qaloon-v2`.
- New scripts: `src/training_with_gpu/benchmark_*.py`, `audit_labels.py`, `src/dataset_collection/segment_by_pauses.py`, `apply_team_merges.py`.
