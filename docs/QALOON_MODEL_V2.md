# Qaloon model: from the team model to v4

Details of v1: `QALOON_MODEL_V1.md`.
Test voice everywhere: Sheikh Waleed, never used in training.

## Results (WER, lower is better)

| | Team model | v1 | v2 | v3 | **v4** |
|---|---|---|---|---|---|
| Normal speed | ~25% | 2.6% | 2.0% | 2.2% | **2.0%** |
| 1.25x speed | - | 19.9% | 16.9% | 13.4% | **11.9%** |
| 1.5x speed | - | 30.4% | 21.5% | 21.2% | **20.4%** |
| Qaloon words (ملك, شا, كفؤا) | - | 100% | 100% | 100% | 100% |
| Invents a skipped ayah | 0% | 0% | 0% | 0% | 0% |

Beam search 3 (server `--beams 3`) lowers WER further, e.g. v3: 2.16 -> 1.82% normal, 13.4 -> 11.1% at 1.25x; worst live decode 0.7 s (cycle 1.2 s).

In the app, fast An-Naba (1.4x), words wrongly marked missing (two runs):

| Setup (on latest main code) | Missing words |
|---|---|
| v3, original matcher | 30 / 32 |
| v3 + matcher fix | 26 / 28 |
| v3 + matcher fix + beams 3 | 24 / 18 |
| **v4 + matcher fix + beams 3** | **21 / 21** |

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

## v2 -> v3

- Added Badr's Garu clips (544 after audit; 3 Al-Asr clips shifted by one ayah removed).
- Labels from the team v2 normalizer (adds ءامنوا -> امنوا).

## v3 -> v4

- Trained from v3 on single clips plus 2,648 clips of 2-4 consecutive ayahs joined, half with pauses cut to 0.05-0.35 s (`build_joined_ayahs.py`), because the app hears several ayahs at once.
- 1 epoch, lr 5e-6, 7,853 clips, Qaloon audio only.

## App fixes

- Matcher (branch `all-branches`): with fast reading, a misheard last word made the matcher jump ahead and mark heard ayahs as missed; skipped ayahs now get credit for the words actually heard.
- Run the server with `--beams 3`.

## Open items

- No Hafs audio is used for training (RetaSy and Tarteel tlog stay out).
- Trabulsi (529 clips) not used: the dataset card marks him unauthorized.
- The 4 ayah merges in `AYAH_MERGES` and the Qaloon word list need a Qaloon teacher's review.
- Waleed's al-Fatiha labels are shifted by one ayah.

## Files

- Models (private): `Mathani-Ayat/rattil-qaloon-v1` ... `rattil-qaloon-v4`.
- Data (private): `Mathani-Ayat/qaloon-all-reciters` (all sheikhs, v2 normalizer).
- Training: `src/training_with_gpu/train_rattil.py`; data loader changes in `docs/patches/qaloon_model_v4.patch`.
- Scripts: `benchmark_*.py`, `audit_labels.py`, `segment_by_pauses.py`, `apply_team_merges.py`, `build_joined_ayahs.py`.
