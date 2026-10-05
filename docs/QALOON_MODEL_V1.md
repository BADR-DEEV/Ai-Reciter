# Rattil Qaloon model v1 (`rattil_qaloon_v1`)

Work log and results for the Qaloon speech model, 2026-10-04.
Everything here was measured on an RTX 4060 Laptop GPU with the team's own data.
No numbers are copied from model cards.

## Summary

- `rattil_qaloon_v1` is a Whisper-base model fine-tuned on Qaloon recitations only.
- On a reciter it never heard (Waleed), it reaches 2.64% WER, against about 25% for the previous model and 4.36% for the best public model without our training.
- Where Qaloon and Hafs read a word differently, it writes the Qaloon reading 100% of the time; the public models manage 47-68%.
- It never invented an ayah the reader skipped (0 of 200 tests).
- It is the same size as the previous model, so the existing server and app use it unchanged through `RECITER_MODEL_PATH`.

## Sources and trust

The competition reference pack (المرجعية والحزمة العلمية والبيانات) names the King Fahd Glorious Quran Printing Complex (KFGQPC) edition as the approved Quran text and mp3quran.net as an approved recitation library.

- Quran text: the project's `QaloonData_v10(1).json` is identical, ayah by ayah, to the KFGQPC Qaloon v10 data for all 6,214 ayahs.
  The comparison used the [thetruetruth/quran-data-kfgqpc](https://github.com/thetruetruth/quran-data-kfgqpc) mirror; the official portal (qurancomplex.gov.sa/quran-dev) needs a browser, so a final check against the official download is still open.
- Audio: all training audio comes from mp3quran.net Qaloon recordings.

| Sheikh | Recording | mp3quran reader | Role |
|---|---|---|---|
| Ali Al-Huthaifi (علي الحذيفي) | Qaloon an Nafi | 75 | Training, 569 clips |
| Mahmoud Khalil Al-Husary (محمود خليل الحصري) | Qaloon an Nafi | 270 | Training, 564 clips |
| Al-Dokali Muhammad Al-Alim (الدكالي محمد العالم) | Qaloon an Nafi | 208 | Training, 542 clips |
| Waleed (nquran.com) | Qaloon | - | Test only, never trained on |

The dataset cards mark Huthaify, Husary and Dokali as approved for team training, and Waleed and Trabulsi as unauthorized for training.
Waleed is therefore used only as the unseen-voice test.

## Step 1: benchmark of available models

Five public Quran Whisper models were compared with the team's model on the team's exact test split (232 clips = 58 ayahs x 4 reciters, `qaloon_data.load_splits`, seed 42), with the project's normalizer and blind greedy decoding.
The harness reproduces the published validation WER of the team model exactly (28.74%).

| Model | Size | Test WER | Spelling-insensitive WER | 28 s streaming window |
|---|---|---|---|---|
| tarteel-ai/whisper-base-ar-quran | base | 3.97% | 1.59% | 1.6 s (writes diacritics) |
| naazimsnh02/whisper-large-v3-turbo-ar-quran | turbo | 4.46% | 2.08% | 1.3 s |
| deepdml/whisper-base-ar-quran-mix-norm | base | 4.76% | 2.38% | 0.66 s |
| basharalrfooh/whisper-small-quran | small | 5.75% | 3.67% | 2.6 s |
| MaddoggProduction/whisper-l-v3-turbo-quran-lora-dataset-mix | turbo | 11.21% | 8.43% | 1.1 s |
| Team model (Mathani-Ayat/qalon-reciter, full fine-tune) | base | 26.88% | 22.52% | 0.65 s |

Spelling-insensitive WER ignores orthography conventions such as بهاذا/بهذا and ءامنوا/امنوا.
All public models are Apache-2.0, except basharalrfooh (MIT).

## Step 2: Qaloon-specific word test

The KFGQPC Qaloon v10 and Hafs v18 texts were aligned word by word over al-Fatiha and surahs 78-114.
They differ in 22 places.
Nine are differences in the recited word that plain-text ASR can judge:

| Ayah (Qaloon numbering) | Qaloon | Hafs |
|---|---|---|
| 1:3 | ملك | مالك |
| 79:11 | إذا | أءذا |
| 80:22 | شا | شاء |
| 83:31 | فاكهين | فكهين |
| 89:20 | تحضون | تحاضون |
| 90:20, 104:8 | موصدة | مؤصدة |
| 91:15 | فلا | ولا |
| 112:4 | كفؤا | كفوا |

Six are tashil of hamza (79:10, 79:27, 96:9, 96:11, 96:13, 107:1), a pronunciation quality that plain text cannot judge.
Seven are spelling-only.
This classification is a draft and must be confirmed by a qualified Qaloon teacher.

Each of the 9 positions was tested on all four reciters.
A clip is scored only if the rest of the ayah is recognised, otherwise it is reported as unusable.

| Model | Writes the Qaloon form |
|---|---|
| deepdml base | 64% |
| tarteel base | 62% |
| naazim turbo | 68% |
| bashar small | 47% |

Every public model writes شاء (Hafs) for Qaloon's شا at 80:22, for every reciter.

## Step 3: data problems found

1. **Waleed's al-Fatiha labels are shifted by one ayah.**
   `001001.wav` contains الحمد لله رب العالمين but is labelled as the basmalah, `001002.wav` contains الرحمن الرحيم but is labelled الحمد لله, and so on.
   `001007.wav` contains only غير المغضوب عليهم ولا الضالين but is labelled as ayahs 6 and 7 together.
   The previous team model trained on these labels.
2. **Dokali clips contain audio from neighbouring ayahs.**
   A label audit (three strong models must independently see extra or missing words at a clip edge) flagged 27 Dokali clips and 1 Husary clip.
   They were excluded from training and are listed for review in `docs/generated/qaloon_label_audit_flagged.json`.
3. **The text normalizer corrupts some Quran words.**
   In `normalize_quran_for_asr`, a rule written for a ya that only carries an alif (أَتَيٰكَ -> اتاك) also fires when the ya has its own fatha.
   Across the Qaloon text it changes 840 words in 787 ayahs, for example القيامة -> القامة, ءايات -> ءاات, الشياطين -> الشاطين, ديارهم -> دارهم, and it drops the vocative يا (يَٰٓأَيُّهَا -> اايها).
   All stored training labels come from this normalizer, so 11 ayahs per reciter in our scope have wrong labels.
   The fix (only touch a ya without a vowel) is in `docs/patches/qaloon_model_v1.patch`; it has not been applied to the repository.
   The model was trained on labels re-derived with the fix.

## Step 4: training experiments

| Experiment | Result |
|---|---|
| Start from Tarteel base, full fine-tune | Stopped after one epoch: validation WER went from 1.95% to 47%. Tarteel writes full diacritics and our labels do not; switching format garbled words (نومكم -> نوماكم). |
| A: start from deepdml base, full fine-tune | Chosen recipe. |
| B: start from deepdml base, encoder frozen | Worse than A on every check. |

On Waleed (unseen voice, 562 clips, al-Fatiha excluded because of its shifted labels), strict WER / spelling-insensitive WER:

| Model | Ayah text seen in training (457) | Ayah text unseen (105) | Qaloon words | Invented skipped ayah |
|---|---|---|---|---|
| deepdml zero-shot | 4.91% / 2.88% | 2.05% / 0.68% | 64% | 0.0% |
| tarteel zero-shot | 4.43% / 2.40% | 2.05% / 0.68% | 62% | 0.5% |
| A: full fine-tune | 2.93% / 2.72% | 2.50% / 1.59% | 100% | 0.0% |
| B: decoder only | 3.42% / 3.04% | 4.09% / 3.18% | 100% | 0.0% |

The fine-tune is best on ayahs whose text it trained on, which covers every ayah the studio supports.
On ayahs it never saw it is slightly worse than zero-shot, so surahs outside al-Fatiha and 78-114 are better served by a zero-shot model.

The Trainer's in-training WER (bf16, no cache) disagreed with fp16 server-style decoding (26% vs 11% on one checkpoint), so all reported numbers use the benchmark script.

## Step 5: the release model

Recipe A, retrained on every ayah of the three approved reciters so the whole supported scope is seen text.

- Base model: `deepdml/whisper-base-ar-quran-mix-norm` (Apache-2.0).
- Data: Huthaify, Husary and Dokali, al-Fatiha and surahs 78-114, 1,675 clips after excluding the 28 flagged clips and 4 Husary clips longer than 30 s.
- Labels: `text_asr_normalized`, re-derived from `text_raw_uthmani` with the normalizer fix.
- Full fine-tune, learning rate 1e-5, cosine schedule, 3 epochs, effective batch 16, bf16, SpecAugment time masking 0.05, existing noise and speed augmentation, Dokali sampling weight 0.35.
- Training time: 31 minutes on the RTX 4060 Laptop GPU.

| Check | deepdml zero-shot | rattil_qaloon_v1 |
|---|---|---|
| Waleed WER, 562 clips (strict / spelling-insensitive) | 4.36% / 2.46% | **2.64% / 2.51%** |
| Qaloon word test | 64% | **100%** |
| Invented a skipped ayah (200 probes) | 0.0% | **0.0%** |
| Finished an ayah cut at 60% (144 probes) | 1.4% | 2.8% |
| 28 s streaming window | 492 ms | 507 ms |

All four early-stop cases were inspected.
In each, the final word was spoken before the cut: two Huthaify clips end in silence (the zero-shot model "fails" them too) and two Dokali clips are shifted by the boundary problem.
Dokali 109:2 (a 2.7 s clip) should still be confirmed by listening.

### End-to-end test in the app

Waleed's al-Fatiha was streamed at real speed through the studio's WebSocket pipeline (`src/streaming/replay_audio.py`) with ayah 3 deliberately skipped.
Ayahs 1, 2, 4, 5 and 7 were matched at 100%, ayah 3 was correctly reported missed (ملك يوم الدين), and ayah 6 was matched at 75%.
The unconfirmed word in ayah 6 (صراط) is the server's deliberate rule of holding the first word after an ayah change until a second word confirms it.

## What this model does not do

- It does not judge tashil, madd length, ghunnah or other pronunciation quality; it only recognises which words were recited.
- The Qaloon/Hafs classification above has not been reviewed by a qualified Qaloon teacher.
- The unseen-voice test uses one professional reciter; learner voices on laptop microphones are still untested.

## Running the app with this model

The Quran cache under `web/public/quran` is not in Git and must be built once.
`cache_quran_pages.py` downloads a Hafs numbering reference from `risan/quran-json`, whose URL no longer exists.
For this run it was replaced by the KFGQPC Hafs v18 text in the same shape (see `docs/patches/qaloon_model_v1.patch`); `hafs-reference.json` is cached afterwards.

```powershell
python src/dataset_collection/cache_quran_pages.py --trained-only --skip-metadata
$env:RECITER_MODEL_PATH = "D:/islamicaich/runs/rattil_qaloon_v1"
python -m uvicorn src.streaming.server:app --host 127.0.0.1 --port 8000 --ws-max-size 16384
cd web
npm run dev
```

`/health` on port 8000 reports `"model": "rattil_qaloon_v1"` when the new model is loaded.

## Files

| File | Purpose |
|---|---|
| `src/training_with_gpu/benchmark_models.py` | WER, spelling-insensitive WER, streaming latency and VRAM per model, on the team split |
| `src/training_with_gpu/benchmark_qaloon_words.py` | The Qaloon-specific word test |
| `src/training_with_gpu/benchmark_autocorrect.py` | Skipped-ayah and early-stop probes |
| `src/training_with_gpu/audit_labels.py` | Flags clips whose audio and label disagree at the edges |
| `docs/patches/qaloon_model_v1.patch` | Normalizer fix and training options (`--base-model`, `--include-reciter`, `--exclude-clips`, `--select-by wer`, `--train-on-all`, ...); not applied |
| `docs/patches/test_normalize.py` | Tests for the normalizer fix; move to `src/dataset_collection/` when applying the patch |

## Recommendations for the team

1. Decide on the normalizer fix; until it is applied, stored labels and the studio's cached text keep the corrupted words.
2. Fix Waleed's al-Fatiha labels and review the flagged Dokali clips.
3. Serve `rattil_qaloon_v1` for al-Fatiha and surahs 78-114, and a zero-shot model for other surahs.
4. The studio's tafsir and translations come from api.quran-tafseer.com, which is not in the competition's approved list; the pack approves quranenc.com for translations and dorar.net for tafsir.
5. Record a few team members on laptop microphones to test learner voices before the demo.
