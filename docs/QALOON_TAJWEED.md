# Qālūn tajweed: rules, colours, riwāyah topics and the tajweed model

Everything here is for **Qālūn ʿan Nāfiʿ, ṭarīq al-Shāṭibiyyah**. Rules and page numbers come
from the Libyan Awqaf curriculum «المنهج العلمي في أحكام التجويد وأصول رواية الإمام قالون»
(الهيئة العامة للأوقاف والشؤون الإسلامية، 2nd ed. 1443/2022, ISBN 978-9959-58-025-2):
chapter 2 for general tajweed and letter phonetics, chapter 3 for the riwāyah topics.
Every colour is applied by software from the written text. A qualified Qālūn teacher has the
final word.

| Piece | Where |
| --- | --- |
| Rule engine | `src/tajweed/` (`text.py`, `rules.py`, `engine.py`, `hafs.py`) |
| Reader data | `python -m src.tajweed.build` → `web/public/quran/tajweed/` (built on first app start) |
| Riwāyah topics | `src/tajweed/topics.py`, shown on `/tajweed` |
| Tajweed-model labels | `python -m src.tajweed.targets` → `data/tajweed/labels.jsonl`, `src/tajweed/model_tokens.txt` |
| Ahkam feedback | `src/tajweed/feedback.py` (called by the inference server) |
| Speaker-robust training | [`SPEAKER_ROBUST_TRAINING.md`](SPEAKER_ROBUST_TRAINING.md) |

## The engine

It reads the real Qālūn text (KFGQPC encoding in `web/public/quran/surahs/*.json`), not the
simplified `displayText`, because the marks carry the riwāyah:

| Mark | Meaning in this text |
| --- | --- |
| ۬ ۪ ۟ on a word-initial alif | hamzat al-waṣl (silent when joining). After a hamza-final word and not an article, the changed second of two hamzas (يَشَآءُ اِ۪لَىٰ) |
| ۬ on a mid-word alif or with a dagger alif | tas-hīl (ءَٰا۬نذرتهم, أَرَٰ۬يْتَ, هَٰا۬نتم) |
| ۟ inside or at the end of a word | tas-hīl (هَٰؤُلَآ۟ إِن, أَٰ۟ذَا) |
| ۬ on سـ / on ع ه خ م | ishmām (سِيءَ) / ikhtilās (لا تعْدّوا، يهْدّي، يخْصّمون، تأمنّا) |
| ۪ on ه / on ي / on ع | imāla (هارٍ) / tas-hīl without idkhāl (أئمة) / ikhtilās (نعِمّا) |
| ۢ | iqlāb |
| ۥ ۦ ۧ | ṣilah of the pronoun hā, extra yāʾ, small letters |
| ٓ | madd farʿī as written by the muṣḥaf |
| U+0657/065E/0656 | open (staggered) tanwīn, displayed as U+08F0–08F2 so Amiri draws it |

Each word is split into clusters (a letter plus all its marks). Silent letters are classified
first (waṣl alifs, sun-letter lām, اْ, وْ after ḍamma, tanwīn seats), then long vowels and līn,
then every rule looks through silent letters to the next pronounced one. Context is **waṣl
inside the ayah and waqf at its end**, which is how the app plays and checks one ayah at a time.

Rules (`src/tajweed/rules.py`, with book pages):

- **Madd** (pp. 58–66, Qālūn pp. 90–92): lāzim kalimī/ḥarfī 6, ʿayn of كهيعص/عسق 6 or 4,
  muttaṣil 4 (4 or 6 when stopping on the hamza), munfaṣil 2 or 4 (also yā of calling, hā of
  attention, ṣilah before a hamza), madd before an eased hamza 4 or 2, isqāṭ 2 or 4, badal 2,
  ʿāriḍ and līn 2/4/6, ʿiwaḍ, ṣilah 2, natural 2, and long vowels dropped before a sākin.
- **Nūn and mīm** (pp. 46–53): iẓhār, idghām with/without ghunna, iqlāb, ikhfāʾ, iẓhār muṭlaq,
  ghunna of نّ مّ, lip ikhfāʾ/idghām/iẓhār.
- **Letters** (pp. 41–57): qalqala (sākin and at the stop), rāʾ tafkhīm/tarqīq with the
  book's exceptions (ʿāriḍ or separated kasra, isti'lāʾ after it, فِرق both ways, مصر/القطر at
  the stop), lām of Allāh, sun/moon lām, hamzat al-waṣl, silent letters, idghām of mithlayn,
  mutajānisayn and mutaqāribayn.
- **Qālūn uṣūl** (chapter 3): mīm al-jamʿ, the eight short pronoun-hā words and يأتِه, two hamzas
  in one word (tas-hīl with/without idkhāl) and across words (isqāṭ, tas-hīl, ibdāl by vowel
  pair, بالسوء إلا, النبيء إن), single hamza (ibdāl, ḥadhf, naql), imāla, taqlīl, ishmām,
  ikhtilās, sākin hā in وَهْوَ/فَهْيَ, idghām/iẓhār choices, no sakt, yāʾāt al-iḍāfa and zawāʾid.

`hafs.py` aligns each surah word by word with the Ḥafṣ reference (`hafs-reference.json`) and
classifies the differences: isqāṭ, yāʾ al-iḍāfa, yāʾ zāʾida, and any remaining farsh word
("Ḥafṣ reads: …", ~800 words such as مَلِكِ، يُخادعون، فَمَنُ اضطُرّ). Spelling-only differences
(Maghribi vs Mashriqi rasm) are normalised away.

The whole Quran annotates in about 3 seconds; `python -m src.tajweed.build` (with the Ḥafṣ
comparison) takes about 7. Tests: `python -m unittest src.tajweed.test_engine src.tajweed.test_targets`,
including a whole-Quran check that every ayah annotates, keeps the reader's word count and
rebuilds every word exactly.

## In the app

- **Reader text**: the full Qālūn spelling from the generated files (ṣilah, iqlāb mīm, tas-hīl
  dots). The files are static (one per surah, 4.6 MB in total, al-Baqarah 43 KB gzipped)
  instead of the old 21.5 MB `qalon_majwad_mushaf.json` served through an API route.
- **Colours** (toggle "Tajweed colors"): necessary madd, connected madd, permitted madd,
  natural madd, ghunna, merged/silent, qalqala, heavy letters and Qālūn riwāyah points (purple
  plus a dotted underline). Contrast is at least 4.2:1 on the paper and 3.3:1 on every word
  status background. The legend switches groups on and off; "every detail" adds natural madd,
  light letters, iẓhār, isti'lāʾ heaviness and mīm al-jamʿ. Settings are kept per browser.
- **Tap a coloured letter** for its rule, Qālūn count, ways, any note (for example "Ḥafṣ
  reads: …" or "two hamzas in one word: … idkhāl") and book page.
- **Check my ahkam** asks the server for the tajweed model. Missed rules get a wavy underline
  and the session card counts applied vs expected rules. Without the model installed the
  session checks words only and says so.
- **`/tajweed`**: the 17 makhārij and the ṣifāt of every letter, every rule with its Qālūn count,
  occurrences and coloured examples, and the 14 riwāyah topics with every occurrence linked to
  `/studio?surah=S&ayah=A`.

Colour spans only change `color` on inline elements inside a word, never split a letter from
its marks, lām-alif or the Allāh ligature, and never insert joiners. The Playwright spec
`tests/reader-controls.spec.ts` checks that word text, font runs and widths are identical with
colours on and off.

## The tajweed model

The plain model writes words. The tajweed model writes the same words, each followed by one tag
token per audible rule on it, for example
`ان<tj:ghunna> الذين كفروا سواء<tj:madd_muttasil> عليهم اانذرتهم<tj:tasheel><tj:ikhfa> …`.
Stripping the tags gives exactly the plain target, so the server matches words the same way and
reads the tags separately. The 17 tags (`src/tajweed/model_tokens.txt`) cover ghunna, ikhfāʾ,
iqlāb, the idghāms, qalqala, madd lāzim and muttaṣil, ṣilah, tas-hīl, imāla, ibdāl, isqāṭ and
naql. Optional ways (munfaṣil length, mīm al-jamʿ ṣilah) and spelling notes get no tag.

```bash
# 1. labels (one per ayah, plus clip-specific ones for multi-ayah clips)
python -m src.tajweed.targets --reader-root data/hf/qaloon-reciter-dataset --reader-root data/hf/qaloon-new-reciters

# 2. train from the production model (training PC)
python src/training_with_gpu/train_base_full.py --init-model runs/rattil_qaloon_v3 \
  --label-field text_tajweed --labels-jsonl data/tajweed/labels.jsonl --extra-tokens src/tajweed/model_tokens.txt \
  --reader-root data/hf/qaloon-reciter-dataset --reader-root data/hf/qaloon-new-reciters \
  --reciters huthaify husary dokali abusnaina akri daawob deeban kshidan qeniwa \
  --augment-profile speaker-robust --learning-rate 5e-5 --epochs 10 --patience 3 \
  --output-dir runs/rattil_qaloon_tajweed_v1

# 3. serve: the default command already offers it once the folder exists; ahkam
#    sessions take words from the plain model and tags from this one
python -m src.streaming.serve --model rattil-v3
```

Checkpoints are selected by `tajweed_score` = tag F1 − macro WER (`--select-by`), so a run
keeps the epoch with the best tags without accepting worse word recognition. Continuing a
tagged run (`--init-model` a tajweed checkpoint) reuses its tag embeddings.

### Results so far (proof of concept on this Mac)

Only three readers are on this Mac (Huthaify, Husary, Dokali: 1,388 training clips, Juz ʿAmma
and al-Fātiḥah), trained on MPS from `rattil_qaloon_v3`:

| Run | Tag precision | Tag recall | Tag F1 | Word error |
| --- | --- | --- | --- | --- |
| v1: lr 2e-5, 5 epochs (validation) | 0.52 | 0.40 | 0.45 | 3.2% |
| v2: continued at lr 5e-5, 8 epochs, best of 8 by `tajweed_score` (test) | 0.94 | 0.69 | 0.80 | 10.3% |
| **Ahkam mode: plain model's words + v2's tags (test, 170 clips)** | **0.94** | **0.66** | | **2.3%** |

v2 is installed as `runs/rattil_qaloon_tajweed_v1`, so `python -m src.streaming.serve --model rattil-v3`
serves both. The higher learning rate that taught the tags cost the tajweed model word accuracy
(محفوظ heard as "محسن"), so in ahkam mode the server runs both models (`PairedEngine` in
`src/streaming/server.py`): words and matching come from the plain model, and the tajweed model's
tags are aligned onto those words (`transfer_tags`). Both decodes take about 0.2 s per 9 s of audio
on the Mac CPU.

These numbers are on unseen ayahs of the same three voices, and v3 has already heard all of
them, so word error rate is optimistic. Tag F1 is the meaningful number.

### What the model can and cannot judge yet

All training audio is correct recitation by sheikhs, so the model has never heard a missed
ghunna or a short madd. It learns where rules belong and some of how they sound, so "missed"
should be read as "the model did not hear it", not as a verdict. To make it a real judge:

1. Train on all ten readers on the training PC (above), with speaker-robust augmentation.
2. Add negatives. Learners from Ḥafṣ are the most common Qālūn mistake, so record or collect
   Ḥafṣ recitations of the same ayahs and label them with Ḥafṣ realisations (no tas-hīl, full
   hamza, Ḥafṣ farsh). The aligner in `hafs.py` already finds every place the two differ.
3. Duration negatives: with a forced alignment of letters, shorten madd and ghunna segments by
   time-stretching and drop their tags, so length is learned from audio.
4. Learner recordings reviewed by a teacher (`src/training_with_tajweed/make_annotation_manifest.py`).

## Known limits

- One ayah at a time: no internal waqf, no joining across ayahs, waqf always at the ayah end.
- Rawm and ishmām at the stop, and the basmalah/istiʿādhah joining ways, are explained on
  `/tajweed`, not coloured.
- The source text has a stray sukūn in قَالْ أَرَٰ۬يْتَكَ (17:62), which the engine reads as written.
- Isqāṭ and the yāʾāt need `hafs-reference.json` (cached by `cache_quran_pages.py`); without it
  those points are not marked.
