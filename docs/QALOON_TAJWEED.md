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

The plain model writes words. The tajweed model writes the same words, each followed by one token
per audible rule on it, for example
`ان<n_ghunna> الذين كفروا سواء<mad> عليهم اانذرتهم<tasheel><n_ikhfa> …`.
Stripping the tokens gives exactly the plain target, so the server matches words the same way and
reads the tokens separately.

### Tokens

The tokens follow the acoustic-driven design of the "Evaluating Automatic Speech Recognition" report
(Table 4.6, Ḥafṣ): rules that sound alike share one token, rules without an acoustic marker get none.
They are adapted to Qālūn:

| Token | Rules (engine ids) | Qālūn note |
| --- | --- | --- |
| `<n_ikhfa>` | ikhfāʾ of nūn/tanwīn | |
| `<m_ikhfa>` | iqlāb, lip ikhfāʾ | same labial ghunna |
| `<n_ghunna>` | nūn mushaddada, incl. full idghām into nūn | token on the doubled letter |
| `<m_ghunna>` | mīm mushaddada, incl. idghām of nūn/mīm into mīm | token on the doubled letter |
| `<idgham_ghunna>` | idghām into ي و | |
| `<qalqala>` | qalqala (sākin and at the stop) | |
| `<mad>` | madd lāzim (6), muttaṣil (4), opening letters | **not** munfaṣil: Qālūn reads it 2 or 4 |
| `<tasheel>` | tas-hīl of a hamza | Qālūn |
| `<silah>` | ṣilah of the pronoun hā | |

Idghām without ghunna, iẓhār and the optional ways (munfaṣil length, mīm al-jamʿ ṣilah) get no token.
The server still reads the first model's `<tj:name>` tags and renames them.

### Negatives: plain reading

A model trained only on sheikhs who apply every rule learns where rules belong from the text and
then "hears" them everywhere. The report measured a 115% tajweed error rate on plain audio, and
mixing in about 30% plain synthetic readings brought it to about 0. `src/tajweed/prepare_data.py` does
the same with Meta's MMS Arabic TTS (`facebook/mms-tts-ara`, CC-BY-NC-4.0): three takes per ayah
(different seeds and speaking rates), kept only when rattil-v4 reads the letters (character
error ≤ 0.25; v4 has only heard sheikhs, so its word error on flat TTS speech is high even when
the letters are right). 973 of 1,695 takes were kept. They get targets with no tokens.

### Training v2 (10 readers + negatives, from v4)

```bash
# data: the ten approved readers (unapproved/ and audit_flagged clips are left out) and Waleed for testing
python src/deployment/pull_hf_assets.py --models v4 --datasets qaloon-all-reciters qaloon-reciter-experiments
PYTHONPATH=src/deployment/mac_shim python -m src.tajweed.prepare_data --reader-root data/hf/qaloon-all-reciters \
  --waleed data/hf/qaloon-reciter-experiments/dataset_qaloon_waleed --out data/tajweed
python -m src.tajweed.targets --reader-root data/hf/qaloon-all-reciters --reader-root data/tajweed --out data/tajweed/labels.jsonl

python src/training_with_gpu/train_base_full.py --init-model runs/rattil_qaloon_v4 \
  --label-field text_tajweed --labels-jsonl data/tajweed/labels.jsonl --extra-tokens src/tajweed/model_tokens.txt \
  --reader-root data/hf/qaloon-all-reciters --reader-root data/tajweed \
  --reciters huthaify husary dokali garu daawob abusnaina qeniwa akri deeban kshidan ttsplain waleed \
  --holdout-reciter waleed --augment-profile speaker-robust --learning-rate 5e-5 --epochs 5 --patience 2 \
  --output-dir runs/rattil_qaloon_tajweed_v2

# serve: words from v4, tokens from the newest tajweed model (v2 once its folder exists)
python -m src.streaming.serve --model rattil-v4 --beams 3
```

Checkpoints are selected by `tajweed_score` = tag F1 − macro WER (`--select-by`), so a run
keeps the epoch with the best tags without accepting worse word recognition. Each clip's
errors are capped at its word count (`capped_macro_wer`): in the first v2 run one clip that
repeated a word 73 times pushed epoch 3's WER to 59% and the run kept epoch 1, although epoch 3
was the better ahkam model (below). Continuing a tagged run (`--init-model` a tajweed
checkpoint) reuses its tag embeddings.

### How to test

1. **Ahkam mode end to end** (`src/tajweed/evaluate.py`): words from the plain model, tokens from the
   tajweed model aligned onto them, exactly as the server does it.

   ```bash
   PYTHONPATH=src/deployment/mac_shim python -m src.tajweed.evaluate --plain runs/rattil_qaloon_v4 \
     --tajweed runs/rattil_qaloon_tajweed_v2 --reader-root data/tajweed --reciters waleed ttsplain --device mps
   ```

   - **waleed**: a voice never trained on, reciting correctly. `token_recall` = expected rules
     heard, `token_precision` = heard tokens that belong there.
   - **ttsplain**: plain TTS readings of ayahs outside the training split. `false_alarms_per_word`
     = tokens the model invents where no rule was applied. Lower is better.
   - **An unseen plain voice.** The TTS negatives are one voice, so a model could learn "this voice
     has no tajweed". Read non-train ayahs with another voice and score it the same way, e.g. the
     macOS Arabic voice: `say -v Majed -o 083017.wav --data-format=LEI16@16000 "<text_asr_normalized>"`
     in a `dataset_qaloon_majed` folder with `"tajweed": false` rows, then `--reciters majed`.
2. **In the app**: start the server, open `/studio`, switch on "Check my ahkam" and recite, or
   upload a clip. The "What to fix" card lists each missed rule with the word and how to fix it;
   click one to jump to the word. Test both a careful recitation and a deliberately plain one:
   the plain one should list missed ghunna, madd and qalqala.
3. **Code**: `python -m unittest src.tajweed.test_engine src.tajweed.test_targets src.streaming.test_tajweed_tags`.

### Results

**v2** (`runs/rattil_qaloon_tajweed_v2`, published privately as
[`Mathani-Ayat/rattil-qaloon-tajweed-v2`](https://huggingface.co/Mathani-Ayat/rattil-qaloon-tajweed-v2);
`python src/deployment/pull_hf_assets.py --models tajweed-v2`): ten readers (5,036 training clips) plus
the plain TTS negatives, from rattil-v4, speaker-robust augmentation, lr 5e-5. Early stopping ended
the run after epoch 3. Ahkam mode (v4 words + tajweed tokens), greedy decoding:

| Tajweed model | Waleed precision | Waleed recall | False alarms / word, TTS plain | False alarms / word, Majed plain (unseen voice) |
| --- | --- | --- | --- | --- |
| v1 (3 readers, no negatives) | 0.83 | 0.84 | 0.130 | 0.147 |
| v2, epoch 1 (kept by the uncapped score) | 0.93 | 0.83 | 0.006 | 0.045 |
| **v2, epoch 3 (installed)** | **0.99** | **0.94** | **0.006** | **0.019** |

Waleed: 562 clips; TTS: 182 takes; Majed: 104 ayahs. Word error in ahkam mode is v4's: 2.1% on
Waleed. The tajweed model alone at epoch 3 had 59% validation WER because of a few looping clips,
which ahkam mode does not use. On the unseen plain voice the negatives cut invented tokens 7.7×
relative to v1; on the TTS voice they were trained with, the cut is 21×, so part of that effect is
the voice. Over-generation is reduced but not gone.

**v1** (proof of concept, three readers on this Mac, from rattil-v3): 13 epochs in two runs. Tag F1
rose from 0.45 (5 epochs at lr 2e-5) to 0.80 (8 more at 5e-5), while the tajweed model's own word
error grew from 3% to 10%. That forgetting is why ahkam mode takes words from the plain model
(`PairedEngine` in `src/streaming/server.py`, `transfer_tags`). Both decodes take about 0.2 s per
9 s of audio on the Mac CPU.

### What the model can and cannot judge yet

All training audio is correct recitation by sheikhs, so the model has never heard a missed
ghunna or a short madd. It learns where rules belong and some of how they sound, so "missed"
should be read as "the model did not hear it", not as a verdict. To make it a real judge:

1. Done in v2: all ten readers with speaker-robust augmentation, and plain TTS negatives.
2. Ḥafṣ negatives. Learners from Ḥafṣ are the most common Qālūn mistake, so record or collect
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
