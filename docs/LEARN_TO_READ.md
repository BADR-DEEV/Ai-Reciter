# Learn to read: beginner track

A course for people who can't read Arabic yet: people curious about Islam, new
Muslims, and anyone who wants to recite properly. It goes from the alphabet to
reciting short surahs and needs no vocabulary. The interface is English by
default, with an Arabic toggle (العربية) in the header.

| Route | What it is |
| --- | --- |
| `/` | Landing page. The hero is the alphabet itself: tap a letter to hear it. |
| `/learn` | Course map with progress, XP and a daily streak (stored in the browser). |
| `/learn/<lesson>` | Lesson player. |
| `/studio` | The existing recitation studio (moved from `/`). Accepts `?surah=N`. |

## Curriculum (5 units, 22 lessons, about 270 steps)

1. **Start here.** What the Quran is, right-to-left reading, how the listening
   works and its limits, and what a riwāyah (Qālūn) is.
2. **The letters.** 7 lessons by shape family. Each letter card shows its sound,
   an English comparison, where it is made, heavy or light, its four joined
   forms, and a pronunciation tip. Then listen-and-pick, see-and-pick, and say-it.
3. **Sounds English doesn't have.** Minimal pairs: ح/ه, ع/ء, خ/غ, ص/س, ط/ت,
   ق/ك, ض/د, ث/س, ذ/ز, ظ/ذ. This is where the model helps most.
4. **Reading syllables.** Joining, the three short vowels, sukūn and qalqalah,
   long vowels (madd), shaddah, ghunnah and tanwīn, the word Allāh, and sun
   and moon letters.
5. **Your first surahs.** Al-Fātiḥah, Al-Ikhlāṣ, Al-Kawthar, Al-Falaq and
   An-Nās. For each ayah: Al-Husary's Qālūn audio (MP3Quran reader 270 with its
   own timings), transliteration, meaning, and read-aloud. Each surah ends by
   handing off to the studio.

The content lives in `web/lib/learn/` (`letters.ts`, `curriculum.ts`,
`surahs.ts`). Lessons are generated from data, so adding a letter drill or a
surah means adding data, not components. A wrongly answered question is
repeated once at the end of the lesson.

## How speaking is checked (`POST /api/practice`)

The lessons use the same model and server as the studio
(`src/streaming/server.py`, `src/streaming/practice.py`).

- **`sound` mode** (letters, minimal pairs, long vowels): the model scores the
  log-likelihood of each candidate, the target plus its confusables (for
  example `حا` vs `ها`), and picks the most likely. The target is always one
  option among several, so the answer is not leaked to the model. The model
  also produces a blind transcript. If that transcript spells a different
  candidate, the verdict is downgraded from *correct* to *close*.
- **`reading` mode** (words and ayahs): a blind transcript is aligned word by
  word, using the studio's fuzzy matcher.

Practice syllables use a long "ā" (`حَا`, not `حَ`), because Whisper
transcribes a held syllable much more reliably than a bare consonant.

**What it can't do:** the model was trained on *unvowelled* text, so it cannot
tell `بَ` from `بِ` or judge madd length, ghunnah or qalqalah. Those lessons
teach by listening and choosing, and ask for speech only where letters differ.
The UI says this plainly: it is a practice partner, not a tajweed certificate.

Letter and syllable sounds use the browser's Arabic text-to-speech voice. That
is a stopgap for single sounds only. Ayahs always use a human reciter.

## Running it

```bash
# Backend: the trained model, as before
python -m uvicorn src.streaming.server:app --host 127.0.0.1 --port 8000

# Without runs/gpu_base_full (e.g. a laptop), for UI development only:
RECITER_MODEL_PATH=openai/whisper-base python -m uvicorn src.streaming.server:app --port 8000

cd web && npm install && npm run dev   # http://127.0.0.1:3000
```

The server now starts without the Mushaf cache, because lessons don't need it
(the studio still does). If the backend is offline, lessons still work and
speaking steps show a **Skip** button.

Tests:

```bash
python -m unittest src.streaming.test_practice -v
cd web && npx playwright test tests/learn.spec.ts tests/learn-speaking.spec.ts
```

## Recommended next steps for the model (in priority order)

1. **Record a real reference set.** Have a qualified teacher record all 29
   letters with each harakah, the long vowels, and the word list. This replaces
   browser TTS (inconsistent, wrong on ḍ/ẓ/ʿ) and gives the scorer a clean
   per-sound test set. It is cheap and has the biggest impact on learners.
2. **Measure the scorer before trusting it.** Collect consented learner
   attempts (correct and deliberately wrong) for each minimal pair, have a
   teacher label them, and report per-pair accuracy and false-reject rate
   using speakers that were never in training. Tune the 0.6 confidence
   threshold on validation data only. Today's sanity check used synthetic macOS
   speech on vanilla `whisper-base` (4/6 pairs correct). That proves the
   plumbing works, not that the scores are accurate.
3. **Add a phoneme/character CTC head.** Train it on `normalized_with_harakat`
   (V2 data already exists), e.g. wav2vec2/w2v-BERT or the Whisper encoder
   plus CTC. A seq2seq decoder has a strong language-model prior and will
   "autocorrect" a learner's mistake into the Quran word it expects. CTC with
   forced alignment gives per-phoneme Goodness-of-Pronunciation scores, which is
   what letter, harakah and madd feedback needs. Keep Whisper for ayah-level
   recognition.
4. **Calibrate the forced choice against a prior.** Some spellings (e.g. `ها`)
   are common Arabic words, so Whisper favours them. Subtract each candidate's
   score on silence/noise, or train a small logistic calibrator on step 2's
   labelled attempts.
5. **Train on learner-like audio.** Today's training audio is studio
   recordings by four professional reciters. Learners are slow, hesitant, use
   laptop microphones, and some are children or have accents. Add opt-in lesson
   recordings (with explicit consent and a retention policy), plus noise and
   room augmentation. Evaluate on held-out *speakers*, not held-out ayahs.
6. **Use a closed vocabulary for ayahs.** Decode against the expected ayah's
   n-gram lattice (or rescore an N-best list) for the studio, and keep blind
   decoding as the omission check, as described in the training README.
