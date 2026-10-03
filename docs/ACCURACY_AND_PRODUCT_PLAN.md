# Rattil: what is built, what is missing, and what to improve next

Assessment date: 2026-10-03. No new training has been run for this update.
The product teaches **Quran reading and recall**, not general Arabic. Quran
text, reading aids, transcription and religious assessment are separate layers.

## Executive decision

**Keep the competition's supported recognition scope at Fātiḥah + Juz ʿAmma.
Prioritize clean labels and unfamiliar learner voices before full-Quran ASR.**
All 114 surahs remain available for reading/text challenges. Availability is
not a claim that the speech model can reliably recognize all of them.
More GPU optimization improves throughput, not evidence of pronunciation quality.

Order of investment:

1. Repair audio boundaries/labels and freeze honest evaluations.
2. Collect consented **new human readers**, including the foreign learners we serve.
3. Validate streaming on those readers, microphones and natural pauses.
4. Compare model/augmentation/decoding changes on the same validation sets.
5. Expand distinct Qālūn text coverage only after the restricted product works.
6. Develop a separately reviewed phoneme/tajweed assessment track.

## What was implemented in this update

| Area | Delivered | Important limit |
|---|---|---|
| Qālūn phonetics | Source-driven rule-based G2P; vowels, shadda, long vowels, article sun letters, basic pause forms, Allah/relative-pronoun/disjoint-letter rules | Draft orthographic aid, not a complete phonetic/tajweed transducer |
| Audit | Full cached-source CSV, pause/connected forms, mapped Hafs comparison, warnings and reviewer fields | **0 scholar-approved entries**; exact source script/route still needs verification |
| Studio | Opt-in reading aids, starting/resume ayah, connection heartbeat/status | No automatic religious judgment; no forced answer prompting |
| Streaming | Recover from high partial matches; keep a next-ayah audio prefix; promptly surface worker errors/completion | Window still capped at 28s; heuristic RMS VAD and ordered text matching remain |
| Challenges | Next ayah (blind ASR or choices), match three real audio clips, identify surah, restore word, order ayahs; difficulty and scope | Local prototype, not a secure exam; answer keys are inspectable |
| Audio similarity | 565 aligned local Al-Husary clips indexed by MFCC summary + duration | Real acoustic **heuristic**, not learned phonetic embeddings or tajweed scoring |
| Tafsir/meaning | Arabic tafsir and English translation from Quran Tafseer API; text-aligned verse-number mapping | English API books are translations of meanings, not an English scholarly tafsir |
| Profiles | Browser-local name/password profiles, salted PBKDF2 hashes, separate progress, sign-out/delete | Not real authentication; no cloud sync/recovery; never reuse a real password |
| Training/evaluation | Speaker-disjoint fresh-run option, split dry-run, error breakdown/cluster bootstrap, prevent speed augmentation truncation, save inference cache config | Does not retroactively make current weights speaker-independent |

No existing dataset text, audio labels or reciter timestamps were replaced.
No audio was synthesized. Model weights and the Hugging Face release were not
retrained or replaced by this product update.

## Your teammate is correct: fix the claim, not the definition of WER

The existing four-reciter split groups ayahs, but **all four voices occur in
training, validation and test**. Its proper label is:

> Known-voice / held-out-ayah Qālūn transcription on expert recordings.

It is **not** unseen-reader accuracy, novice-reading accuracy, omission recall,
harakat accuracy or tajweed accuracy. Keep WER and CER; add evaluation protocols
and task-specific metrics. Do not rename WER as a pronunciation score.

### Current saved results (same 232 test / 188 validation clips)

| Model | Test WER | Test CER | Validation WER | Validation CER |
|---|---:|---:|---:|---:|
| Full Whisper Base | 26.39% | 7.96% | 28.74% | 7.20% |
| Base LoRA v2 | 31.15% | 9.05% | 33.02% | 10.10% |

Recomputed from saved predictions, not new live tests. Fractions in JSON,
percentages here. Keep the project's normalization fixed and publish it.
The earlier Tiny/three-reciter split is not a valid comparison to this table.

Full-model test errors: **212 substitutions, 30 deletions, 24 insertions**
against 1,008 reference words. Improving substitutions/segmentation is more
important than tuning only deletion alerts. There are just **58 unique test
ayahs** repeated across four readers, not 232 independent content samples.

| Known test voice | Full-model WER | CER |
|---|---:|---:|
| Dokali | 32.54% | 11.72% |
| Husary | 27.78% | 7.52% |
| Huthaify | 20.63% | 5.30% |
| Waleed | 24.60% | 7.28% |

Descriptive 1,000-replicate **ayah-cluster** bootstrap gives approximately
20.34–33.06% WER and 5.99–10.25% CER for the full model. This interval describes
the saved split only. It is not an unseen-reader population estimate. Four expert
voices are far too few to claim robust speaker generalization.

Runner-up selection already used saved test results. Treat those tests as a
published historical benchmark, not an untouched future selection set.

### Maintain four distinct evaluations

1. **Known voices / unseen ayahs:** retain historical regression benchmark.
2. **Unseen voices / seen texts:** essential for a memorize-and-recite companion.
3. **Unseen voices / unseen texts:** strongest check before expanding coverage.
4. **Actual learner sessions:** unfamiliar readers, accents, devices, rooms,
   restarts, pauses, omissions, repeats, substitutions and prompted corrections.

Never split fragments from one recording/session across train/validation/test.
Detect duplicate recordings with hashes; near-duplicates need audio review.
Keep validation readers separate from test readers. Choose model, thresholds and
decoder settings on validation only, then run the frozen test once.

The fresh full-training CLI supports distinct validation/test reciters:

```powershell
python src/training_with_gpu/train_base_full.py --validation-reciter dokali --test-reciter waleed --output-dir runs/speaker_holdout_waleed --dry-run
```

Verified local dry-run: **924 training clips (Husary/Huthaify), 569 validation
(Dokali), 568 test (Waleed)**. Held-out readers use all their clips; training
readers retain the old training-ayah buckets. Reports separately include text
seen/unseen in the *actual training labels*. Basmalah/Fātiḥah merge conventions
must still be reviewed in speaker comparisons. Rotate the outer test reader for
a four-fold diagnostic; this is still expert-reader, not novice-reader evidence.

Remove `--dry-run` only when deliberately launching a **fresh** experiment. It
starts from `openai/whisper-base`, not the current four-reader fine-tune. A
holdout is invalid if those voices already trained the starting checkpoint.
Use a new output folder; existing model weights are protected from overwrite.

### What to report beyond WER/CER

- Corpus WER/CER with S/D/I counts and fixed normalization; per-reader, surah,
  ayah duration, accent/device/noise and seen/unseen text strata.
- Macro-average reader WER as well as corpus-weighted WER; reader-cluster
  bootstrap once there are enough independently recruited readers.
- Omission alert precision/recall/F1, false alarms per minute, detection delay,
  exact missing-word spans and skipped-ayah precision/recall.
- Repetition/substitution confusion matrices, coverage and **abstention rate**.
- Streaming median/p95 latency, real-time factor, queue delay, transition error
  rate, stalls per session and recovery time. Offline WER is not streaming WER.
- Beginner learning outcomes reviewed by teachers, not just XP/streaks.

The old "apparent omission ayah rate" on correct expert clips is a false-alarm
proxy. It cannot measure recall when no intentionally omitted words are labeled.
Collect labeled omission sessions; do not substitute ASR deletions for truth.

## Data expansion: more Quran or more reciters?

**Immediate answer: more clean, diverse learner voices within our small scope.**

Professional reciters share fluent cadence and clean recording setups. Learners
have different vowel lengths, microphones, pauses and restarts. Adding another
professional is useful, but does not replace learner-domain data. Recruit with
informed consent, track permission/version, and use qualified Qālūn reviewers.
Do not clone named reciters or collect children's recordings without safeguards.

Pilot target, not an accuracy guarantee: 20–30 independent adult readers across
fluency/accent/device conditions. For example, reserve 5 readers for validation
and 5 entirely untouched for test, with remaining readers for training. Increase
size after seeing variance; don't count each ayah as an independent new person.
Record some correct sessions and explicit, consented error scenarios. Store
wrong pronunciations/errors as **labeled errors**, not correct target examples.

Before broadening Quran coverage, fix:

- Dokali boundary bleed and neighboring-ayah content; listen before recutting.
- Long ayahs: verified text-aligned segments, not truncation with full labels.
- Basmalah, Qālūn numbering, merged ayahs and timestamp alignment.
- Script/normalization consistency; never replace Qālūn words with Hafs labels.
- Silence/noise/hallucination examples and interrupted/repeated learner phrases.

Then expand incrementally: additional short surahs/contexts with unfamiliar
words, freeze a new benchmark, validate, and only then advertise full-Quran ASR.
Avoid training only more copies of the same expert voices on the same 38 surahs.

## Training/GPU pipeline priorities

1. Preserve baseline predictions/split hashes and run one-variable ablations.
2. Fix augmentation overflow (implemented): varispeed can exceed Whisper's 30s
   feature window; reject it instead of silently losing labeled words. Current
   resampling changes pitch too; call it **varispeed**, not pitch-preserving tempo.
3. Cache unchanged validation features; profile CPU feature extraction, disk IO,
   RAM, dataloader wait, CUDA memory and GPU utilization before adding workers.
4. Use BF16 on supporting GPUs, FP16 otherwise; retain correct CUDA wheel.
   Batch/accumulation choices trade memory against throughput, not correctness.
5. Consider generation-based validation WER/checkpoint selection instead of only
   validation loss. This is still a remaining improvement, not implemented here.
6. Test LR/early stopping, rank/targets, full vs LoRA and mild train-only realistic
   noise independently. No augmentations on validation/test.
7. Profile inference with cache on, FP16 and greedy decoding; benchmark small
   beam widths on validation, including latency. No expected ayah decoder prompt
   when reporting blind WER. Optional constrained recognition must have a separate
   task label and an acoustic verification/abstention path.
8. Investigate CTranslate2/faster-whisper exports only after parity checks on the
   frozen audio benchmark; compare speed, memory and WER before switching.

This update sets `use_cache=True` before saving future full-model artifacts and
in the running engine. It does not promise a measured latency gain without a
profiling comparison. No retraining or conversion was launched.

## Why streaming used to get stuck, and remaining work

Code-level failure modes repaired:

- A high current-ayah partial match blocked lookahead even when the next ayah
  was strongly recognized. Recovery now permits ordered later-ayah evidence;
  an inaccurate terminal word alone still does not force progression.
- Advancing cleared an entire decoded window even when it included a prefix
  of the next ayah. Keep the full utterance until a quiet boundary, and retain
  a recognized next-ayah prefix. At a safe boundary otherwise keep all
  concurrently received PCM, without an arbitrary cropped tail.
- A real microphone regression exposed hallucinations from tiny terminal-word
  tails. Retained-window decoding now strips strongly aligned completed-text
  context before matching the next ayah. It does not inject expected text into
  Whisper or force an answer; partial transcripts can still vary.
- An isolated first word after a transition remains tentative until a second
  target-word anchor corroborates it (multiword ayahs). Keep its audio rather
  than losing a genuine prefix. This intentionally trades a small highlight
  delay for fewer hallucinated green words; calibrate it on real learners.
- The receive loop could wait 60s before surfacing a failed/finished worker.
  It now races the receive task with the worker, with clean cancellation.
- UI now pings every 5s, detects an unresponsive connection, displays decoding
  backlog and offers an explicit starting ayah. It does not invent a transcript
  or reconnect invisibly after losing audio. Stop has a longer final-decode budget.

Remaining: adaptive/neural VAD, robust local-agreement partial commits, token
timestamps and exact audio trimming, cumulative matching for >28s ayahs, bounded
per-client GPU fairness, trustworthy acoustic omission verification, learner
calibration and backpressure tests. The current RMS cutoff can miss quiet voices;
heartbeat establishes connection liveness, not acoustic correctness.
Resume creates a new session; previous results are not merged into a scored
continuous attempt. Do not mark untouched verses as omissions on resume.

Operational note: stop a Next.js development server before running `npm run
build` against the same `.next` directory. Mixing concurrent development and
production builds can serve mismatched client bundles. Run `npm start` after a
completed production build for the presentation; restart the backend after
Python streaming code changes.

## Tajweed, harakat and Qālūn-specific pronunciation

Unvowelled ASR normalization **removes the evidence needed to score many vowel
errors**. A recognizer can produce the expected word despite a wrong vowel or
mispronounced consonant. Latin transcription likewise cannot certify makhārij,
ṣifāt, nasalization, madd duration, qalqalah or accepted variants.

Separate the work into three products:

1. **Text tracking:** current blind ASR + transparent text agreement.
2. **Reviewed reading aids:** explicit Qālūn route, source spelling, wasl/waqf,
   per-word phoneme targets and curated human reference recordings.
3. **Acoustic assessment:** teacher-labeled learner phonemes and error spans,
   forced alignment/phoneme recognizer or CTC acoustic model; calibrated
   phone-level scores with abstention and held-out readers.

Specify the supported ṭarīq and permitted choices first. Qālūn mīm al-jam,
pronoun ṣilah, hamza combinations and madd choices are not recoverable simply
by transliterating Hafs letters. A diacritic-aware ASR experiment can report
harakat edit/error rates **separately**, but expected-text reconstruction is not
proof that the audio contained the right harakat. Evaluate phoneme errors,
teacher agreement, error-type recall and false accusations. Do not grade
valid variant choices as mistakes. Never infer religious quality from WER/XP.

See [QALOON_PHONETICS_AUDIT.md](QALOON_PHONETICS_AUDIT.md) for the review process.

## Reproduction and generated deliverables

```powershell
cd web
npm.cmd run phonetics:audit
npm.cmd run typecheck
npx.cmd playwright test
# Back at the repository root:
python src/learning/build_audio_similarity.py
python src/training_with_gpu/report_metrics.py --predictions runs/gpu_base_full/test_predictions.jsonl --output docs/generated/full-model-metrics.json
python src/training_with_gpu/report_metrics.py --predictions runs/gpu_base_v2/test_predictions.jsonl --output docs/generated/base-lora-v2-metrics.json
python -m unittest src.streaming.test_matcher src.streaming.test_buffer src.training.test_speaker_split -v
```

Generated files: `docs/generated/qaloon-phonetics-audit.csv`,
`docs/generated/phonetics-summary.json`, the two metric reports, and
`web/public/quran/audio-similarity.json`. Quran assets, recordings and CSV files
are Git-ignored; regenerate them locally. They are not silently fetched at build.
The acoustic index is served only when present; missing indexes are explicitly
described as text-similarity fallback, never as measured acoustic similarity.

## Before this becomes a public product

- Qualified Qālūn sign-off on the source, route, lesson audio and reading aids.
- Document recording/derived-weight/source/API redistribution rights. Existing
  model repository stays private for developers; do not invent a public license.
- Real accounts, server-side access controls, password recovery and consented
  progress sync; localStorage profiles are not a security boundary.
- HTTPS/WSS, authentication, quotas, abuse limits and clear privacy disclosures.
  Restrict local audio endpoints to development deployments; validate consent
  and rights before hosting training/reference recordings publicly.
- Teacher-reviewed bilingual pedagogy/UI and translations; accessibility,
  longer learner sessions, support/recovery and mobile microphone testing.
- API resilience/licensing/cache plan. Quran Tafseer currently responds over
  HTTP; the server proxy avoids browser mixed content but upstream text is not
  transport-authenticated. Render it as text, never HTML, and do not present it
  as a riwāyah-specific authoritative Quran source.

## Verification of this update

- Production Next.js build: passed (including TypeScript checks).
- Python: **46 tests passed** across matcher, buffer, WebSocket lifecycle,
  practice scoring, speaker holdout and Quran geometry.
- Browser: **28 tests passed** against the production server, including
  microphone-worklet-to-GPU and uploaded-audio-to-GPU tests using actual
  Al-Husary reference clips (Fātiḥah ayah 1 then ayah 3, omitting ayah 2).
  The remaining browser checks include isolated mocked-service contracts;
  those check UI behavior, not model quality.
- Live Arabic/English API lookups: passed; Qālūn Fātiḥah 1:3 correctly requests
  provider 1:4. These checks do not establish the provider's long-term uptime.
- Speaker-disjoint dry-run and no-leakage tests: passed; no training launched.
- Full phonetics audit: 6,210 ayahs, **5,427 with additional review flags**,
  **0 approved**. Automated fixtures are not scholarly pronunciation validation.
- Desktop/mobile challenge layouts checked at 1440px and 390px; no browser
  errors in the screenshot check. Git diff whitespace check: passed.

Passing one real-reference streaming fixture is a regression check, not a
claim about arbitrary foreign learner speech, all omitted verses or long ayahs.
The historical WER/CER remain unchanged; no new accuracy gain is claimed.

## Sources consulted

- [Quran Tafseer API documentation](http://api.quran-tafseer.com/en/docs/)
  and live book list (`/tafseer/`): Arabic tafsir IDs 1–8, English translation
  IDs 9–10; books are discovered by language, not guessed from numeric IDs alone.
- [Quran JSON methodology](https://github.com/risan/quran-json): current Hafs
  machine transliterations use vowels, sun letters, wasl and pause/assimilation
  rules and explicitly state they are not qualified-reader reviewed. Reuse those
  *design principles*, not Hafs output as Qālūn ground truth. No upstream code
  was copied. Its mapped-numbering warning supports our numbering safeguards.
- Project `src/training_with_gpu/README.md`, saved predictions/metrics,
  `QaloonData_v10`-derived cache and canonical reciter folders.
