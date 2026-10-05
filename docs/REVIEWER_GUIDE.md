# Reviewer entry points

## Run the product, not a training script

Start with the root [README](../README.md). Use a working CUDA Python environment
for `src/streaming/server.py`; run the Next.js app from `web/`. Development output
uses `.next-dev`, production `.next-production`; their compiled assets never
share a directory. Stop a production server before rebuilding, and stop any
frontend on port 3000 before starting another server on the same port.
If `runs/gpu_base_full` is missing, `python src/deployment/restore_local_full.py`
restores the existing private published full-model release at the receipt's pinned
commit, checks every recorded SHA-256 and refuses to replace differing files.
It requires authorized cached Hub login; no tokens in source or CLI arguments.

| What to review | Entry points |
| --- | --- |
| Shared top navigation and Arabic UI | `web/components/learn/site-header.tsx`, `web/lib/i18n.tsx` |
| Reader/dashboard | `web/app/studio/page.tsx`, `web/app/reader.css` |
| Reference reciter and local ayah audio | `web/lib/reciters.ts`, `web/lib/quran-server.ts`, `web/components/ayah-listen.tsx` |
| Whole-Quran **draft** tajweed annotations | `src/learning/build_qalon_tajweed.py`, `web/lib/tajweed.ts`, `web/app/api/tajweed/route.ts` |
| Phonetics rules/audit | `web/lib/qaloon-g2p.ts`, [audit guide](QALOON_PHONETICS_AUDIT.md) |
| Challenges and browser-local profiles | `web/app/games/page.tsx`, `web/app/profile/page.tsx` |
| GPU streaming/matching | `src/streaming/`, [accuracy plan](ACCURACY_AND_PRODUCT_PLAN.md) |
| Quarantined source collection | `src/dataset_collection/collect_review_sources.py` |
| Experimental segmentation proposals | `src/learning/review_audio_segments.py`, [audio review guide](AUDIO_SEGMENTATION_REVIEW.md) |
| Existing experiments and publication | `src/training_with_gpu/`, `src/deployment/` |

The active navigation is horizontal on every main page, including studio and
lesson pages. Detailed lesson source content remains English and is labeled as
such when using Arabic; it is not presented as a completed Arabic curriculum.
Studio, challenge and profile controls are bilingual. Arabic tafsir is distinct
from English translation of meanings; neither replaces canonical Qālūn text.

## Generated/private files: do not publish automatically

- `web/public/quran/`: local Quran cache, per-reciter similarity indexes and
  **`qalon_majwad_mushaf.json`**. Rebuild using commands below; this directory is
  Git-ignored. The name requested by the product does **not** confer certification.
- `data/review_sources/`: restricted source audio/inventories; Git-ignored and
  completely separate from `dataset_qaloon_*` used by training/reference playback.
- `runs/`: existing model experiments; do not delete or rerun as a setup step.
- Root WAVs, `walid_qaloon/`, `presentation/`, old dataset collectors and course
  source timing notes are pre-existing assets, not safely disposable “unused files.”
  They were preserved. Obsolete full-surah `playClip` playback was removed; active
  lessons now use the same validated ayah endpoint as the reader and challenges.

```powershell
# Root, using the existing CUDA environment
python src/dataset_collection/cache_quran_pages.py
python src/learning/build_qalon_tajweed.py
python src/learning/build_audio_similarity.py --reciter huthaify
# Optional alternate-reciter indexes, if their datasets exist:
python src/learning/build_audio_similarity.py --reciter husary
python src/learning/build_audio_similarity.py --reciter dokali
# Challenge distractors (docs/CHALLENGE_DISTRACTORS.md); training is optional
python src/learning/train_ayah_embedder.py
python src/learning/build_text_embeddings.py

python -m unittest src.learning.test_tajweed src.learning.test_review_audio src.learning.test_text_embeddings -v
python -m unittest src.dataset_collection.test_segment_and_slice -v
python -m unittest src.training.test_reviewed_audio src.training_with_gpu.test_tarteel_training -v
python -m unittest src.deployment.test_restore_local_full -v
python -m unittest src.streaming.test_matcher src.streaming.test_server src.dataset_collection.test_quran_geometry -v
cd web
npm.cmd run typecheck
npm.cmd run build
npm.cmd start
# In another terminal in web/:
npx.cmd playwright test
```

## Tajweed release gate — currently NOT met

Generated for all **6,210 source ayahs**, but **zero teacher approvals**.
Offsets use UTF-16 for browser slices; source text remains unchanged. Spans carry
`needs-review`, and the UI is opt-in with explicit limitations.
Spans address the existing simplified `displayText`, with original `source_text`
preserved separately; matching text is checked before any colors are applied.
Display typography does not preserve enough notation for complete tajweed inference.

- Natural madd: **2 ḥarakāt**, necessary madd: **6**. Not 4/6 fixed seconds.
- The current draft lists **Shāṭibiyyah** options: ordinary unchanged connected
  madd **4**, separate madd **2 or 4**, with consistent performance choices.
  Changed adjacent hamzas are flagged separately, without an unconditional count.
  Other routes/combinations are not inferred. The supplied Scribd link served a
  client challenge and could not be read; the public secondary reference
  *The Secure Way to Rewayat Qalun*, pp. 8–14 and 18–22, was consulted instead.
  Uploading the supplied document is still necessary for exact-source review.
- Context is joined words *within an ayah*, stopping at its end. It does not model
  arbitrary internal waqf or joining across ayahs. Pause-induced madd has 2/4/6 options.
- The sample image guides the grouped palette, **not** its recitation-specific
  durations: purple necessary madd, pink ordinary connected madd, amber permitted
  madd, gold natural madd, green nasalization/ikhfa/iqlab, gray merged/silent
  letters, blue-teal tafkhim, cyan qalqalah. The bilingual key appears below the
  reader, with expandable rule/context notes and letter tooltips. Basmalah also
  receives text-preserving annotations. Source silent marks are used only when
  source/display base-letter sequences match exactly.
- Color spans inherit the surrounding font/weight and remain inline; no inserted
  joining characters or rewritten Quran text. Browser regression checks compare
  word text, font runs and widths before/after coloring, including Arabic/mobile.
- No comprehensive claims for mīm al-jam, pronoun ṣilah, hamza variants, contextual
  rā/Allah-lām heaviness, small-letter spellings, or disjoint-letter rules.
- Speech-match **backgrounds** are independent of tajweed **letter colors**.
- SVG artwork is the previously supplied Hafs-numbered artwork, text-aligned for
  regions; it is **not** a verified Qālūn mujawwad mushaf.

Qualified review must select a tariq, check every annotated ayah against an
authorized Qālūn source and recordings, resolve overlap/context behavior, and
record reviewer/source/version outside regenerated drafts before a certified release.
Regenerating the JSON replaces machine drafts, not a teacher-review workflow.

## Color/joining and slicer verification — 2026-10-04

### Latest Trabulsi direct-source repair

**Superseding terminal-boundary correction:** the earlier 333-candidate artifact
below missed a real `طوى` leak at 79:16→17 despite transcript agreement. Current
`segment_and_slice.py` is `terminal-protected-coverage-v4`: terminal-end protection,
noncascading shared-neighbor quarantine, PCM16 verification, bounded blind regional
retries, retained failed proposals, reviewer-hashed frame overrides and default
failure for incomplete canonical coverage. See
[the 569-row contract](AUDIO_SEGMENTATION_REVIEW.md#terminal-word-leakage-and-the-569-row-contract).
`coverage.jsonl` inventories every reference but is **not training metadata**;
null-audio rows cannot substitute for a complete 569-WAV dataset. Validator default
scope now includes entirely absent surahs and checks the common reader schema.
The v4 79-only repaired proposal cuts after طوى at 88.8475s; recognizer spelling
disagreements still require listening adjudication. Preserve earlier outputs as
history, not acoustically approved data.

Latest full-scope output: `dataset_qaloon_trabulsi_terminal_guarded_v2/` (289
unapproved paired WAVs, 569 explicit coverage rows, **280 withheld references**).
Original candidate/context evidence is retained under
`data/segmentation_review/trabulsi-terminal-v4-full/`; the completeness gate failed
as intended. **126 Python tests passed.** Independent exported-WAV validation:
256 automatic passes / 33 voiced-start flags / zero text disagreements, with
source-hashed predecessor-terminal evidence preventing unnecessary cascading
quarantine. Report:
`data/segmentation_review/trabulsi-terminal-v4-validation-v2/validation.json`.
This does not satisfy a complete,
perfect 569-WAV dataset yet; it exposes every remaining decision for qualified
review rather than hiding skips or accepting leaked terminal words.

- Original dataset audit: 497 incorrectly named reciter rows, 144 strict ASR
  disagreements, with partial `112:3` and prior-tail leakage into `112:4` confirmed
  by blind transcripts. Originals remain unchanged.
- Verified all 38 full-surah MP3s against official MP3Quran Qālūn bytes; saved
  affirmative reuse policy evidence. Direct publisher provenance is distinct
  from the restricted Hub collection. See the training guide for source review.
- `dataset_qaloon_trabulsi_repaired/`: 333 paired raw/cleaned review candidates,
  one combined `review_bundle.json`, 569 explicit reference decisions, 236
  withheld references, zero approvals. Decoder fallback recovered surah 92.
- Independent exported-WAV audit: **302/333 automatic passes**, 31 boundary flags
  including one ASR disagreement; all four Ikhlāṣ ayahs pass. Report:
  `data/segmentation_review/trabulsi-repaired-validation-v2/validation.json`.
- **115 Python tests passed.** Trabulsi admission now requires the strict validation
  report AND actual listening/boundary review AND source-specific permission review.
- README now includes reproducible Tarteel **base and tiny** training commands,
  with/without optional admitted Trabulsi; tiny uses a pinned checkpoint and the
  same corpus/gates. An actual tiny one-step CUDA training/save/decode smoke run
  passed at `runs/tarteel_tiny_qaloon_smoke_v1/`. No complete fine-tuning or accuracy
  gains claimed; tiny smoke-set scores are not a benchmark.

### Previous madd/cleaning verification

- Segmentation logic is consolidated in `src/dataset_collection/segment_and_slice.py`.
  Runs now create a complete `review_bundle.json` with reasons for every withheld
  reference and clearly separate raw contexts for uncertain proposals.
- Default joint measured-pause boundaries repair both sides of a cut; raw waveform
  analysis is not limited by a possibly early Whisper next-word timestamp.
  Conservative cleaning preserves frame counts, originals and paired raw WAVs.
- Final local Taha pilot (79, 87, 93): **58 unapproved raw/cleaned pairs**, 17
  withheld references, 11 wider contexts. See the single bundle at
  `data/segmentation_review/taha-madd-clean-v6-pilot/review_bundle.json`.
- **103 Python tests passed**, including vowel-tail/shared-boundary, one-word
  preceding-pause, integer sample-grid, cleaning, Trabulsi admission and Tarteel setup regressions.
- Tarteel three-reader dry-run: 1,388/141/174 train/validation/test. Actual one-step
  CUDA training/save/decode smoke run passed; **no full training or new benchmark
  improvement is claimed**. See [TARTEEL_QALOON_TRAINING.md](TARTEEL_QALOON_TRAINING.md)
  for comparable OpenAI/Tarteel commands and the Trabulsi permission/review gate.
- No Trabulsi training admission, deployed-model replacement or Hub upload.

### Earlier color/joining checks

- Regenerated all 6,210 ayahs, 110,721 candidate spans, zero approvals; canonical
  display/source retention and UTF-16 offsets checked by tests.
- Production build (including type validation) passed. **32 browser tests passed**;
  the two actual GPU/live-audio tests were **skipped** this time, not revalidated.
  Tested the existing development frontend on `http://127.0.0.1:3001` via
  `PLAYWRIGHT_BASE_URL`; no existing frontend process was stopped/replaced.
- **83 Python tests passed**, including 18 regressions for the fixed slicer.
- Local Taha large-v3/int8-float16 pilot: **44 unapproved WAV proposals** in
  `data/segmentation_review/taha-boundary-v3-pilot/`, with all 18 unexported ayahs
  listed in reports. No training dataset was overwritten/admitted. Details and
  commands: [AUDIO_SEGMENTATION_REVIEW.md](AUDIO_SEGMENTATION_REVIEW.md).
- The palette and software checks are not teacher approval or certified acoustic
  boundaries. Source permissions remain unresolved; no model publication occurred.

## Local verification — 2026-10-03

- Production build and TypeScript check passed.
- All 33 browser tests passed, including the real-reference microphone-to-GPU
  and uploaded-audio-to-GPU regressions. The fixture deliberately skips Fātiḥah
  ayah 2; this is regression evidence, not general learner-accuracy validation.
- 60 Python test cases passed across streaming, geometry, speaker splitting,
  segmentation proposals, draft tajweed and restore-file safety checks.
- All 6,210 generated ayahs were checked for exact display/source-text retention,
  in-range UTF-16 spans and unapproved review status; this is a software integrity
  check, **not a religious/tajweed correctness validation**.
- Default Huthaify index contains 569 locally aligned clips. Browser checks
  compared actual Huthaify/Husary audio bytes and verified missing/unknown
  recording behavior; no voice fallback.
- The previously missing full model was restored from the pinned private
  release; all 12 artifacts matched their recorded SHA-256 hashes. No retraining
  or upload was performed. Backend `/health` reports ready on CUDA at port 8000;
  the production frontend runs at port 3000.
- Development/production output directories are now isolated after a shared
  `.next` cache overwrite caused missing frontend bundles during an earlier run.
- The 10-source-clip blind large-v3 CUDA pilot completed: 12 flagged/reference
  candidates, zero approvals. Basmalah overlaps and an ASR-truncated fused Fātiḥah
  ending stayed untrusted. See the audio review guide; no training cuts exported.
