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

python -m unittest src.learning.test_tajweed src.learning.test_review_audio -v
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
- Connected/separate madd requires a confirmed Qālūn route and consistent choices;
  route-dependent spans do not contain an unconditional numeric length.
- Context is joined words *within an ayah*, stopping at its end. It does not model
  arbitrary internal waqf or joining across ayahs. Pause-induced madd has 2/4/6 options.
- Red = madd candidates (strong red for necessary six-count madd), green =
  nun/tanwin idgham, cyan = qalqalah, blue = inherently emphatic letters,
  gray = nun/tanwin ikhfa; additional colors distinguish iqlab/ghunna.
- No comprehensive claims for mīm al-jam, pronoun ṣilah, hamza variants, contextual
  rā/Allah-lām heaviness, small-letter spellings, or disjoint-letter rules.
- Speech-match **backgrounds** are independent of tajweed **letter colors**.
- SVG artwork is the previously supplied Hafs-numbered artwork, text-aligned for
  regions; it is **not** a verified Qālūn mujawwad mushaf.

Qualified review must select a tariq, check every annotated ayah against an
authorized Qālūn source and recordings, resolve overlap/context behavior, and
record reviewer/source/version outside regenerated drafts before a certified release.
Regenerating the JSON replaces machine drafts, not a teacher-review workflow.

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
