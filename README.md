# Rattil · رَتِّل

Learn to read and recite the Quran in the Qālūn riwāyah, from the first letter to
whole surahs. The name comes from 73:4, *wa rattili l-qurʾāna tartīlā*: “and recite
the Quran with measured care.”

A green-and-white Next.js presentation interface connected to the local full
Whisper model in `runs/gpu_base_full`. Includes live microphone streaming and a
**clearly labeled, simulated presentation demo** that needs no microphone or GPU.

## Start

**Local setup (model v3 + datasets from Hugging Face, dev mode):** see
[`docs/LOCAL_SETUP.md`](docs/LOCAL_SETUP.md).

**Reviewers:** see [`docs/REVIEWER_GUIDE.md`](docs/REVIEWER_GUIDE.md) for active
entry points, setup/check commands and release gates.

**Audio preparation:** [`docs/AUDIO_SEGMENTATION_REVIEW.md`](docs/AUDIO_SEGMENTATION_REVIEW.md)
describes the single-file slicer, shared madd boundaries, reversible noise cleaning
and the consolidated `review_bundle.json`. **New training experiments:**
[`docs/TARTEEL_QALOON_TRAINING.md`](docs/TARTEEL_QALOON_TRAINING.md) covers the
three-reader Tarteel/OpenAI-base comparison and approval/permission-gated Trabulsi.

Requires Node.js 22 LTS or newer and the Python/CUDA environment used for training.
Do not replace your working CUDA PyTorch build with a CPU wheel.

```powershell
# Repository root: install the API dependencies and cache the Quran once.
python -m pip install -r src/streaming/requirements.txt
python src/dataset_collection/cache_quran_pages.py
# Only if runs/gpu_base_full is missing, with authorized private-Hub access:
python src/deployment/restore_local_full.py

# Terminal 1 — local inference (model loads once, not per request)
python -m src.streaming.serve --model gpu-full-base
# Or the new private DeepDML-trained adapter:
python -m src.streaming.serve --model deepdml

# Terminal 2 — interface
cd web
npm install
npm run dev
```

Open **http://127.0.0.1:3000** for the beginner course (alphabet → short surahs; see
[`docs/LEARN_TO_READ.md`](docs/LEARN_TO_READ.md)). The recitation studio lives at
**http://127.0.0.1:3000/studio**. Allow microphone access and choose **Begin recitation**.
No microphone/headphones? Select your surah, click **Upload audio to test**, and
choose a WAV, MP3, or another browser-supported audio recording (up to 50 MB / 10
minutes). The recording is decoded locally and silently streamed at real-time
speed through the same GPU pipeline. Results are genuine, not simulated. Testing
ends automatically when the file finishes; **Finish recitation** stops it early.
For an interview, use fullscreen and **Try the presentation demo** to demonstrate
green matches, a red omission, gray unreached ayahs, and uninterrupted progression.
The demo is not a claim about measured model performance.

For a production presentation build: `npm run build`, then `npm start` in `web`.
Stop the frontend on port 3000 before starting another server on that port.
Development uses `.next-dev`; production build/start use `.next-production`, so
development compilation cannot invalidate presentation bundles. Do not build
against an actively running production server; stop it before rebuilding.
The app and API bind to loopback by default. `/health` on port 8000 reports the
loaded model, CUDA/CPU device, and active sessions.

## Local Quran assets and metadata

- `web/public/quran/pages/`: **604 unique SVG pages**, downloaded once from MP3Quran.
- `web/public/quran/geometry/`: shared API geometry from **one reciter**, read ID 5.
  Shared geometry deliberately excludes that reciter’s audio timestamps.
- `text-mapping.json` aligns Qālūn source ayahs to the Hafs-numbered page artwork
  by ordered text, not timing-array index. `hafs-reference.json` is cached solely
  for this numbering alignment; scoring still uses the project's Qālūn text.
- `web/public/quran/surahs/` and `manifest.json`: all 114 surahs, Qālūn reference
  text, merged-source-ayah regions, page dimensions, and training-coverage flags.
- Existing `dataset_qaloon_*/metadata.jsonl` and `walid_qaloon/metadata.jsonl` gain
  `quran_regions`, `geometry_source`, and single-region convenience fields.
  `start_time`, `end_time`, audio paths, and training labels remain unchanged.
  Waleed gets the same geometry **without invented reciter timestamps**.
- Fātiḥah’s Waleed audio numbering and combined final clip are handled explicitly.
  Merged ayahs keep **all** source polygons, including polygons on different pages.
- Every metadata file receives a `.before-geometry.bak` backup before its first
  atomic update. Subsequent cache runs reuse existing files. Assets and dataset
  backups are generated locally and excluded from Git (about 346 MB of SVGs).

Qālūn timing IDs can return null or wrongly numbered geometry. The cache instead
uses the artwork's Hafs numbering and aligns it to the actual reference text.
Where Qālūn divides a whole-page ayah differently, those shared polygons are
flagged and **Ayah view is selected automatically for accurate live highlighting**.
No coordinates are fabricated. **Ayah view** always provides the complete local
Qālūn text. MP3Quran artwork and numbering are shown as supplied; the shared
page API is not proof that its glyphs/numbering are a verified Qālūn mushaf. The
Qālūn source has different ayah divisions in some surahs; consult a verified
mushaf/teacher for authoritative reading. Training covered **Fātiḥah and 78–114**;
other surahs are available but live recognition is explicitly experimental.

Cache only training coverage with `--trained-only`. Use `--skip-metadata` to build
assets without enriching datasets. Adding geometry does not retrain the model.

## Streaming behavior

1. An AudioWorklet downmixes/resamples microphone PCM to mono **16 kHz Float32**.
   Small frames go over WebSocket; no browser speech recognizer or cloud ASR.
2. The server collects a bounded 28-second rolling window and performs greedy
   local Whisper decoding around every 1.2 seconds (actual latency depends on GPU).
   Simple RMS activity detection suppresses silence decoding; a quiet boundary
   triggers a final decode. Receive and inference tasks run independently.
   Utterance context is preserved across early ayah transitions; strongly
   aligned completed text is removed from retained-window matching, avoiding
   tiny cropped terminal tails being interpreted as the next ayah.
   An isolated first word after an ayah transition may stay pending until a
   second target-word anchor corroborates it; its audio is retained meanwhile.
3. The selected starting ayah initializes the **matcher context**, not a
   decoder prompt. Expected text is never fed into Whisper to inflate scores.
4. Ordered, one-to-one fuzzy word matching gives a **text agreement score**.
   Each corroborated reference word becomes **green** individually, without
   waiting for an ayah to reach 65%. Words skipped before later recognized words
   become **red** after two consistent observations or a voiced boundary; future
   words remain **gray**, including when you pause partway through an ayah.
   A completed ayah or recognized later ayah confirms the remaining omissions.
   Already-heard words remain green if the reader skips the end of an ayah.
   At **65%** the ayah's summary counts as matched. An exact normalized terminal word, with
   earlier ordered evidence, advances after two consistent observations or a
   final voiced-boundary decode. Multiple ayahs in one decode are supported.
5. Strong evidence for one of the next three ayahs marks intervening omissions
   red and continues. A finalized poor match also turns red but does not stop
   recording. **Silence alone never marks a verse missed**. Future verses stay gray.
6. Finish stops the microphone immediately, waits for the final decode, and
   retains results. Reset/surah changes clean up audio tracks, worklets, timers,
   and sockets. No audio is recorded to disk.

This is **near-real-time windowed ASR**, not token-streaming Whisper. Recognition
can hallucinate, and long ayahs exceeding the rolling window may lose early words.
The 65% rule and skip margins are configurable heuristics, not calibrated clinical
or religious judgments. Validate on real student recordings before production.
This interface does **not** assess pronunciation or tajweed.
Word-by-word colors use **Ayah view**, which is selected automatically for live
sessions and audio uploads. MP3Quran SVGs contain whole-ayah polygons, not reliable
word coordinates, so **Mushaf** remains a reading/ayah-summary view. Red indicates
an expected word absent from the matched transcript, not proof of an acoustic
omission; ASR errors can still produce false alerts. Demo mode now animates words,
including one simulated within-ayah omission.

## Learning challenges, local profiles and Qālūn phonetics

- `/games`: next-ayah recall (blind local AI transcription or choices), match
  three real Qālūn audio clips, identify a surah, restore a word, and order ayahs.
  Choose difficulty and Fātiḥah/Juz ʿAmma or whole-Quran text scope.
- `/profile`: name + **demo password** stored only as a salted PBKDF2 hash in
  localStorage, with per-profile progress. This is not real authentication;
  never reuse a real password. No cloud sync or recovery.
- Studio: shared top navigation, starting/resume ayah, SVG/Arabic/**large
  phonetics**/meaning-with-phonetics views and per-ayah listening. **Al-Huthaify is
  the default**; select Al-Husary or Al-Dokali for their own validated Qālūn clips.
  Playback is disabled during studio recording; missing audio never substitutes
  a different reader or riwāyah.
  The rules use the vocalized Qālūn source, not copied Hafs Latin text.
  No entries are claimed scholar-approved. Read the audit before teaching.
- Meaning/tafsir: Arabic tafsir and English **translation of meanings** from
  Quran Tafseer API, behind a server-side proxy with numbering alignment,
  attribution and failure handling. Panels load on opening; inline meanings
  load only when their ayahs become visible. Arabic switching updates UI direction
  and commentary language. Detailed course teaching content remains labeled English.
  Commentary is not a replacement riwāyah-specific recitation text.

Generate the default reader's acoustic distractor index and whole-Quran tajweed draft:

```powershell
python src/learning/build_audio_similarity.py
python src/learning/build_qalon_tajweed.py
cd web
npm.cmd run phonetics:audit
```

Local audio/listening defaults to the existing Huthaify dataset; no Hafs recordings are
substituted. Without the generated index, difficulty explicitly falls back to
text similarity. MFCC similarity is a heuristic, not a phoneme/tajweed model.
Alternate indexes: `--reciter husary` / `--reciter dokali` with the matching datasets.
Tajweed colors come from the Qālūn rule engine in `src/tajweed` (rules from the Libyan
Awqaf curriculum on tajweed and the uṣūl of Qālūn's riwāyah). `python -m src.tajweed.build`
writes one small file per surah to `web/public/quran/tajweed/`; the first app start does it
automatically. Rules are machine-applied, not a certified mushaf: counts are ḥarakāt (natural 2,
muttaṣil 4, munfaṣil 2/4, lāzim 6). The `/tajweed` page teaches letter sounds, every rule and
Qālūn's riwāyah topics, and the studio's "Check my ahkam" switch uses the tajweed model when
it is installed. Details: [`docs/QALOON_TAJWEED.md`](docs/QALOON_TAJWEED.md).

New-reader collection/segmentation and restrictions:
[`docs/AUDIO_SEGMENTATION_REVIEW.md`](docs/AUDIO_SEGMENTATION_REVIEW.md).

Analysis and next priorities: [`docs/ACCURACY_AND_PRODUCT_PLAN.md`](docs/ACCURACY_AND_PRODUCT_PLAN.md).
Phonetics review guide: [`docs/QALOON_PHONETICS_AUDIT.md`](docs/QALOON_PHONETICS_AUDIT.md).
Full generated spreadsheet: `docs/generated/qaloon-phonetics-audit.csv`.

## Fine-tune Tarteel Whisper-base or Whisper-tiny on Qālūn

Both entry points fully adapt **Tarteel's existing weights**, defaulting to
**Dokali + Huthaify + Husary**, canonical Qālūn labels, seed 42 and Fātiḥah/Juz
ʿAmma scope. **Trabulsi is excluded unless explicitly admitted below.** Tarteel's
upstream corpus is undocumented; Hafs/Qālūn transfer and pretraining overlap need
evaluation. WER does not certify pronunciation or madd/tajweed.

Use your existing CUDA PyTorch/torchaudio environment; do not replace its CUDA
wheel. Choose a **new** output directory for each run. Nothing uploads/deploys.

```powershell
python -m pip install -r src/training_with_gpu/requirements.txt
python src/training_with_gpu/train_tarteel_base.py --dry-run
python src/training_with_gpu/train_tarteel_tiny.py --dry-run

# WITHOUT Trabulsi (default). Run one GPU job at a time.
python src/training_with_gpu/train_tarteel_base.py --output-dir runs/tarteel_base_three_reader
python src/training_with_gpu/train_tarteel_tiny.py --output-dir runs/tarteel_tiny_three_reader

# Optional short implementation check, not an accuracy benchmark.
python src/training_with_gpu/train_tarteel_tiny.py --output-dir runs/tarteel_tiny_smoke --max-steps 1 --max-samples-per-split 6 --batch-size 2 --gradient-accumulation 1
```

### Repair and validate Trabulsi before including it

**Requested experiment history:** the user chose **LoRA**, not a new full-layer
fine-tune. See [AI approaches and the current run](docs/AI_APPROACHES_AND_CURRENT_RUN.md)
for the launched Tarteel-base recipe, actual four-reader membership, recording-safe
splits, withheld Waleed test, added metrics and honest full-model comparison.
Taha was rebuilt and excluded from v1; its later user-directed private-research
inclusion is documented in the repair recipe below. See the
[project map](docs/PROJECT_MAP.md) for authoritative dataset paths and preserved
historical assets. Experimental acceptance does not create production approvals.

The first LoRA run has **completed, not been deployed**: **61.46% test WER** and
**9.43% Waleed adaptation-held-out WER**, with severe repetition failures.
See [completed results and comparison caveats](docs/TARTEEL_LORA_V1_RESULTS.md).

The subsequent requested repair adds a **five-reader private research recipe**,
canonical vowelled Qaloon targets, development-only decoder controls, pinned
adapter/config reload and abstention on unreliable decoding. See
[the measured repair findings](docs/TARTEEL_DECODE_REPAIR.md): plain beam search
helped v1, but a “true 7%” human-ready result is not assumed or advertised.
The selected v2 adapter's evaluation was recovered without retraining after a
post-export process exit: **3.80% test WER / 5.49% adaptation-held-out Waleed WER**,
zero flagged loops/EOS failures on those partitions. Its development advantage
over unadapted Tarteel is only one word error; it remains a **private candidate,
not validated learner grading or an approved deployment**. See
[v2 results](docs/TARTEEL_LORA_V2_RESULTS.md).

The later [DeepDML upgrade](docs/DEEPDML_QALOON_UPGRADE.md) fixes verified
vocative/consonantal-yaa normalization bugs with immutable versioned overlays,
adds fitting-only boundary quarantine, rank32 all-projection LoRA and base/small
12GB BF16 recipes, and separates blind omission evaluation from optional surah-word
trie assistance. Screenshot benchmarks remain unverified, not promised results.

**New reader UI and segmentation experiment:** translation/tafsir now occupy a
left sidebar; Arabic/phonetics use a WAV-hash-bound square word cursor when real
timing proposals are available, and Mushaf uses whole-ayah highlighting. See
[playback and the pinned Tarteel-base experiment](docs/PLAYBACK_AND_TARTEEL_SEGMENTATION.md)
for timing-generation and `--model tarteel-base` commands, concrete aḥkām fixes,
and the actual failed pilot results (no automatic replacement of better cuts).

**Latest terminal-word guard:** the earlier `dataset_qaloon_trabulsi_repaired/`
outputs below are now historical; a listener found leaked `طوى` at 79:16→17 that
ASR had failed to report. Do **not** treat the older automatic pass list as proof
of acoustic boundaries. See
[the updated completeness contract](docs/AUDIO_SEGMENTATION_REVIEW.md#terminal-word-leakage-and-the-569-row-contract)
and regenerate in a new directory using the current slicer.

Current runs write:

- `metadata.jsonl`: actual exported, auto-checked WAVs with the other readers'
  usual schema; **not padded** with absent/duplicate/partial clips.
- `coverage.jsonl`: exactly **569 canonical rows** for a full Fātiḥah/Juz ʿAmma
  run, including explicit unresolved rows with null audio fields and reasons.
  This is **not training metadata** and cannot substitute for 569 valid WAVs.
- `review_audio/*.candidate.wav`: retained uncertain boundary proposals rather
  than silently discarding them, plus wider `*.context.wav` files.

The slicer now exits **2 for incomplete coverage**, unless `--allow-incomplete`
explicitly requests a review pilot. `--surahs 79` expects 45 rows, not 569. The
validator defaults to **all 569 expected IDs**, including wholly absent surahs:

```powershell
python src/dataset_collection/validate_ayah_audio.py --dataset PATH_TO_NEW_TRABULSI_OUTPUT --output-dir data/segmentation_review/trabulsi_latest_audit --expected-reciter trabulsi --expected-scope fatiha-juz-amma
```

Zero training approvals are implied by either file's line count. No automatic
pipeline here promises perfect cuts; unresolved phonemes/orthography need qualified
listening/alignment review. Reviewer-authored, source-hashed frame corrections
can be supplied using `--boundary-overrides` (documented in the review guide).

**Latest local output:** `dataset_qaloon_trabulsi_terminal_guarded_v2/` has 289
auto-checked, unapproved WAV pairs plus a **569-row `coverage.jsonl`**. The full
run at `data/segmentation_review/trabulsi-terminal-v4-full/` also retains 134
uncertain complete-cut proposals and 209 wider contexts. **280 references remain
withheld**; the actual 569-WAV dataset is **not complete yet**. Additional shared-
boundary quarantine is intentional, not a claim that those ayahs are absent.
The independent exported-WAV audit at
`data/segmentation_review/trabulsi-terminal-v4-validation-v2/validation.json`
has **256 automatic passes / 33 voiced-start review flags**, with **no text
disagreements**. Every shared-start dependency has recorded predecessor-end
evidence; rejecting a predecessor's start does not mechanically invalidate its
separately verified end or quarantine all later ayahs.

**Do not train on `dataset_qaloon_trabulsi/` as it stands.** Its reciter field says
Taha incorrectly, and old cuts can contain half an ayah or the preceding tail.
The audit confirmed `112:3` transcribes **`لم يلد`**, while the label is **`لم يلد
ولم يولد`**; the missing phrase leaks into `112:4`. Originals are preserved.

**Already generated locally:** `dataset_qaloon_trabulsi_repaired/` has 333
raw/cleaned candidate pairs and a single `review_bundle.json` covering all 569
scoped reference decisions. The exported-WAV audit at
`data/segmentation_review/trabulsi-repaired-validation-v2/validation.json` has
**302 automatic passes / 31 review flags**; all four Ikhlāṣ ayahs pass. **236
references remain withheld**, not assumed absent. This is not a complete,
human-approved dataset. Start review with those existing files rather than
recreating them; the commands below demonstrate reproducible runs in NEW folders.

```powershell
python -m pip install -r src/learning/requirements-review.txt

# Optional original audit. Exit 2 means QA failures/partial audit, not a crash.
python src/dataset_collection/validate_ayah_audio.py --dataset dataset_qaloon_trabulsi --output-dir data/segmentation_review/trabulsi_old_audit --expected-reciter trabulsi

# Verify local full-surah MP3s against official Trabulsi Qaloon recording links.
python src/dataset_collection/verify_mp3quran_trabulsi.py --input-dir ahmad_tarabulsi/mp3 --output-dir data/review_sources/trabulsi_direct

# Rebuild complete ayahs, including internal pauses; never guess missing words.
python src/dataset_collection/segment_and_slice.py --input-dir ahmad_tarabulsi/mp3 --output-dir data/segmentation_review/trabulsi_rebuild --model large-v3 --compute-type int8_float16 --reciter "Ahmad Al-Trabulsi (Qaloon)" --reciter-key trabulsi --content-match exact
python src/dataset_collection/prepare_trabulsi_review.py --candidates data/segmentation_review/trabulsi_rebuild --source-inventory data/review_sources/trabulsi_direct/source_inventory.json --output-dir dataset_qaloon_trabulsi_corrected_new

# Re-read exported WAVs; require exact normalized word sequences.
python src/dataset_collection/validate_ayah_audio.py --dataset dataset_qaloon_trabulsi_corrected_new --output-dir data/segmentation_review/trabulsi_corrected_audit --expected-reciter trabulsi
```

The validator writes **`validation.json`**, readable `validation.csv` and a blank
`listening_decisions.csv`. It checks canonical labels, PCM16/mono/16kHz, duration,
silence, clipping/DC offset, voiced cut edges, frame/hash consistency, duplicates,
orphans and missing references. Blind ASR uses **no expected-text prompt or VAD
cropping**; partial, extra, repeated or substituted words fail. This is exact
**after the project's Arabic ASR normalization**, not literal harakat/phoneme
agreement. ASR can falsely reject good audio or miss an acoustic defect.

To admit only passing clips:

1. A qualified reviewer listens to **raw and cleaned** cuts against the full
   recording/Qālūn text. Fill `listening_approved` and `boundary_approved` with
   `true`, plus `reviewer` and `reviewed_at`, only for genuinely approved clips.
   Leave failures blank/false; never auto-fill approvals from ASR agreement.
2. Review [MP3Quran's reuse policy](https://www.mp3quran.net/eng/privacy) and create
   the source-specific permission receipt described in
   [the training guide](docs/TARTEEL_QALOON_TRAINING.md#direct-mp3quran-source).
   MP3Quran explicitly permits copying/using its material and links. The verified
   **direct** source is not the previously restricted Hugging Face collection.
   Record the responsible training-use decision; public release needs a separate
   licensing review.
3. Assemble the reviewed manifest, then supply **all three** admission arguments:

```powershell
python src/dataset_collection/approve_ayah_audio.py --dataset dataset_qaloon_trabulsi_corrected_new --validation-report data/segmentation_review/trabulsi_corrected_audit/validation.json --decisions data/segmentation_review/trabulsi_corrected_audit/listening_decisions.csv --permission-file PATH_TO_REVIEWED_PERMISSION_JSON

# WITH only approved, exact-validated Trabulsi clips. Inspect the dry-run first.
python src/training_with_gpu/train_tarteel_base.py --output-dir runs/tarteel_base_with_trabulsi --trabulsi-reviewed-manifest dataset_qaloon_trabulsi_corrected_new/metadata.reviewed.jsonl --trabulsi-validation-report data/segmentation_review/trabulsi_corrected_audit/validation.json --trabulsi-permission-file PATH_TO_REVIEWED_PERMISSION_JSON --dry-run
python src/training_with_gpu/train_tarteel_tiny.py --output-dir runs/tarteel_tiny_with_trabulsi --trabulsi-reviewed-manifest dataset_qaloon_trabulsi_corrected_new/metadata.reviewed.jsonl --trabulsi-validation-report data/segmentation_review/trabulsi_corrected_audit/validation.json --trabulsi-permission-file PATH_TO_REVIEWED_PERMISSION_JSON --dry-run
```

Remove `--dry-run` to launch the chosen experiment. If no clips pass **both** QA
and listening review, omit all Trabulsi arguments and use the three-reader recipe.
Only the passing, hash-matched subset is admitted, even if other candidates fail.
All Trabulsi cuts from one recording stay in one partition; this mixed protocol
is **not** an unseen-ayah benchmark.

### How reliable ayah-by-ayah preparation works

There is no single universal "industry-standard" cutting algorithm. Use
**recording/reciter-specific** timing annotations or a validated Quran/riwāyah-aware
aligner, canonical ayah IDs/text, waveform/phoneme boundary refinement, independent
content checks and human listening review. MP3Quran exposes an
[ayah timing API](https://www.mp3quran.net/eng/timing-api), but generic timing IDs or
Hafs numbering must not be assumed compatible with a different voice/Qālūn.
**A pause is not automatically an ayah boundary:** `لم يلد` can be an internal
waqf and must remain with `ولم يولد`. Continuous wasl/restarts/ambiguous cuts need
review, not guessed boundaries. Preserve originals, exact frames, hashes and
review decisions; avoid aggressive denoising, vowel fades or >30s target truncation.

## Service configuration

- Select at startup with `python -m src.streaming.serve --model gpu-full-base` or
  `--model deepdml`. Presets use `runs/gpu_base_full` and
  `runs/deepdml_qaloon_lora_base_v1/adapter`, respectively. Only one model/worker
  loads; restart the command to switch. DeepDML selection explicitly enables
  private experimental staging, not certified learner/tajweed assessment.
- `--model-path PATH`: override the selected model's location; DeepDML accepts
  either its parent run or adapter directory. No missing-model fallback.
- Optional `--device cuda --dtype fp16` (CUDA FP16 is automatic when available);
  `--beams 1` trades recognition accuracy for faster decoding. Without an override,
  DeepDML uses its saved beam3 policy and the full model uses greedy decoding.
  `/health` exposes the active model, precision, beam count and adapter status.
- `RECITER_MODEL_PATH`: direct-uvicorn model directory override; the launcher
  sets this for the chosen preset, so no environment-variable setup is needed.
- `RECITER_ALLOWED_ORIGINS`: comma-separated allowed UI origins; defaults to
  `http://localhost:3000,http://127.0.0.1:3000`. Other origins are rejected.
- `web/.env.local`: copy `web/.env.example` to override `NEXT_PUBLIC_RECITER_WS`.
  Remote deployments need HTTPS/WSS, authentication, quotas, and explicit origins.
- The backend serializes GPU access and allows at most two local sessions. It
  validates PCM size, rate, and finite values. Never expose this demo API publicly
  without authentication and additional operational controls.

## Checks

```powershell
python -m unittest src.streaming.test_matcher -v
python -m unittest src.dataset_collection.test_quran_geometry -v
python -m unittest src.learning.test_tajweed src.learning.test_review_audio -v
python -m src.streaming.replay_audio --surah 1 --audio fatiha.wav
cd web
npm run typecheck
npm run build
npm audit
npx playwright install chromium
npx playwright test
```

Replay needs mono 16 kHz PCM16 WAV files and sends them at real microphone speed.
Matching unit tests cover ordered duplicate words, partial green without advancing,
terminal-word stability, incorrect terminal words, omissions, multi-ayah decoding,
and Fātiḥah’s unnumbered basmalah. Browser tests cover asset loading, surah selection,
demo/reset behavior, help, mobile overflow, phonetic rule fixtures, five challenge
modes, hashed local profiles with isolated progress, and bilingual commentary.
The streaming regression suite also covers retained audio/context, tentative
boundary words, worker failures, starting-ayah validation and heartbeat handling.
The opt-in microphone-to-GPU browser test uses `RECITER_LIVE_TEST_AUDIO`, an absolute
WAV path containing Fātiḥah ayah 1 followed by ayah 3 (deliberately skipping 2).

Fonts are self-hosted under `web/public/fonts` with their accompanying licenses.
Quran source: [MP3Quran](https://www.mp3quran.net/) page SVG and ayah-timing APIs;
reference text: the existing project’s `QaloonData_v10(1).json`. Respect their usage
terms and attribution when publishing. Existing training instructions are in
`src/training_with_gpu/README.md`.
Numbering-alignment reference: [risan/quran-json](https://github.com/risan/quran-json).
