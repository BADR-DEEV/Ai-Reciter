# Rattil · رَتِّل

Learn to read and recite the Quran in the Qālūn riwāyah, from the first letter to
whole surahs. The name comes from 73:4, *wa rattili l-qurʾāna tartīlā*: “and recite
the Quran with measured care.”

A green-and-white Next.js presentation interface connected to the local full
Whisper model in `runs/gpu_base_full`. Includes live microphone streaming and a
**clearly labeled, simulated presentation demo** that needs no microphone or GPU.

## Start

Requires Node.js 22 LTS or newer and the Python/CUDA environment used for training.
Do not replace your working CUDA PyTorch build with a CPU wheel.

```powershell
# Repository root: install the API dependencies and cache the Quran once.
python -m pip install -r src/streaming/requirements.txt
python src/dataset_collection/cache_quran_pages.py

# Terminal 1 — local inference (model loads once, not per request)
python -m uvicorn src.streaming.server:app --host 127.0.0.1 --port 8000 --ws-max-size 16384

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
Stop `npm run dev` first; do not run development and production builds against
the same `.next` directory simultaneously.
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
- Studio: choose a starting/resume ayah and enable **draft Qālūn phonetics**.
  The rules use the vocalized Qālūn source, not copied Hafs Latin text.
  No entries are claimed scholar-approved. Read the audit before teaching.
- Meaning/tafsir: Arabic tafsir and English **translation of meanings** from
  Quran Tafseer API, behind a server-side proxy with numbering alignment,
  attribution and failure handling. Only opening the panel triggers lookup.
  Commentary is not a replacement riwāyah-specific recitation text.

Generate the acoustic distractor index (565 aligned clips in this dataset):

```powershell
python src/learning/build_audio_similarity.py
cd web
npm.cmd run phonetics:audit
```

Local audio quizzes need the existing Al-Husary dataset; no Hafs recordings are
substituted. Without the generated index, difficulty explicitly falls back to
text similarity. MFCC similarity is a heuristic, not a phoneme/tajweed model.

Analysis and next priorities: [`docs/ACCURACY_AND_PRODUCT_PLAN.md`](docs/ACCURACY_AND_PRODUCT_PLAN.md).
Phonetics review guide: [`docs/QALOON_PHONETICS_AUDIT.md`](docs/QALOON_PHONETICS_AUDIT.md).
Full generated spreadsheet: `docs/generated/qaloon-phonetics-audit.csv`.

## Service configuration

- `RECITER_MODEL_PATH`: full local model directory (default `runs/gpu_base_full`).
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
