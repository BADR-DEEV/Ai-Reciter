# Tarteel · Qālūn recitation studio

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

Open **http://127.0.0.1:3000**. Allow microphone access and choose **Begin recitation**.
No microphone/headphones? Select your surah, click **Upload audio to test**, and
choose a WAV, MP3, or another browser-supported audio recording (up to 50 MB / 10
minutes). The recording is decoded locally and silently streamed at real-time
speed through the same GPU pipeline. Results are genuine, not simulated. Testing
ends automatically when the file finishes; **Finish recitation** stops it early.
For an interview, use fullscreen and **Try the presentation demo** to demonstrate
green matches, a red omission, gray unreached ayahs, and uninterrupted progression.
The demo is not a claim about measured model performance.

For a production presentation build: `npm run build`, then `npm start` in `web`.
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
3. The selected surah’s first ayah initializes the **matcher context**, not a
   decoder prompt. Expected text is never fed into Whisper to inflate scores.
4. Ordered, one-to-one fuzzy word matching gives a **text agreement score**.
   Each recognized reference word becomes **green** individually, without
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

## Configuration

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
demo/reset behavior, help, and mobile overflow.
The opt-in microphone-to-GPU browser test uses `RECITER_LIVE_TEST_AUDIO`, an absolute
WAV path containing Fātiḥah ayah 1 followed by ayah 3 (deliberately skipping 2).

Fonts are self-hosted under `web/public/fonts` with their accompanying licenses.
Quran source: [MP3Quran](https://www.mp3quran.net/) page SVG and ayah-timing APIs;
reference text: the existing project’s `QaloonData_v10(1).json`. Respect their usage
terms and attribution when publishing. Existing training instructions are in
`src/training_with_gpu/README.md`.
Numbering-alignment reference: [risan/quran-json](https://github.com/risan/quran-json).
