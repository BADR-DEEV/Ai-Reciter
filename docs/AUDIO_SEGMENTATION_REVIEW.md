# New reader audio: quarantine and segmentation

## Collected scope and restrictions

- Trabulsi: `Mathani-Ayat/qaloon-reciter-experiments`, pinned revision
  `06cebc9ed98ef5e308da92aa4f09240460a760c7`. Downloaded/hash-verified **568**
  source clips for Fātiḥah and surahs 78–114, plus source audit/license metadata.
  Inventory: `data/review_sources/trabulsi/inventory.json`.
  The upstream structural audit lists nine expected-code gaps and 43 fusion
  candidates, but **zero confirmed fusions**. In particular `001001` is a
  basmalah candidate and `001007` a possible fused ending with `001008` absent.
  Those are file-structure hints, not verified spoken-ayah assignments; the
  imported audit's full 603-file scope is wider than our 568-file requested scope.
  The source explicitly reports **unauthorized-source experimental terms** and
  prohibits training use. Downloading for the requested audit is not permission
  to train, redistribute, publish or add these clips to learner playback.
- Taha Al-Fahad: the supplied Assabile collection labels itself Qālūn ʿan Nāfiʿ.
  Catalogued **38** scoped tracks in `data/review_sources/assabile/inventory.json`.
  **Zero audio downloaded**: robots.txt disallows the player/audio-resolution
  endpoints; public download links redirect to HTML, not direct MP3s. No bypass,
  guessed audio URLs or replacement reciter. Obtain permitted direct URLs and
  recording/source authorization first. MP3Quran's current reciter catalogue
  did not contain a matching Taha Al-Fahad entry during inspection.

```powershell
python src/dataset_collection/collect_review_sources.py trabulsi
python src/dataset_collection/collect_review_sources.py assabile
```

Uses the existing Hugging Face login, never tokens in arguments or source code.
Retries resume files from a pinned revision. Quran/audio publication restrictions
remain separate from the private model repository's access control.

## Best practice: align what was actually spoken

**Do not trust the filename as an ayah label. Do not force every reference ayah
into the recording. Whisper large alone is not a boundary ground truth.**

1. Retain immutable raw recordings, source/revision, hashes and restriction flags.
   Audit duration, channel/rate, clipping, near-duplicates and recording origin.
2. Decode *blindly* in Arabic with a strong ASR model, preferably large-v3 as an
   independent proposal model rather than the four-reader fine-tune being tested.
   No expected-Quran `initial_prompt`; disable previous-text conditioning to
   reduce propagation of hallucinated text. VAD proposes speech regions, not
   verse boundaries. Listen to omitted VAD regions too.
3. Match timestamped words against the **Qālūn** source with explicit paths for
   skipped ayahs, fused adjacent ayahs, basmalah, repetitions, partial restarts,
   insertions and waqf. Keep separate occurrence IDs for repeated passages.
   An unmatched reference is not proof of absent audio. Ambiguous common phrases
   must remain flagged rather than receive the first plausible ayah label.
4. Refine strong candidate boundaries with independent Quran-aware acoustic
   alignment where available, listening and waveform/spectrogram review.
   Whisper cross-attention timestamps can drift; normalized-text agreement alone
   cannot distinguish harakat, madd, riwayah fidelity or every acoustic omission.
5. Prefer a naturally complete, clean occurrence. For repeats/restarts, either
   retain one reviewed complete occurrence or keep a longer sequence with an
   exact transcript including repeats. Do not silently splice a “perfect” ayah.
   If a clip holds two ayahs, cut only when the boundary is acoustically verified;
   otherwise retain the true multi-ayah label in a separate sequence dataset.
   Basmalah gets its own segment or explicit text, never a fixed-second trim.
6. Preserve modest *measured* boundary margins after listening, but avoid clipping
   word onsets or including the next ayah. An internal waqf is not a new ayah.
   Store `start/end`, source hash, occurrence, full spoken text, coverage,
   anomalies, reviewer, approval and provenance alongside exported 16kHz mono WAV.
7. Manually review the pilot (beginner surahs first), then review all anomalies
   and a stratified random sample. Review boundary accuracy separately from WER.
   Training admission needs **both** authorized provenance and approved alignment.
8. Keep source files/recording sessions in one split; evaluate with unseen readers
   and actual learners. Never allow fragments/repetitions of the same recording
   to leak across train/validation/test. Do not treat new expert-reader clips as
   evidence of unfamiliar learner performance.

## Implemented prototype and pilot command

`review_audio_segments.py` implements blind faster-whisper ASR, word timestamps,
local reference proposals with exact first/last anchors, coverage/precision gates,
repeat occurrences, overlap flags and unmatched references. It is **not** a
complete forced aligner or automatic training exporter. Partial ayahs may remain
unmatched. Output is always `usable_for_training: false`; authorization restrictions
are propagated. Existing review JSONs are never overwritten.

Install optional dependencies in a separate environment without replacing your
existing CUDA PyTorch build:

```powershell
python -m pip install -r src/learning/requirements-review.txt
python src/learning/review_audio_segments.py --inventory data/review_sources/trabulsi/inventory.json --output data/review_sources/trabulsi/proposals-large-v3-pilot --model large-v3 --limit 10
```

This downloads a large ASR model if not cached. Run a small pilot first; `--limit 0`
explicitly requests all source clips. Do not launch concurrently with a busy
GPU inference/training session. Missing Assabile audio cannot be segmented.

### Executed large-v3 pilot — 2026-10-03

Ten source clips were decoded on CUDA using blind large-v3, producing **12
timestamped reference candidates** in
`data/review_sources/trabulsi/proposals-large-v3-pilot/`. **Zero are approved**.
No complete automatically labeled training dataset or verified cuts were exported.

- `001001`: ASR says basmalah, with an overlapping Fātiḥah 1:2 subphrase
  candidate; both are explicitly flagged as ambiguous. Do not count this as
  an extra spoken ayah.
- `001002`–`001006`: local candidates for Qālūn Fātiḥah 1:1–1:5. Filenames are
  therefore not interchangeable with the Qālūn ayah IDs.
- `001007`: ASR contains Fātiḥah 1:6 followed by most of 1:7, but renders the
  last word incompletely (`الضال`). Only 1:6 passes the exact terminal anchor.
  The unmatched 1:7 is **not** evidence of absent audio; listen/refine the boundary
  and transcript rather than weaken the gate or relabel the whole source clip.
- `078001`: ASR proposes basmalah at 0.00–4.02s and ayah 1 at 4.02–9.66s;
  `078002`/`078003` propose ayahs 2/3. These are cross-attention timestamp
  candidates, not validated sample-accurate boundaries.

The pilot supports the conservative review workflow, not a claim that all clips
are segmented correctly. Review source licensing and listen to each accepted cut
before any training or learner use; current source authorization still prohibits it.
