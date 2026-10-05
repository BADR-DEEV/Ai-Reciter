# Playback cursor, commentary and Tarteel-base experiment

## Reader behavior

- English translation and Arabic tafsir are separate sections in a **physical
  left sidebar**, even with the Arabic interface. On phones the bar stacks below
  the reader. Select an ayah, click its row, listen, or recite to move commentary.
- Arabic and phonetic views share a **square word outline**. Listening follows
  the Web Audio **actual audio clock**, not a set-interval animation or evenly
  divided ayah duration. Live input outlines the latest recognized matching word;
  it is recognized-text feedback, not synchronous phoneme assessment.
- Mushaf/SVG mode marks **the whole ayah's mapped regions**, never individual SVG
  letters. This remains supplied artwork with draft Qālūn text-to-region mapping,
  not a certified Qālūn mushaf. Live recitation no longer disables this view.
- Phonetics follows the Arabic word index only when both source representations
  have the same word count. There are **no letter timestamps**; a Latin spelling
  cannot be timed by splitting a word's seconds between its letters.
- Missing, stale, disagreeing or overlapping timings fall back explicitly to
  **ayah-only** highlighting. No alternate reader/audio is substituted.

`/api/reference-timing` accepts the same reader/ayah IDs as reference audio.
Sidecars live at `data/playback_timings/large-v3/<reader>.json` or the initial
Tarteel experiment's `data/playback_timings/<reader>.json`. Each record binds exact
existing `displayText` and indexed words to the served WAV's **SHA-256** and
duration. Server and client validate text, indices, nonoverlapping finite times,
and duration; the server rejects changed WAV hashes. Proposals carry an explicit
machine-draft label. These files are **not human-approved acoustic alignments**.

Generate in a fresh path (never overwrite an older review sidecar):

```powershell
python src/dataset_collection/build_playback_timings.py --dataset src/dataset_collection/dataset_qaloon_hutafi --reciter huthaify --model large-v3 --output data/playback_timings/large-v3/huthaify.json
# The same command supports --reciter husary / dokali with their actual dataset.
```

The completed default-reader run produced **455/569** exact, monotonic word-timing
proposals with large-v3. The initial Tarteel-base run produced **29/569**. The rest
remain ayah-only; recognizer agreement is not proof of precise word boundaries.
The completed Husary run provides **391/569**, and Dokali **359/569**, word
proposals. Source WAV hashes,
all unique canonical IDs and monotonic word intervals were independently checked
for all three readers. Large-model timing jobs should run **one at a time**
on this machine: simultaneous model loads hit host-memory allocation limits.
All seven Fātiḥah ayahs have default-reader word proposals. Existing WAVs and
dataset metadata were not modified.

## Concrete aḥkām corrections

Superseded: `build_qalon_tajweed.py` now delegates to the rule engine in `src/tajweed`
(see [`QALOON_TAJWEED.md`](QALOON_TAJWEED.md)), which works on the full Qālūn text and fixes the
v0.3 draft's mīm-sākin rules that never fired, the madd-lāzim false positives on كَفَرُواْ and
the lost iqlāb after tanwīn.

## Pinned Tarteel Quran-base for segmentation

This is **Tarteel's pretrained Quran-base**, not OpenAI base, not Whisper-small,
and not one of this project's fine-tuned/private deployment checkpoints:

- Weights: `tarteel-ai/whisper-base-ar-quran`
- Revision: `5c3c53fdf9272c4f6ee0bee09a1e5a4a615ee25c`
- Runtime: faster-whisper / CTranslate2, converted without fine-tuning.
- References: the existing `QaloonData_v10(1).json` loader and canonical Qālūn
  numbering, **not Hafs expected text**.

The legacy Tarteel tokenizer omits 1,501 timestamp tokens. Conversion restores
matching OpenAI token metadata only after verifying every shared ID (the explicit
Hebrew `iw`→`he` metadata rename preserves ID 50279). **Tarteel weights are not
replaced**, and no private/deployed model is modified. Conversion provenance is
saved in `segmentation_model.json`; missing/mismatched provenance fails loading.

```powershell
python src/dataset_collection/prepare_tarteel_segmenter.py --output-dir models/segmentation/tarteel-whisper-base-ct2
python src/dataset_collection/segment_and_slice.py --input-dir ahmad_tarabulsi/mp3 --output-dir data/segmentation_review/trabulsi_tarteel_new --model tarteel-base --compute-type float16 --reciter "Ahmad Al-Trabulsi (Qaloon)" --reciter-key trabulsi --surahs 79,112
# Omit --surahs for all 38 scoped recordings / 569 canonical references.
# Output metadata contains real accepted WAVs only. Incomplete coverage exits 2.
```

### Actual result—not an improvement claim

The CUDA 79/112 pilots ran end-to-end, preserving originals at:

- `data/segmentation_review/trabulsi-tarteel-base-079-112-v1/`
- `data/segmentation_review/trabulsi-tarteel-base-079-112-v2/`

Both exported **0/49** clips through the unchanged strict admission checks. In
the second pilot, **34** failed raw content agreement, **9** had no full-text
match and **6** had unresolved boundaries. It retains **34** failed cut proposals
and **40** wider contexts; `coverage.jsonl` accounts for all 49 expected IDs.

Tarteel-base frequently transcribed the Quran words but placed their edges across
pauses, omitted the preamble, added an initial hamza (e.g. `أَاللَّهُ`), or generated
extra speech on cut clips. Its timestamps cannot simply be treated as robust
ayah boundaries. Restoring token IDs does **not** retrain timestamp/alignment
behavior or certify Qālūn pronunciation.

One real generic detector fix from this experiment: analyze a **whole pause**
when a late terminal-word estimate falls inside it, while keeping the **cut**
protected by the terminal-end threshold. A quiet gap entirely before `طوى`
remains rejected. Raw/cleaned content, neighbor dependencies, sample grid and
full expected-ID coverage gates are unchanged.

Do not make Tarteel-base the default cutter based on this pilot. A future
two-stage experiment can use its transcript with a separately validated acoustic/
phoneme aligner and actual listening decisions. No such alignment certification,
full training, deployment replacement or 569-WAV completeness is claimed here.

## Verification

Latest checks: **131 Python tests passed**, **39 browser tests passed** (two
live-GPU tests explicitly skipped), TypeScript checks and production build passed.
Browser coverage includes physical-left placement in Arabic/English, Arabic and
phonetic playback cursors driven by decoded audio, stop/reader-change cleanup,
ayah-only fallback, whole-ayah Mushaf overlays, commentary failure handling and
the actual default-reader timing API. The real desktop playback screenshot is at
`C:\Users\Gaming\AppData\Local\Temp\opencode\rattil-left-sidebar-playback.png`.
