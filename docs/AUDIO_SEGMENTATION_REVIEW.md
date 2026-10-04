# New reader audio: quarantine and segmentation

## Terminal-word leakage and the 569-row contract

### Root cause of 79:16 → 79:17

The earlier v3 cutter picked the nearest quiet gap **87.005–87.115s**, cutting at
**87.06s**, although the ASR terminal-word estimate ended at 87.52s. That gap was
**before `طوى`**, not the ayah end. `79:16` was withheld because its clip lacked
the terminal word. The shared right start was still moved backwards and `79:17`
was exported; its blind transcript omitted the leaked word. Thus ASR agreement
did **not** guarantee the physical boundary. The user's listening caught a real
defect, not just an overly strict gate.

The v4 79-only pilot proposes the cut **88.8475s**, after the terminal word, and
retains full unapproved proposals in:

- `data/segmentation_review/trabulsi-terminal-v4-079-pilot/review_audio/079016.candidate.wav`
- `data/segmentation_review/trabulsi-terminal-v4-079-pilot/review_audio/079017.candidate.wav`

16's repaired raw transcript contains all six word positions, but ASR writes
`اذ ناداه ربه بالوادي المقدس طواء` versus the canonical normalized
`اذ ناداه ربه بالواد المقدس طوى`. That is a **recognition/orthographic disagreement**,
not evidence that the recording is absent. The new terminal guard fixes the
specific premature cut; listening remains required. Neither proposal is silently
approved by altering Quran labels or weakening the strict content gate.

### Why rows were missing

The earlier combined 333-export run withheld 236 references:

| Gate/reason | References |
| --- | ---: |
| Unsafe/unresolved waveform boundaries | 86 |
| No ordered full-text ASR match | 80 |
| Raw clip not exactly recognized | 65 |
| Cleaned clip not exactly recognized | 5 |

These are **pipeline withholding reasons**, not a claim that a high-quality full
recording skipped the ayahs. Wrong boundary selection, weak timestamps, phonetic/
orthographic recognition differences, repeats/continuous joins and occasional
hallucinations all affect automatic coverage. The 30s Whisper target limit is an
additional general guard; never chop a long ayah while retaining its full label.

### Current guardrails (`terminal-protected-coverage-v4`)

1. A measured pause **before the terminal-word end** cannot become the ayah cut.
   Tiny internal dips no longer win solely by distance; sustained pauses are
   preferred within a bounded edge-word window. Terminal energy is calibrated
   from the word itself, not mostly-silent EOF frames.
2. A failed left-ayah check quarantines the **shared right start**, even if the
   next transcript looks exact. Failures are snapshotted so quarantine does not
   cascade mechanically into every following ayah. A preceding unsliceable
   candidate also cannot silently certify the next shared cut.
3. Raw/cleaned verification runs on the **PCM16 samples that will be exported**,
   avoiding the earlier float-vs-serialized-WAV disagreement. Canonical source
   text/labels remain unchanged; no expected-text decoding prompts are used.
4. Missing full-text matches receive **blind regional ASR retries**, bounded by
   known neighboring matches with overlapping ≤28s windows. Candidate timestamps
   stay absolute; retries that overlap existing evidence or reverse canonical
   order are rejected. Retry transcripts/outcomes are retained in the report.
5. Failed but located proposals are **saved for review**, not deleted or admitted
   as training data. Unlocatable references remain explicit unresolved entries;
   no boundary is inferred just to fill the row count.
6. Default completeness checks fail a run with missing canonical IDs (exit 2),
   including entirely missing source surahs. `--allow-incomplete` is an explicit
   pilot option, not permission to call a partial export complete.
7. Independent validation checks the common metadata schema, canonical filenames/
   ID set, all expected surahs, duplicates, null/nonexistent WAVs, word sequences,
   exact frames, overlap, engineering defects and shared-boundary dependencies.
   Default scope is Fātiḥah/Juz ʿAmma; `--expected-scope represented-surahs` is an
   explicitly narrower audit, not a 569-ayah certification.

### Two different JSONL files—not a padded dataset

For a full scoped run, **`coverage.jsonl` has exactly 569 rows** with the canonical
IDs, original Quran text and common reader-schema fields. Unresolved rows have
`coverage_status: "withheld-for-review"`, null `audio_filename/relative_audio_path/
start_time/end_time`, `missing_reason`, and any review-audio pointer. All remain
`usable_for_training: false`. This file makes omissions impossible to hide.

**`metadata.jsonl` contains only actual exported candidates.** It can be called
complete only if it has **569 unique exact expected IDs AND 569 valid full-ayah
WAVs**, with passing content/boundary checks and independent human review. A
569-line coverage inventory is not a substitute for a complete acoustic dataset.
The current algorithm cannot honestly promise the latter for all recordings.

### Reviewer frame corrections

Use `--boundary-overrides PATH_TO_JSON` for actual reviewer-authored corrections.
Each array row identifies `surah`, `ayah`, the immutable full-recording
`source_sha256`, absolute decoded `start_frame/end_frame`, `sample_rate: 16000`,
`reviewed_by`, and `reviewed_at`. Obtain the decoded frame grid/source hash from
the source report, and listen to both sides of each cut. Correct adjacent cuts
together, not only the left end. Unknown/duplicate IDs, wrong source hashes,
out-of-range/reversed/overlapping cuts and absent reviewer identity/date fail.
Overrides preserve canonical text and exact samples; **all content checks still
run**. This is not a switch to fabricate ASR agreement or training approval.

```powershell
python src/dataset_collection/segment_and_slice.py --input-dir ahmad_tarabulsi/mp3 --output-dir data/segmentation_review/trabulsi_terminal_review --model large-v3 --compute-type int8_float16 --reciter "Ahmad Al-Trabulsi (Qaloon)" --reciter-key trabulsi
# Expected coverage inventory: 569 rows. Nonzero exit if the WAV dataset is partial.
python src/dataset_collection/validate_ayah_audio.py --dataset data/segmentation_review/trabulsi_terminal_review --output-dir data/segmentation_review/trabulsi_terminal_audit --expected-reciter trabulsi --expected-scope fatiha-juz-amma
```

This is an auditable, review-first workflow using common production QA practices,
**not a certified "industry-standard" acoustic aligner or perfect-mushaf guarantee**.
For genuine full coverage, adjudicate the remaining cuts against the complete
recordings with a qualified Qālūn reviewer/validated phoneme aligner; do not
manufacture completeness with fuzzy labels, duplicated WAVs or placeholder audio.

The earlier pilot counts below are historical. In particular, the v3 `79:17`
automatic pass must not be treated as a reliable acoustic-boundary approval.

### Actual full-scope v4 run

`data/segmentation_review/trabulsi-terminal-v4-full/` processed all 38 local
recordings. Its **569-row `coverage.jsonl`** accounts for every expected ID.
It exported **289** unapproved raw/cleaned candidate pairs, retained **134**
failed complete-boundary proposals and **209** wider contexts; bounded retries
recovered **9** previously unmatched references. **280 IDs remain withheld**.
The default run correctly failed its completeness gate rather than claiming a
complete dataset. Fewer accepted exports than v3 are expected because v3 did
not quarantine unsafe shared starts and could accept the leaked-word example.

Current withholding breakdown: **75** unresolved/unsafe boundaries, **71** no
full-text match after regional retry, **69** raw transcript disagreements, **62**
rejected-neighbor shared-start dependencies, and **3** cleaned disagreements.
This is conservative automatic triage—not proof that the reciter omitted them.

Verified publisher provenance and common-schema metadata are prepared separately
at **`dataset_qaloon_trabulsi_terminal_guarded_v2/`**, with a combined `review_bundle.json`
and the same honest 569-row coverage inventory. Review-audio pointers resolve to
the retained original candidate/context files. The full-run 79:16 terminal cut
is **88.845s** (small energy-calibration variation from the first 79-only pilot);
79:17 uses the same shared cut. No `طوى` fragment is deliberately assigned to 17.
Both require review because 16's recognizer spelling still disagrees with the JSON.

**This is not the requested perfect 569-WAV training dataset yet.** Getting there
requires resolving the remaining acoustic/recognition cases, not relabeling a
569-line coverage inventory as complete training metadata. Originals and all
previous runs remain intact. **126 Python tests passed**, including the internal-
pause/terminal-word regression, shared-neighbor quarantine, PCM sample grid,
regional-retry collisions, absent-surah/duplicate-padding coverage, common schema
and reviewed manual-cut validation.

The independent audit at
`data/segmentation_review/trabulsi-terminal-v4-validation-v2/validation.json`
re-read all 289 exported WAVs: **256 automatic passes**, **33 voiced-start flags**,
**zero text disagreements**, zero shared-frame/dependency failures. Hashes,
exact paired frame counts, nonoverlapping intervals and all 569 inventory IDs
were separately checked. Coverage remains incomplete: 280 IDs are withheld.

To keep quarantine noncascading **without losing the evidence**, a successor whose
predecessor was withheld *only for its own start* carries
`previous_terminal_evidence`: the predecessor's exact raw/cleaned checks, immutable
raw candidate hash/path, original source hash, frame cut and terminal pause event.
The validator verifies those facts rather than demanding an approved whole
predecessor clip. An actual failed terminal/content check cannot supply this
evidence. This does not approve the predecessor or waive human listening.

## Current Trabulsi repair and strict WAV validation

The original `dataset_qaloon_trabulsi/` was audited without modification:
**497 WAVs**, all carrying a copied **Taha** reciter name; **144 strict blind-ASR
disagreements** (not 144 acoustically confirmed mistakes), 105 voiced-start and
98 voiced-end flags (overlapping categories). 353 transcripts exactly match the
normalized canonical text despite the identity-label defect. 51 references were
missing within represented surahs; surah 92 was absent entirely, so that count
is not whole-scope coverage. Report:
`data/segmentation_review/trabulsi-original-audit-v1/validation.json`.

Reported-example evidence:

- Old `112:3`: **1.550s**, transcript `لم يلد`, label `لم يلد ولم يولد`.
- Old `112:4`: transcript **`ولم يولد ولم يكن له كفؤا احد`**, containing the prior tail.
- Rebuilt `112:3`: **3.2899375s**, raw/cleaned transcripts both exactly `لم يلد
  ولم يولد`; rebuilt `112:4` also exactly matches its own reference.

Source verification matched **38/38** local full-surah recordings to official
MP3Quran **Trabulsi Qālūn** links and saved its affirmative reuse policy. This
direct source is **separate from** the restricted Hub collection below.
See the [README](../README.md#repair-and-validate-trabulsi-before-including-it) and
[training guide](TARTEEL_QALOON_TRAINING.md#direct-mp3quran-source).

New tools:

- `verify_mp3quran_trabulsi.py`: publisher recording hashes, links, policy snapshot
  and robots checks; no overwriting local sources.
- `segment_and_slice.py --content-match exact` (**now the default**): exact
  normalized raw/cleaned token sequences, no fuzzy acceptance or final-alif folding
  at the clip gate, no expected-text prompt and no clip VAD removal. Full-source
  matching remains proposal generation, not ground truth.
- `prepare_trabulsi_review.py`: separate corrected candidates, verified direct
  provenance, corrected reciter fields, canonical labels and matching raw cuts.
- `validate_ayah_audio.py`: re-read-WAV technical/content checks, explicit word
  differences/classifications, coverage/orphans and a blank listening CSV.
- `approve_ayah_audio.py`: actual human CSV approvals, strict validation and
  source permission review are all required for the accepted training manifest.

All machine candidates remain unapproved. Surah 92 exposed a libsndfile decoding
failure; the slicer now records a PyAV/FFmpeg fallback without silently dropping
packets or substituting recordings. Preserve previous outputs when retrying; merge
separate scoped candidates only when their ayah IDs do not overlap.

### Actual repaired output and independent exported-WAV audit

`dataset_qaloon_trabulsi_repaired/` contains **333** exact-checked raw/cleaned
candidate pairs from all **38** full-surah recordings. Its **`review_bundle.json`**
consolidates every source, source-verification evidence and **569** explicit
reference decisions. **236** references were withheld for review; no missing
content was fabricated. The successful 92-only retry is included while the
earlier decoder error remains in the bundle's history. Originals and prior runs
were not overwritten.

`data/segmentation_review/trabulsi-repaired-validation-v2/validation.json` re-read
every candidate WAV from disk: **302 automatic passes**, **31 boundary-review
flags**, including **one** serialized-WAV ASR disagreement (`90:12`). All four
Ikhlāṣ candidates pass both engineering and exact text checks, including the full
`112:3` and the separated `112:4`. Automatic passing rows remain **unapproved**.
All 333 paired WAV hashes/frame counts and nonoverlapping sample intervals were
separately checked. A full dataset-level pass is intentionally false; passing a
subset does not hide withheld references or approve the flagged candidates.

**Earlier pilots below used the then-current weaker/fuzzy transcript gate.** Their
spelling-equivalence behavior and pass counts are historical review evidence, not
the current strict training-admission policy.

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
   recording/source authorization first. This describes the earlier collector,
   not files subsequently supplied locally (see the slicing pilot below).
   MP3Quran's current reciter catalogue
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

## `segment_and_slice.py`: skip/leakage fix — 2026-10-04

The previous greedy first/end-word search could borrow an ending from a later
ayah, invent a start after a failed search, and move the cursor past intervening
references. A `max(rough_end + 0.5, next_start)` waveform wall permitted crossing
the next onset; a backwards EOF scan could also include unrelated closing speech.

The first skip/leakage fix used a separate alignment module; its logic is now
consolidated in `src/dataset_collection/segment_and_slice.py` (see the update
below). The following describes that earlier timestamp-wall pilot:

- Ordered full-text candidates, coverage/precision gates and adjacent opening/
  terminal anchors. A monotonic beam retains alternative occurrences; a missing
  reference does not advance a guessed cursor. Suspected cross-ayah matches are
  filtered **before** path selection, so rejection cannot consume the next ayah.
- Explicit full-prefix basmalah/istiʿādhah recognition, not a search for an isolated
  `الرحيم` (which also appears in Fātiḥah). Repeats and shared/overlapping word
  timestamp groups are reported; unsafe shared boundaries are not sliced.
- Waveform decay is limited by the **minimum** of the next observed speech onset,
  next known ayah start and source duration. An unmatched next ayah's timestamp
  still protects its onset. A rejected long/overlapping clip never shifts later
  starts. The final ayah uses its observed terminal, not an invented EOF endpoint.
- Blind per-clip second-pass ASR, enabled by default, rejects missing/extra speech.
  This is another decoding context with the **same** model, not independent
  acoustic certification. `--skip-clip-verification` is explicit and never grants
  approval. Clips over the default 30 seconds are listed for review, not trimmed
  to fit Whisper. `--max-clip-seconds` can permit longer review clips deliberately.
- Immutable source hashes, frame boundaries, sample rates and candidate WAV hashes;
  per-surah word transcripts, occurrences, selected candidates, rejected boundaries,
  clip-verification failures and **every unexported reference ID** in reports.
  An unexported ayah is uncertain, not proved absent from the recording.
- A non-empty output directory is refused. Existing datasets, audio and teacher
  edits are untouched. JSON/JSONL retain the existing collector fields with review
  metadata added; all outputs remain `usable_for_training: false` and require
  source authorization **and** listening/boundary approval.

Run a small pilot in a **new** output directory:

```powershell
python src/dataset_collection/segment_and_slice.py --input-dir taha_al_fahad_juz_amma/mp3 --output-dir data/segmentation_review/taha-my-review --model large-v3 --compute-type int8_float16 --surahs 1,78,112,113,114
python -m unittest src.dataset_collection.test_segment_and_slice -v
```

`int8_float16` reduces CUDA/model-loading memory; float16 failed to allocate
memory on the local machine before decoding. No smaller/substitute model was
silently selected. Install the optional review dependencies described above.

### Actual local-file pilot

The existing local Taha MP3 folder was processed; **no new audio was downloaded**.
Final-code run: `data/segmentation_review/taha-boundary-v3-pilot/`, CUDA large-v3
with int8/float16 and blind second-pass verification. **44 unapproved review WAVs**
out of 62 reference ayahs, zero training approvals:

| Surah | Review clips | Unexported Qālūn ayahs |
| --- | ---: | --- |
| 1 | 6 / 7 | 7 |
| 78 | 29 / 40 | 10, 14, 16, 22, 26, 28, 31, 32, 36, 39, 40 |
| 112 | 3 / 4 | 2 |
| 113 | 3 / 5 | 3, 4 |
| 114 | 3 / 6 | 1, 3, 4 |

Fātiḥah's final ASR word was `الضال`, not the full reference `الضالين`; ayah 7 was
not guessed or assigned EOF. Surah 78:31's second pass included `حدا` after the
target, an opening fragment of the next ayah, and was rejected (precision 0.75).
Other rejections include recognizer substitutions and `شكرا` hallucinations.
These do **not** prove the source audio itself has missing/incorrect verses;
listen and refine proposals separately instead of weakening gates to fill counts.

There are 18 slicing/alignment regressions covering merged audio, missing
terminals, repeated terminal phrases, long loose matches that could consume short
next ayahs, repeats/restarts, basmalah, long unmatched gaps, invalid/shared
timestamps, hard walls, rejected-long-clip continuation and trailing speech.

## Single-file implementation, shared madd boundaries and cleaning

All segmentation-specific logic now lives in **`segment_and_slice.py`**:
ordered reference matching, monotonic paths, timestamp validation, pause proposals,
joint boundary repair, waveform slicing, cleaning, blind ASR checks, review contexts
and reporting. Shared canonical Quran loading/normalization and geometry utilities
are still imported rather than duplicated. The separate alignment file was removed.

Every run creates **`review_bundle.json`**, the single place to inspect the entire
run: settings, source hashes, words/timestamps, matching occurrences, boundary
events/shifts, per-ayah decisions and reasons, cleanup measurements, clip checks,
candidate metadata and paths. Older JSON/JSONL/per-surah files remain for tooling
compatibility. Uncertain matches get wider raw `review_audio/*.context.wav` files,
explicitly **not** labeled whole-ayah training clips. References with no plausible
timing remain unassigned; their audio is not assumed absent.

### Why an ayah may be withheld

- No complete ordered ASR match (e.g. `الضال` instead of `الضالين`), corrupted
  timestamps, fusion evidence or a conflicting occurrence.
- Raw or cleaned second-pass recognition has missing/extra words, hallucinations
  or substitutions. This is a recognizer failure, not proof of incorrect audio.
- A boundary has no measured pause, has too little edge-word evidence, or crosses
  another boundary. Continuous wasl cannot be acoustically separated by an RMS
  heuristic; keep it for qualified alignment/listening, not invented timestamps.
- The complete clip exceeds 30s. It is reported, never truncated with a full label.

ASR sometimes writes final long /ā/ as `ا` instead of canonical `ى` (`سعا/سعى`,
`قلا/قلى`). This **comparison-only** spelling equivalence is now explicitly
audited. It does not rewrite source/labels, equate final `ي` with `ى`, or fold a
medial alif (`ملك` and `مالك` remain different). Other substitutions are not
silently accepted to fill missing counts.

### Madd tail fix

A Whisper next-word timestamp can lie **inside the preceding vowel**. The older
hard wall would then clip that vowel and put its remainder at the next clip's
start—even if both transcripts looked correct. The default `--boundary-mode
silence` now proposes a **shared cut inside a measured pause**, adjusting the
left end and right start together. It includes low-energy release and preserves
onset headroom. After the preamble or another repaired boundary, a one-word ayah
cannot borrow a pause before its own adjusted onset.

Search is bounded to the two edge words (default radius 1.75s, configurable up to
3s). A sustained gap needs voiced evidence and at least 100ms of low energy. A
quiet-run threshold/midpoint is still a heuristic: room echo, breath, internal
waqf and weak phonemes can confuse it. This is not Quran-aware phoneme forced
alignment or a certified madd-duration assessment. Raw/cleaned clip checks and
listening approval remain separate requirements. `--boundary-mode timestamp`
explicitly retains the earlier, weaker review-only behavior; it is not the default.

### Conservative noise-cleaning baseline

`--cleaning conservative` is the default:

- Remove DC offset and apply a zero-phase 40Hz rumble filter.
- Optional `--hum-frequency 50` or `60` applies a narrow notch **only when that
  mains frequency is actually observed**, not automatically guessed.
- Learn stationary noise only from sustained quiet regions, excluding their
  edges; smoothed, bounded spectral-subtraction attenuation (maximum 9dB). With
  insufficient quiet evidence, skip spectral denoising and report that fact.
- Preserve sample count/grid. **No VAD cropping, tempo change, compression, AGC,
  edge fades or automatic silence removal.** Boundary proposals use the original
  waveform, not denoiser-created silence. Clip verification checks both raw and
  cleaned versions. `--cleaning none` is an exact unprocessed-copy option.
- `audio/` holds cleaned candidates, `audio_raw/` matching raw cuts. Original MP3s
  and old exports are never overwritten. Frame boundaries, processing settings,
  clipping measurements and both candidate hashes are recorded.

These are common, conservative audio-processing practices—not a guarantee of an
"industrial-standard" certification or universally inaudible processing. Neural
enhancement can damage soft consonants/madd as well; introducing it would require
model provenance and raw/processed acoustic listening evaluation. Always A/B
listen to vowels, fricatives and pauses before admitting cleaned training audio.

```powershell
python -m pip install -r src/learning/requirements-review.txt
python src/dataset_collection/segment_and_slice.py --input-dir taha_al_fahad_juz_amma/mp3 --output-dir data/segmentation_review/taha-madd-clean-review --model large-v3 --compute-type int8_float16 --surahs 1,87,93
# Open output/review_bundle.json first; keep reviewer approvals elsewhere.
python -m unittest src.dataset_collection.test_segment_and_slice -v
```

For training initialization choices and authorization-gated Trabulsi admission,
see [TARTEEL_QALOON_TRAINING.md](TARTEEL_QALOON_TRAINING.md).

### Final shared-boundary/cleaning pilot (2026-10-04)

`data/segmentation_review/taha-madd-clean-v6-pilot/review_bundle.json` is the
single review entry. Existing local Taha recordings for surahs 79, 87 and 93 were
processed on CUDA large-v3/int8-float16. No new audio was downloaded:

| Surah | Raw/cleaned review pairs | Withheld Qālūn ayahs |
| --- | ---: | --- |
| 79 | 32 / 45 | 5, 6, 7, 9, 10, 16, 27, 29, 30, 31, 42, 44, 45 |
| 87 | 17 / 19 | 18, 19 |
| 93 | 9 / 11 | 6, 7 |

**58 pairs**, **75 explicit decisions**, **17 withheld references**, **11 wider
raw context WAVs**, **zero approvals**. Both raw/cleaned WAV hashes were verified.
The first stricter pause prototype is retained separately in the v4 pilot folder;
its rejected one-word boundary helped expose the preceding-pause selection bug.
The v5 pilot is also retained: a sample-grid audit caught a one-sample floating-
point round-trip overlap at one shared boundary. The final slicer clamps each
start against the previous **integer end frame**, with a regression for that case.

Examples in the final report:

- `93:1 والضحى`: end moves from the ASR estimate **5.900s** to the joint pause
  **6.8925s**; the next candidate starts at that same cut, not inside the vowel.
- `79:35 يوم يتذكر الانسان ما سعى`: end moves **0.915s later** to **202.905s**;
  left end and next start use the shared pause cut.
- `79:22 ثم أدبر يسعى`: end moves **0.985s later** to **118.625s**.

These examples pass raw and cleaned second-pass text gates and have waveform pause
evidence. They are still **proposals requiring listening**, not certified madd
durations or sample-accurate phoneme ground truth. Do not admit them to training
or learner playback based on normalized text agreement alone.
