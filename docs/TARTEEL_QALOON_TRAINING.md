# Tarteel Whisper-base/tiny Qālūn adaptation

## Recommendation

Run a **controlled comparison**, not a checkpoint substitution in the learner.
Tarteel initialization and full fine-tuning are not competing modes: fully
fine-tuning **all layers of Tarteel's Whisper-base** on Qālūn is supported here.

1. **OpenAI-base → full Qālūn adaptation**: the general-ASR initialization control.
2. **Tarteel-base → full Qālūn adaptation**: Quran-domain initialization, potentially
   useful for recitation acoustics, but potentially biased toward Hafs spellings/
   readings. All supervised targets remain our canonical Qālūn labels; no Hafs
   audio, labels or expected-text inference prompts are added.

There is no evidence yet that arm 2 is better. The repository's existing full-model
26.39% test WER is on its earlier four-reader experiment, not the new three-reader
protocol. Tarteel's advertised 5.7544 WER is on an unspecified upstream evaluation
set, **not comparable** to ours. Its model card leaves the training corpus and
limitations undocumented; pretraining overlap with our readers/verses is unknown.
Do not claim speaker-disjoint *pretraining* or general learner accuracy.

Checkpoint inspected:
`tarteel-ai/whisper-base-ar-quran`, revision
`5c3c53fdf9272c4f6ee0bee09a1e5a4a615ee25c` (model card labels Apache-2.0).
[Pinned model card](https://huggingface.co/tarteel-ai/whisper-base-ar-quran/blob/5c3c53fdf9272c4f6ee0bee09a1e5a4a615ee25c/README.md).
The implementation pins this revision, loads the matching processor, clears
legacy forced language tokens, and limits decoding to Whisper-base's real 448
positions rather than the old card/config's 1,024-token generation setting.
Its legacy checkpoint lacks modern language/task mappings; the trainer uses
**decoding metadata only** from pinned OpenAI-base revision
`e37978b90ca9030d5170a5c07aadb050351a65bb`, after validating tokenizer IDs.
No OpenAI weights are substituted when the Tarteel arm is selected. The OpenAI
comparison arm also pins that revision rather than following a moving branch.
The old Tarteel processor includes all text/control tokens but omits the 1,501
timestamp vocabulary entries. That specific trailing omission is supported by
suppressing those unused output IDs in transcription mode; other vocabulary/token
mismatches fail. No embeddings are resized or reinitialized. Evaluation supplies
explicit feature attention masks for padded audio.

## Entry point and scope

`src/training_with_gpu/train_tarteel_tiny.py` is the equivalent full-adaptation
entry for `tarteel-ai/whisper-tiny-ar-quran`, pinned to
`c3d7e624af5c81bef25a10a2af3a5a84cb4dd0f0`. It uses the same corpus and safeguards.
Use `--init-model openai/whisper-tiny` for the tiny control, pinned to
`169d4a4341b33bc18d8881c4b69c2e104e1cc0af`. All variants use the common multilingual
Whisper language/control metadata, not substituted weights. See the README for
complete base/tiny commands with or without admitted Trabulsi clips.

`src/training_with_gpu/train_tarteel_base.py` reuses the full-model trainer; defaults:

- **Dokali + Huthaify + Husary** only; Waleed is not automatically trained.
- Equal reader weights (Dokali 1.0), seed 42, normalized Qālūn ASR labels.
- Full learnable-layer adaptation, not LoRA, with checkpointing/early stopping and
  saved predictions. Whisper's fixed sinusoidal encoder positions remain fixed.
- Fātiḥah/Juz ʿAmma corpus; no long clip or augmented clip silently truncated at 30s.
- New empty output directory required. Existing model, deployments and private
  Hugging Face release are unchanged; the scripts do not upload or deploy models.
- `experiment_manifest.json` records initialization revision, hyperparameters,
  exact splits/labels/audio hashes, approvals for optional data, and smoke status.

Dry-run on the current three-reader corpus: **1,388 train / 141 validation / 174
test**. Four Husary recordings longer than 30s are explicitly excluded (78:40,
98:5, 98:6, 98:8), not chopped while retaining their full labels. These are
known-reader/held-out-ayah partitions, not unfamiliar learners.

```powershell
# Existing CUDA environment; install requirements without replacing its torch wheel.
python -m pip install -r src/training_with_gpu/requirements.txt
python src/training_with_gpu/train_tarteel_base.py --dry-run

# Arm 1 and arm 2: SAME reader selection, seed, sampling weights and settings.
python src/training_with_gpu/train_tarteel_base.py --init-model openai/whisper-base --output-dir runs/openai_base_three_reader_ab
python src/training_with_gpu/train_tarteel_base.py --output-dir runs/tarteel_base_three_reader_ab
```

Do not run both GPU jobs together. Compare generated validation/test WER/CER,
per-reader results, deletions/apparent-omission proxy, and failure samples, with
bootstrap intervals using the existing reporting tools. Use the same normalization
and split contents, and select settings on validation, not the final test set.
Review Qālūn/Hafs-sensitive words separately (e.g. `ملك` vs `مالك`); aggregate
unvowelled WER does not evaluate madd timing, ṣilah, harakat or tajweed correctness.

For **adaptation-speaker holdout**, explicitly reserve Dokali for validation and
Waleed for testing:

```powershell
python src/training_with_gpu/train_tarteel_base.py --validation-reciter dokali --test-reciter waleed --output-dir runs/tarteel_base_speaker_holdout --dry-run
```

This trains **Huthaify/Husary only**, not all three; the same command without
`--dry-run` launches that separate experiment. Held-out voices must not first
enter a checkpoint's Qālūn adaptation. Upstream Tarteel pretraining overlap still
cannot be excluded. Actual unfamiliar-learner evaluation remains required.

## Trabulsi: conditional implementation, currently NOT admitted

Good boundaries are **necessary but insufficient**. The earlier inspected
**Hugging Face collection** explicitly prohibits training under its unauthorized-
source terms. The **direct MP3Quran** recordings now verified separately have a
public reuse clause (below); do not conflate the two sources. No machine-only
clip is admitted now.
Do not flip machine-review flags or write a fictitious permission receipt.

Optional arguments are implemented:

```powershell
python src/training_with_gpu/train_tarteel_base.py --trabulsi-reviewed-manifest PATH_TO_HUMAN_REVIEWED_JSONL --trabulsi-permission-file PATH_TO_REVIEWED_SOURCE_PERMISSION_JSON --trabulsi-validation-report PATH_TO_COMPLETE_VALIDATION_JSON --dry-run
```

All three files are required. `src/training/reviewed_audio.py` fails closed unless:

- The source-level receipt names `reciter_key: "trabulsi"`, `training_allowed: true`,
  a real `rights_holder`, `grant_reference`, permission `reviewed_by/reviewed_at`,
  and the specific
  `authorized_source_sha256` list. A trusted person must verify the actual grant;
  a JSON assertion cannot itself confer legal permission. The restricted Hub
  source additionally requires `supersedes_unauthorized_source_restriction: true`;
  direct publisher evidence follows the separate policy below.
- A **complete strict WAV/text audit** identifies each selected candidate as
  passing technical AND exact normalized blind-ASR checks. Dataset-root, metadata
  and WAV hashes must still match; partial/technical-only audits cannot admit data.
- Each row has `alignment_status: "approved"`, `usable_for_training: true`,
  `authorization_status: "authorized"`, identified `reviewer/reviewed_at`, and
  `listening_approved: true`, `boundary_approved: true`. The segmenter never sets
  these approvals. Preserve teacher work separately from generated drafts.
- The immutable recording, candidate WAV hash, canonical Qālūn label, 16kHz mono
  PCM16 format and exact frame count all validate; paths cannot escape the dataset.
- No duplicate audio is present. All cuts/repeats of one source recording share
  a partition, preventing train/test leakage from full-surah cuts. This may result
  in different counts from an ayah-only split; the manifest makes that explicit.

Do not use the pre-existing unreviewed Trabulsi dataset folder as an automatic
training source. Authorization and acoustic review are separate release gates.

### Direct MP3Quran source

Verified publisher page: <https://www.mp3quran.net/eng/trabulsi-qalon>.
The `r1` recording links on that page identify the Qālūn collection; do not
substitute the separate Hafs page. All **38/38** existing local MP3s matched the
publisher's bytes, including `112.mp3`; evidence and saved policy HTML are at
`data/review_sources/mp3quran-trabulsi-direct-v1/`.

<https://www.mp3quran.net/eng/privacy> states:
> All rights are available to everyone, and we allow any visitor or developer to
> copy any material or use any link on the websites

This is affirmative public-material reuse evidence, not merely public playback.
It is not a standard SPDX license or an explicit ML/model-redistribution clause.
A responsible reviewer records the intended training-use decision, relying on
that real policy rather than fabricating a personal rights-holder grant. Public
model/audio release still requires its own licensing/provenance review.

A direct-source receipt has the following fields (**schema, not a granted receipt**;
complete it only after actual review):

```json
{
  "reciter_key": "trabulsi",
  "source_provider": "mp3quran-direct",
  "training_allowed": false,
  "rights_holder": "MP3Quran publisher public reuse policy",
  "grant_reference": "https://www.mp3quran.net/eng/privacy#Copyrights",
  "reviewed_by": "",
  "reviewed_at": "",
  "source_inventory": "ABSOLUTE_PATH_TO_VERIFIED_source_inventory.json",
  "source_inventory_sha256": "SHA256_OF_THAT_INVENTORY",
  "authorized_source_sha256": []
}
```

After review, set `training_allowed` to the actual decision, identify the reviewer/
date and list **only** covered full-recording hashes from the inventory. The loader
checks the inventory hash, Qālūn publisher/policy URLs, reuse evidence, byte identity,
recording hashes and per-row source URLs. It does **not** carry a Hub-collection
restriction over to separately sourced publisher recordings.

To inspect receipt inputs in PowerShell (this does **not** approve training):

```powershell
Get-FileHash dataset_qaloon_trabulsi_repaired/source_inventory.json -Algorithm SHA256
$inventory = Get-Content dataset_qaloon_trabulsi_repaired/source_inventory.json -Raw | ConvertFrom-Json
$inventory.sources | Select-Object surah, source_sha256, source_url, byte_identical
```

Current repaired candidates: `dataset_qaloon_trabulsi_repaired/`; completed audit:
`data/segmentation_review/trabulsi-repaired-validation-v2/validation.json`.
Use that audit's `listening_decisions.csv` for actual listening review. **302/333**
candidates passed automatic checks; no human approvals/accepted training manifest
or permission receipt have been manufactured. The remaining 236 scoped references
are preserved as withheld decisions, not disguised as an all-ayah dataset.

**Superseded boundary assurance:** a subsequent listener identified a real `طوى`
leak at 79:16→17 in that earlier v3 folder. Its automatic pass count is historical,
not acoustic approval. Regenerate with the current terminal-protected v4 slicer;
review the [updated completeness/boundary contract](AUDIO_SEGMENTATION_REVIEW.md#terminal-word-leakage-and-the-569-row-contract).
Never feed `coverage.jsonl` to training: its 569 rows include unresolved null-audio
entries, while training requires real, separately reviewed/validated WAV metadata.

Human listening decisions are collected separately in the validator's blank CSV.
`approve_ayah_audio.py` accepts only explicitly approved, passing, hash-matched rows
and checks the same permission/admission rules as training. Failed staging files
remain `.pending`, never an accepted manifest. No automatic approvals are generated.

With Trabulsi added, the existing other-reader ayah split and Trabulsi recording-
group split form a **mixed protocol**, not an unseen-ayah benchmark. Use explicit
speaker holdout and genuinely new learners for stronger evaluation; upstream
Tarteel overlap remains unknown.

## Implementation verification (2026-10-04)

- Actual **tiny** CUDA one-step training/save/decode smoke run passed in
  `runs/tarteel_tiny_qaloon_smoke_v1/` with the same six-per-partition three-reader
  smoke subset. **115 Python tests passed**, including exact partial/extra-word
  detection, WAV format/edge checks, decoder fallback and strict/direct-source
  admission. Neither smoke run is a comparative accuracy result.

- Three-reader and speaker-holdout dry-runs passed with the counts/protocols above.
- Actual CUDA full-parameter **one-step smoke run** completed in
  `runs/tarteel_base_qaloon_smoke_v4/`, using six clips per partition (two per
  reader), saving the model/processor, experiment manifest and decoded predictions.
  `is_smoke_run: true`. No full seven-epoch experiment or comparative benchmark was
  launched; tiny smoke-set scores are **not accuracy evidence** and are not used
  to claim superiority or replace the current deployed model.
- Earlier legacy-generation failures were retained separately, not overwritten.
  The old missing-key warning for tied `proj_out.weight` during checkpoint reload
  is expected for shared embedding weights; it is not a separately trained layer.
- Authorization/admission tests reject machine-only approval, absent/restricted
  rights, unrelated recording hashes, changed/duplicate WAVs, wrong labels, path
  traversal and mismatched frames; recording groups remain in one partition.
- Private model publication/deployment and existing experiment directories remain
  unchanged. Publication still requires provenance/licensing/security review.
