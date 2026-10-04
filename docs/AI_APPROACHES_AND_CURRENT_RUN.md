# Rattil: two AI approaches and the current private LoRA run

## What actually trained

**Later requested repair:** see [the decoding/target-format audit and five-reader
v2 recipe](TARTEEL_DECODE_REPAIR.md). The user subsequently directed private
research with Taha; that new decision retains unverified rights status and does
not rewrite v1 membership or grant production approval. This page's v1 recipe
and scores below remain a historical record.

The five-reader v2 fit/export reached early stopping; evaluation-only recovery
completed after a post-export process exit. See [v2 results](TARTEEL_LORA_V2_RESULTS.md):
**3.80% test WER / 5.49% Waleed WER**, without flagged loops/EOS failures, but only
one fewer development word error than unadapted Tarteel. No deployment or
human-learner certification follows from these professional-recording results.

**Completed:** v1 early-stopped after epoch 3 and restored epoch 1. Its test WER
is **61.46%**; Waleed held out of adaptation scored **9.43%**. It is **not a release
candidate**. See [completed results and failure analysis](TARTEEL_LORA_V1_RESULTS.md)
for the same-audio comparison, repetition failures and overlap limitations.

The user changed the initial full-fine-tuning request to **LoRA on Tarteel-base**.
The launched run is **`runs/tarteel_base_lora_private_v1/`**; it is not deployed,
uploaded or approved for release. It starts from the pinned original
`tarteel-ai/whisper-base-ar-quran` revision
`5c3c53fdf9272c4f6ee0bee09a1e5a4a615ee25c`, not the converted segmentation model.

| Setting | Actual choice |
| --- | --- |
| Training voices | Huthaify, Dokali, Husary, direct-publisher Trabulsi |
| Taha | Rebuilt and audited, **excluded from train/validation/test pending source-use clearance** |
| LoRA | rank 16, alpha 32, dropout 0.1, attention q/v projections |
| Learnable parameters | **589,824** (about **0.806%** of adapter+base parameters) |
| Learning rate / effective batch | 5e-5 / 16 (4 × accumulation 4) |
| Epoch limit / early stopping | 6 / patience 2 on **generated macro-reciter validation WER** |
| Regularization | cosine schedule, 10% warmup, weight decay .01, gradient clip 1, mild noise/speed augmentation |
| Reader sampling | inverse reader-count balancing; supplemental reader relative weight .5 |
| Selection partitions | **1,241 train / 365 validation / 353 test** |
| Unheard adaptation reader | **Waleed, 568 clips**, never used for fitting/selection/stopping |
| Safety | complete WAVs ≤30s; >30s targets withheld, never truncated with full labels |

```powershell
python src/training_with_gpu/train_tarteel_lora.py --output-dir runs/NEW_PRIVATE_RUN --supplement src/dataset_collection/dataset_qaloon_trabulsi src/dataset_collection/dataset_qaloon_trabulsi/experiment_decision.json --baseline-model-dir runs/gpu_base_full --epochs 6 --patience 2 --rank 16 --learning-rate 5e-5 --batch-size 4 --gradient-accumulation 4 --eval-batch-size 4
# Add --dry-run before deliberately launching. Do not reuse a nonempty run dir.
```

Outputs: frozen source/clip-hash `experiment_manifest.json`, best checkpoints,
`adapter/`, `generalization_history.json`, `training_log.json`, full metrics and
sample predictions. At completion the existing full model is evaluated on the
**same test WAVs**, producing `comparison.json` with contamination caveats and
paired bootstrap error differences. Final metrics and comparisons are now saved;
a successful run or one-step smoke test is not accuracy improvement evidence.

**Reproducibility caveat for v1:** its adapter was initialized before Trainer set
the seed. The weights/checkpoints are preserved, but a fresh run may not
recreate that exact random adapter start. The entry point now seeds before model
construction and records `adapter_initialization_seeded: true` for future runs.
GPU kernels are not claimed bitwise deterministic. This does not justify quietly
restarting or selecting a new run on the held-out results.

### Data admission—not manufactured certification

The user explicitly accepted imperfect, incomplete Trabulsi candidates for a
private experiment. **256/289** pass the independent WAV/text audit; only those
256 enter the experiment pool. Missing IDs are ignored for corpus completeness,
not replaced with guessed audio. The 33 failed candidates remain outside training.

`experiment_decision.json` records actual user quality acceptance and a **broad
MP3Quran public-material reuse policy basis**, independently byte-verified against
the publisher. It does **not** claim a bespoke ML redistribution grant, create
teacher approvals or change source `usable_for_training: false` flags. The
restricted Hub Trabulsi collection is not used. The original production admission
path in `reviewed_audio.py` remains fail-closed and is not bypassed for releases.

The supplied Assabile page confirms free listening/downloads, but also says
**all rights reserved**. A public download is not by itself a training license.
Taha's 38 recordings were rebuilt into **302 candidates**, independently audited:
**263 automatic passes**, **39 review flags**, **267 withheld canonical references**.
They remain research/review assets, not members of this run's test set either.
To admit Taha later, supply an actual source-specific training grant with covered
recording hashes; the experimental loader supports that evidence separately.

### Leakage and honest generalization claims

Every **surah across every adaptation voice** shares one partition. Consequently
all ayah cuts from a full recording remain together, and no canonical ayah ID or
identical audio hash crosses train/validation/test. This is stricter than adding
recording-group Trabulsi cuts to unrelated existing ayah buckets.

Waleed is unheard **during this adaptation**, not necessarily during Tarteel's
unknown upstream pretraining. His evaluation separates text seen/unseen during
adaptation. One professional held-out reader cannot prove “any reciter”: unfamiliar
learners, children, women, accents, microphones, room conditions, long ayahs and
actual mistakes require new consented, teacher-labelled test material. This run
does not certify phonetics, Qālūn aḥkām, omissions or tajweed.

## Added analysis

- Micro WER/CER, exact-ayah accuracy, substitutions/deletions/insertions and their
  rates, empty output, extra-word and terminal-word mismatch diagnostics.
- Per-reader scores, **macro-reader WER**, worst-reader WER and reader disparity;
  long/medium/short clips and seen/unseen text strata.
- **Surah-clustered 95% bootstrap WER intervals**, not falsely independent
  confidence intervals over correlated ayah cuts. Few recording groups limit
  certainty; repeated runs/seeds are still needed.
- Synchronized generation-only batch latency p50/p95, inference real-time factor,
  peak allocated CUDA memory and trainable parameter counts. Batch latency is not
  advertised as single-user end-to-end latency.
- Fixed, clean, balanced **48-clip training probe** each epoch, alongside validation
  WER/loss, to expose the generalization gap. It does not use test audio. Augmented
  train loss and clean validation loss are not treated as directly equivalent.
- Best model restored by generated **macro-reader WER**, not only teacher-forced
  loss. Validation/test scores remain unweighted even when training samples differ.
- A frozen unheard-reader test, held back until the selected model is final.

Remaining evaluation work: multiple-seed controls, deterministic clean/noisy and
microphone robustness suites, confidence calibration on labelled acceptance/
abstention decisions, learner-specific false-alarm/miss rates, streaming progression
accuracy and latency. Do not invent these metrics without appropriate ground truth.

### Existing full model versus this run

| | Existing full fine-tune | New run |
| --- | --- | --- |
| Initialization | OpenAI Whisper-base | Pinned Tarteel Quran-base |
| Adaptation | all learnable layers | LoRA q/v adapters |
| Test WER / CER (different partitions) | historical **26.39% / 7.96%** | **61.46% / 27.71%** |
| Validation WER / CER (different partitions) | historical **28.74% / 7.20%** | **42.36% / 27.43%** |
| Protocol | earlier four-known-reader, ayah-held-out benchmark | recording/surah-held-out plus separate adaptation-held-out Waleed |
| Waleed | known adaptation voice | excluded from all adaptation/selection |

Those historical percentages **cannot be ranked directly** against the new test
split. The automatic paired comparison will flag reconstructed legacy training-
audio overlap; the old checkpoint lacks an embedded training manifest, so overlap
reconstruction is not proof of a clean control. Existing full weights have seen
Waleed. A causal comparison needs **fresh OpenAI and Tarteel initialization runs
on exactly the same frozen corpus/protocol/settings**; this entry also accepts
`--init-model openai/whisper-base` for that deliberate control. None is launched
implicitly, and neither existing model nor private release is replaced.

## Approach 1 — recitation assistant

Keep the responsibilities separate:

1. Immutable canonical Qālūn text/audio provenance and independent engineering QA.
2. Blind ASR, optional validated word/acoustic alignment, uncertainty and abstention.
3. Canonical text comparison, progression/window management and auditable feedback.
4. Teacher-reviewed lexical/route rules and performance assessment **only when
   suitable acoustic models and ground truth exist**.

Do not let a language model fill an inaudible Quran word or certify a madd from
transcription. Keep reference playback separate from live scoring. Provide an
easy “the model heard me incorrectly” correction path rather than penalizing a
learner as if every ASR mismatch were their recitation error.

## Approach 2 — personalized challenge policy

A learner model is **not another Whisper fine-tune**. It predicts what to review,
what support is useful and which valid distractors are appropriately similar.
No learner-event dataset exists yet, so claiming a trained personalized neural
model now would be misleading.

The initial **opt-in local prototype** in `adaptive-challenges.ts` is transparent:
start gently, require five recent choice observations, increase by one difficulty
level after steady success, back off after two misses, retain a bounded 12-answer
window. Apply a change **on the next question**, not while a learner answers.
Each new difficulty level requires fresh answer evidence instead of promoting
again from the same easy successes. It uses separate skill/scope/reader histories
under each local profile, writes history only while opted in, and removes that
profile's learner history when the profile is deleted. It does
**not** infer competence from XP, use uncertain ASR feedback as mastery evidence,
or penalize skipped/revealed answers. Manual difficulty remains available.

Increasing difficulty uses the existing truthful text/MFCC distractor ranking;
MFCC similarity is not measured phonetic or semantic equivalence. Aim for productive
70–85% success with review/hints, not maximal confusion, pressure or engagement.

Next stages:

- Consent-aware event schema: pseudonymous learner, item/skill IDs, difficulty,
  correct/incorrect, hints/reveals, timestamp, policy/version and item-selection
  probability. No raw learner audio/passwords in policy logs.
- Spaced review and an interpretable logistic/IRT or knowledge-tracing baseline;
  calibration/Brier/log loss, learner-disjoint splits and time-forward evaluation.
- Learn distractor difficulty from actual confusions, with Quran-text uniqueness
  checks and no fabricated/corrupted verses or recitation audio.
- Only after offline retention/calibration checks: an opt-in constrained contextual
  bandit with logged propensities, exploration caps, support after mistakes and
  rollback. Optimize delayed retention, not clicks or streak anxiety.

### Engaging additions worth prototyping

- **Contrast-pair practice**: two authentic similar ayahs, then a short explanation
  of the distinguishing source words; repeat later to measure retention.
- **Explain my next review**: show the weak skill and due memory interval, rather
  than an opaque difficulty jump.
- **Listen → recall → read** mini-sessions, with gradually reduced phonetic support
  and a teacher-reviewed beginner curriculum.
- **Personal confusion map** and optional teacher hand-off, separating confirmed
  learner errors from uncertain ASR results.
- **Retrieval-grounded commentary** with visible sources and numbering alignment,
  never uncited generated Quran wording or unverified aḥkām.

Browser-local profiles remain a prototype, not secure authentication. Production
personalization needs consent, data retention/deletion controls, access controls,
age-appropriate protection and qualified content review before collecting events.

## Verification of this change

- **143 Python tests passed**, including synthetic experimental admission,
  immutable relocation, leakage guards, paired metrics and existing regressions.
- **44 Playwright tests passed**; the **two live-GPU tests were skipped** during
  training. Policy tests cover opt-in storage, next-question difficulty changes,
  bounded/fresh evidence, profile isolation and deletion.
- Typecheck, production build and `git diff --check` passed.
- CUDA one-step LoRA smoke v3 trained, selected/saved an adapter and decoded the
  validation/test/held-out samples. It is marked smoke-only, not a benchmark.
- The first simultaneous large-ASR audit/LoRA smoke attempt hit host-memory
  allocation failure. Serial reruns succeeded; do not launch another large-model
  review job alongside a training run on this 16-GB host.
