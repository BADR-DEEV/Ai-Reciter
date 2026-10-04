# DeepDML Qaloon upgrade: screenshot-inspired, independently measured

The photograph reports **2.64% Waleed WER / 100% Qaloon-word fidelity / .51s per
28s window**. We have no checkpoint, transcripts, exclusion list, normalization
definition, precision settings or timing protocol from that run. These are
**unverified screenshot benchmarks**, not measurements reproduced in this repo.
No code change can promise to beat them before a frozen evaluation.

## Corrected foundations

The supplied `deepdml/whisper-base-ar-quran` name did not resolve publicly. The
verified public normalized-text Quran checkpoints are:

| Recipe | Model | Pinned revision |
| --- | --- | --- |
| Base | `deepdml/whisper-base-ar-quran-mix-norm` | `23758f3e6877ce46077962c2e31a14794e700f1f` |
| Small | `deepdml/whisper-small-ar-quran-mix-norm` | `2d0929ee4d62cf3642e4184c16e9cb939d86ea78` |

Their public model cards cite EveryAyah/EA-UD training and incomplete usage/data
details. Waleed overlap upstream is **unknown**, even though our adaptation never
fits him. The new recipe rejects OpenAI/Tarteel weight substitution. OpenAI's
pinned **generation metadata template** is only a checked control-token mapping,
not initialization weights. The screenshot does not identify which DeepDML variant
Claude used; these are verified candidates, not an asserted reproduction.

## 1. Normalizer and immutable label repair

`normalize_quran_for_asr` now distinguishes a **pronounced yaa carrying fatha**
from a silent dagger-alif seat *before stripping the vowel evidence*:

| Source | V1 bug | Corrected |
| --- | --- | --- |
| `ٱلْقِيَٰمَةِ` | `القامة` | `القيامة` |
| `ءَايَٰتِ` | `ءاات` | `ايات` |
| `يَٰٓأَيُّهَا` | `اايها` | `يا ايها` |
| `أَتَيٰكَ` | `اتاك` | `اتاك`, preserved silent-seat case |

Standalone initial hamza+alif is handled like initial آ; medial hamzas and final
alif/maqsura distinctions remain. No final-alif folding or fuzzy acceptance is
introduced. Every one of the **6,210 canonical ayahs** is checked for idempotence.
**1,306 canonical ayah normalizations change** under these repairs; this is not
the screenshot's differently defined “840 words” count.

`label_integrity.py` creates source-bound **overlays**, preserving original PCM,
metadata, old labels and historical checkpoints. **83 labels change** across the
current adaptation+Waleed inventory. Waleed Fatiha uses its real row source text
to respect its different audio numbering and merged final segments.

The canonical loader's `text_asr` field and merged/Fatiha builder paths now also
derive labels directly from intact `raw` text and record the normalizer version.
Legacy simplified display text is not reused as an ASR reference. Historical
comparison and recovery callers explicitly choose their frozen normalizer/decoder.
The vocative boundary tests include medial `خَطَٰيَٰكُمْ`, which must stay one word,
not be mistaken for a vocative after a dagger-alif annotation.

V1 normalization stays separately callable. Frozen historical scores remain
unchanged and explicitly versioned; they cannot be compared directly with new
V2-normalized references. Supplemental legacy machine-pass audits remain labelled
V1 audits; changed fitting labels require a new exact-content check or quarantine.

Live tracking derives expected normalization from intact source text when present,
instead of trusting stale cached `normalized` fields. Its word alignment now uses
**exact ordered one-to-one matches**, not spelling fuzziness that could silently
color Hafs `مالك` as correct Qaloon `ملك`. This still measures ASR agreement, not
acoustic pronunciation certification.

The streaming loader now supports a saved pinned adapter **only with explicit
private-staging opt-in** (`RECITER_ALLOW_EXPERIMENTAL_ADAPTER=1` and an explicitly
chosen `RECITER_MODEL_PATH`). The existing default full-model path is unchanged.
It stays blind, verifies saved adapter controls and abstains on silence/loop/EOS
failures rather than forwarding hallucinated words to the learner tracker. The
decoder uses the actual 448-position context instead of silently capping a long
recitation at 180 new tokens. This capability is **not a deployment approval**;
no server was started with experimental weights. Staging FP16 must be benchmarked
as FP16, not compared to a float32 score/latency without disclosure.
Practice endpoints return an explicit **non-scoring retry** on unreliable/empty
ASR: no zero learner score, false missed words or forced-choice empty-list crash.

## 2. Five readers and real quarantine—not an invented list

Claude's `label_audit_flagged.json` is unavailable. The new fitting-only blind
audit is saved at **`data/segmentation_review/deepdml-training-audit-v1/`**. It
never opens Waleed audio or adaptation validation/test audio. It records exact
predictions and flags boundary insertions/deletions, unstable decoding and changed
supplemental references failing fresh agreement. Riwayah substitutions alone do
not justify discarding difficult Qaloon examples.

The actual **27 machine-quarantined fitting clips** are:

| Reader | Quarantined fitting clips |
| --- | ---: |
| Dokali | 20 |
| Huthaify | 4 |
| Husary | 2 |
| Trabulsi | 1 |
| Taha | 0 |

These are **conservative suspects, not proven corruptions or listening approvals**.
The screening model is also the initialization, so screening-induced selection
bias is disclosed. Unfiltered validation/test are retained. Never claim to have
reproduced Claude's 28 exclusions. The complete fitting audit inventory, PCM
hashes, normalized references and exclusion reasons must match before a run starts.

The admitted pre-quarantine pool has **2,222 clips**: three core readers plus the
already source-recorded **256 Trabulsi / 263 Taha** passing candidates. After
quarantine: **1,382 train / 412 validation / 401 test = 2,195**. The 568 Waleed clips
are separate, not added to reach a desired training count. Producing 2,400+
clean clips requires genuinely more passing cuts; withheld clips are not promoted
to fill the number. Existing private source-use decisions and production permission
gates remain unchanged; these candidates are not a public release approval.

## 3. Adaptation and 12GB CUDA settings

`train_deepdml_lora.py` supplies the requested recipe through the shared trainer:

The requested **`train.py` / `evaluate.py` CLI entry points now dispatch to this
recipe / frozen benchmark**, instead of silently starting vanilla OpenAI weights
and choosing by loss. Their old implementations remain explicitly named
`legacy_main` for historical inspection, not as the normal CLI or recommended
research recipe. New CLI arguments are those shown below (`--output-dir`,
`--quarantine-manifest` for fitting; `--run` for frozen evaluation).

- Rank **32**, alpha **64**, dropout .1.
- All six requested linear families: **q_proj, k_proj, v_proj, out_proj, fc1, fc2**
  across attention/FFN blocks (not sinusoidal embeddings or the tied output head).
- Base has **4,325,376 trainable adapter parameters**; seeded initialization.
- **BF16**, gradient checkpointing, batch **2 × accumulation 8**, eval batch 2;
  actual device confirmed **RTX 5070, 12 GB**. Small uses the same safe defaults.
- Training-only noise probability **.12**, measured **25–35 dB SNR**, and
  **pitch-preserving** phase-vocoder tempo .95–1.05, probability .2.
- Slow augmentation that would exceed 30s is **rejected**, not cropped. Originals
  and validation/test signals receive no augmentation. Noise cannot add clipping.
- Default LR **1e-5**, max three epochs, patience two, saves/evaluations every 39
  optimizer steps, selection on **generated macro-reader development WER, beam3**.
- `train_base_full.py` also now supports corrected overlays, true tempo and
  generated macro-reader selection rather than teacher-forced loss.
- Matching pinned processor and generation config are saved; real adapter reload
  must agree on fixed development fixtures. Waleed evaluation is a **separate**
  fixed benchmark after selection, never an in-training callback or stopping signal.

## 4. Constrained decoding without hiding learner mistakes

`surah_vocabulary.py` implements a **token prefix trie for whole canonical words**,
including initial versus space-prefixed BPE forms and terminal-prefix ambiguity.
It permits arbitrary word order, repeated words/restarts and early EOS at word
boundaries; it does **not force a complete expected ayah sequence**.
The trie admits source-attested forms and **exact-normalizer-equivalent alif/hamza
carrier spellings**, otherwise it would incorrectly mask acoustic decoder outputs
such as `ألم` and `النبإ` merely because scoring folds their alif carriers. It never
adds an alif to change `ملك` into `مالك` or folds medial hamza to grant Hafs `كفوا`.

It is optional and reported as **surah-vocabulary-assisted**. The known surah is
an oracle input and affects recognition. A complete accepted output is confined
to the allowed normalized word grammar; context-limit failures remain flagged.
This **cannot mathematically prevent hallucinating an absent Quranic word/ayah**
or changing a learner's noncanonical pronunciation to a valid word. Therefore:

- **Official WER and omission diagnostics use blind, unconstrained decoding.**
- Assisted scores are never substituted for blind scores or learner grading.
- Raw output, flags, coverage, all WER errors and both modes remain auditable.
- No fuzzy label matching, expected-text prompt, dropping outliers, arbitrary
  60-token cap or retroactive test-driven settings selection.

## 5. What the benchmark now measures

`benchmark_reciter.py` evaluates an unchanged selected adapter or pinned DeepDML
control on exactly the frozen WAVs/references and records:

- WER/CER, exact ayahs, insertion/deletion/substitution and reader disparity,
  cluster intervals, text-seen/unseen strata and decoding abstention flags.
- **Explicit lexical fidelity** with denominators for two local contrast sites:
  Fatiha `ملك` vs `مالك`, and Ikhlas `كفؤا` vs `كفوا`. No eligible occurrences means
  **null**, not manufactured 100%. This is not full-Qaloon vowel/madd/tajweed fidelity.
- Synthetic whole-ayah-skip probes made from **complete A/C clips with B absent**;
  ASR labels remain the actually spoken A+C, while skipped B is separate exercise
  metadata. The report detects invented full B phrases and scores actual speech.
  These do not establish sensitivity to real learner word omissions.
- Synthetic two-complete-clip **repeats**, with both actual renditions in the ASR
  reference (never deduplicated), plus seeded **28s silence/noise controls** with
  empty truthful speech references and an explicit nonempty-output rate. Raw-model
  noise hallucinations are measured without pretending the streaming RMS gate is
  a validated voice detector. Natural restarts and real learner noise remain untested.
- Actual **28s voiced timing-only windows**, batch 1, beam3, synchronized CUDA,
  warmup excluded, p50/p95 for feature extraction+transfer+generation. No quality
  or training label is attached to cropped timing windows. Model-load/network/UI
  time is excluded. This differs from unknown screenshot timing scope.

Waleed is adaptation-held-out but **previously observed**. New results on him are
descriptive, not fresh confirmatory evidence; no search loop stops when 2.64% is
beaten. A new sealed, consented learner/recording test and qualified listening
review are still needed before broad human-use or omission-grading claims.

## Commands and status

```powershell
# Audit only fitting clips; existing evidence directories are never overwritten.
python src/training_with_gpu/audit_training_inputs.py --output-dir data/segmentation_review/NEW_AUDIT
python src/training_with_gpu/train_deepdml_lora.py --output-dir runs/NEW_BASE_RUN --quarantine-manifest data/segmentation_review/deepdml-training-audit-v1/quarantine.json --dry-run
# Same command without --dry-run fits base. --init-model below selects small.
python src/training_with_gpu/train_deepdml_lora.py --init-model deepdml/whisper-small-ar-quran-mix-norm --output-dir runs/NEW_SMALL_RUN --quarantine-manifest data/segmentation_review/deepdml-training-audit-v1/quarantine.json
# Development assistance ablation is separate, not the learner score.
python src/training_with_gpu/benchmark_reciter.py --run runs/NEW_BASE_RUN --output-dir runs/NEW_BASE_DEVELOPMENT_BENCHMARK --assisted
# After checkpoint/recipe selection is frozen, evaluate Waleed once.
python src/training_with_gpu/benchmark_reciter.py --run runs/NEW_BASE_RUN --partition unheard_adaptation_reciter --output-dir runs/NEW_BASE_WALEED_BENCHMARK
```

**204 Python tests pass**, including the final endpoint/evaluator and CLI-dispatch regressions.
Tests cover corpus-wide idempotence, raw-source label
overlays, no Waleed fitting, quarantine hash binding, pitch/SNR measurements,
trie word-boundary/repeat/skip behavior, exact streaming matches and prior regressions.
The real DeepDML tokenizer trie smoke check passes. Base's one-step CUDA
train/save/reload smoke run passes at `runs/deepdml_qaloon_lora_base_smoke_v1/`;
small's corresponding CUDA smoke also passes at
`runs/deepdml_qaloon_lora_small_smoke_v1/` (**12,976,128 trainable parameters**).

The first two normalized-spelling-only assistance smoke benchmarks regressed
severely and are preserved under the base smoke folder (`assisted_benchmark_fp16`,
`assisted_benchmark_fp16_v2`). After admitting exact-normalizer-equivalent carrier
forms, `assisted_benchmark_fp16_v3` matched the ten blind smoke predictions with
no flags. This is a **compatibility test**, not meaningful accuracy evidence:
only one lexical contrast occurrence was eligible, and no adjacent skip fixture
was eligible in the ten-clip sample. Its FP16 beam3 timing p50 was **1.144s**,
p95 **1.255s** per 28s window—**not** the screenshot's .51s target. Word-level
cycle diagnostics additionally detect loops that can evade raw-token cycle checks.

The fixed full base experiment **completed** at `runs/deepdml_qaloon_lora_base_v1/`.
Early stopping ended at **step 117 / epoch 1.35**, selecting **checkpoint 39 /
epoch .45** by generated macro-reader validation WER. The export's adapter-weight
hash exactly matches that selected checkpoint; eight fixed normalized development
predictions/audio hashes matched after actual reload. Peak CUDA allocated memory
reported by the run was **.791 GB** (not total system/driver reserved VRAM).

| Same corrected-reference, blind beam3 protocol | Clips | WER | CER |
| --- | ---: | ---: | ---: |
| Unadapted DeepDML base, validation | 412 | 3.117% | .989% |
| Selected rank32/all-projection adapter, validation | 412 | 3.242% | .964% |
| Selected adapter, adaptation test | 401 | 3.903% | 1.503% |

The adapter is **not an established gain over its foundation**: validation has
52 versus 50 word errors / 1,604 words. The paired validation WER difference is
**+.125 percentage points**, with a surah-clustered 95% interval of **−.153 to
+.586 points**, including zero. Source: `selected_checkpoint_audit.json`.
Selected validation/test decoding had **zero cycle/EOS flags**, but later training
was unstable: step78 validation WER **146.51%**, step117 **67.71%**, despite loss
decreasing .747 → .354 → .244. Step117's fixed clean-training probe included one
word-cycle/no-EOS decode. Those histories are preserved, not hidden or assigned
an unproven cause. Loss-based stopping would have missed this regression.

Adaptation-test reader WER: **Dokali 9.58% (101), Husary 2.80% (101), Huthaify
3.04% (101), Trabulsi 0% (50), Taha 0% (48)**. The supplemental zeros describe small,
pre-screened professional sets, not broad voice/learner certification. No matching
unadapted test control was run, so no adaptation-test gain is asserted.

The serial frozen benchmarks **completed**:

| Blind Waleed, same 568 corrected references / float32 beam3 | WER | CER |
| --- | ---: | ---: |
| Selected adapter | **4.617%** | **2.632%** |
| Unadapted DeepDML foundation | 4.831% | 2.674% |

This does **not beat 2.64%**. Neither output had decoding flags on these professional
clips, but both missed the **two eligible lexical contrast occurrences (0/2)**.
This sparse lexical result must not be replaced by an assisted-decoder fidelity
claim. Waleed remains descriptive after prior exposure, never fitting/selection.

On development only, FP16 blind WER was **3.242%**, versus **1.122% assisted**;
lexical contrast fidelity was **4/5 blind versus 5/5 assisted**, which can be forced
by the supplied surah vocabulary and is NOT pronunciation evidence. No assisted
Waleed score or test-driven recipe change was made.

The diagnostics expose remaining problems: all 12 synthetic whole-ayah-skip
probes avoided inventing the full skipped phrase, but ten synthetic repeat probes
scored **51.35% blind WER** (50% assisted); raw blind inference hallucinated text
on **all four silence/noise controls**, with no decoder flags. Assisted noise also
produced nonempty output on 4/4 controls, all flagged. These controls bypass the
server's silence checks and do not establish natural-learner omission sensitivity.
Default inference flags and a surah trie are **not enough for safe learner grading**.

Selected adapter 28s timing p50/p95: **1.276/1.343s float32**, **1.277/1.471s FP16**;
foundation float32 p50 was .717s. None establishes the screenshot's .51s target.
Evidence is in `waleed_blind_benchmark_v1`, `unadapted_waleed_blind_benchmark_v1`
and `development_assistance_fp16_v1` under the completed base run. Original training
metrics remain unchanged; these are separately saved evaluations.

## Selecting the server model

```powershell
python -m src.streaming.serve --model gpu-full-base
python -m src.streaming.serve --model deepdml
# Optional faster greedy decoding; may change recognition accuracy:
python -m src.streaming.serve --model deepdml --beams 1
```

The launcher chooses the full model at `runs/gpu_base_full` or the actual saved
DeepDML adapter at `runs/deepdml_qaloon_lora_base_v1/adapter` before starting one
uvicorn worker. CUDA uses FP16 by default; `--device`, `--dtype`, `--port` and
`--model-path` are available. No silent model fallback, retraining, weight mutation,
or permanent background server occurs. Selecting DeepDML explicitly opts into
private staging; it does not assert production readiness. Both presets passed a
real CUDA/FP16 server-engine transcription check; selection/server regressions pass.

**No claimed new Waleed WER, 100% fidelity, .51s latency or production readiness**
follows merely from this implementation. No deployed model, public Hub release,
commit or push was changed.
