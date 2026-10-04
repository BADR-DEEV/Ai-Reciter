# Five-reader Tarteel LoRA v2: stable private recognition candidate

**Historical normalization notice:** all scores below preserve their frozen
V1-normalizer protocol. A later source audit found consonantal-yaa/vocative label
bugs and adds V2-normalized immutable overlays plus a new DeepDML recipe. Do not
retroactively replace these numbers or directly compare them with corrected-label
scores. See [the DeepDML upgrade](DEEPDML_QALOON_UPGRADE.md).

**Outcome:** the selected saved adapter has usable reference-recitation recognition
results under the frozen beam3 protocol, with no flagged token loops/EOS failures
in the evaluated partitions. It is **not approved for deployment, public release,
pronunciation/tajweed certification or automatic learner grading**. No new fit,
checkpoint choice or decoder tuning is authorized merely by this result.

## Run and evidence

- Run: `runs/tarteel_base_lora_private_v2/`.
- Model: pinned `tarteel-ai/whisper-base-ar-quran`, revision
  `5c3c53fdf9272c4f6ee0bee09a1e5a4a615ee25c`.
- Readers: Huthaify, Dokali, Husary, Trabulsi and Taha.
- Pool: 256 passing Trabulsi candidates and 263 passing Taha candidates plus the
  three core voices; reader-balanced sampling, supplemental relative weight .5.
- **1,409 train / 412 validation / 401 test**; **568 Waleed** clips withheld from
  adaptation, checkpoint selection and stopping. Source-recording/surah groups
  share partitions, with canonical-ID/audio-hash separation.
- LoRA q/v projections, rank 16, alpha 32, dropout .1;
  **589,824 trainable parameters** (approximately .806%).
- LR **1e-5**, batch 4 × accumulation 4, seeded initialization, maximum three
  epochs, patience two on generated **macro-reader development WER**. Evaluation
  and saves every 39 optimizer steps. Mild noise/speed augmentation .05/.1.
- Canonical **Qaloon vowelled training targets**, not ASR-predicted/Hafs targets;
  unchanged stripped canonical references for WER. Blind Arabic transcription,
  beam3, no repetition penalty/trigram block, no expected-text prompt and no
  arbitrary 60-token cap. Full 448-position context; >30s clips withheld intact.
- Last fit step **156 / epoch 1.76**; selected checkpoint **78 / epoch .88**.
  Exported adapter's hash exactly matches the selected checkpoint.

The original process exited **code 1 after export and complete validation/test
prediction saves**, before Waleed/report finalization. No traceback was recorded;
the cause remains unresolved. It must not be described as an entirely successful
original process or assumed to be memory exhaustion.

An **evaluation-only fresh-process recovery** completed at:

`runs/tarteel_base_lora_private_v2/recovery_evaluation_v1/metrics.json`

Recovery verified frozen audio hashes, reused complete original validation/test
predictions and evaluated Waleed with the **unchanged saved adapter and decoding
configuration**. Eight development WAVs matched original raw/normalized predictions,
flags and token counts after reload. Original evidence was not overwritten;
missing original latency measurements were not fabricated. No retraining occurred.

## Results

| Partition | Clips | WER | CER | Exact normalized ayahs | Loop/EOS flags |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 412 | **3.69%** | **.89%** | 87.86% | 0 |
| Test | 401 | **3.80%** | **1.20%** | 88.28% | 0 |
| Waleed, adaptation-held-out voice | 568 | **5.49%** | **2.80%** | 86.27% | 0 |

All outputs, including any potential flagged outputs, remain in official metrics.
“No flags” does **not** mean “no recognition mistakes” or complete acoustic review.
The heldout WER's 95% surah-clustered bootstrap interval is **3.32–8.55%**. Its
**205 text-unseen clips** score **5.71% WER**. “Unheard” applies only to this
adaptation: upstream Tarteel voice/recording/text exposure is unknown.

### Test by reader

| Reader | Test clips | WER |
| --- | ---: | ---: |
| Dokali | 101 | **9.15%** |
| Husary | 101 | **2.82%** |
| Huthaify | 101 | **2.82%** |
| Trabulsi | 50 | **.49%** |
| Taha | 48 | **0.00%** |

The 0% Taha result is limited to **48 selected automatic-pass clips**, not a
whole-reader, whole-Quran or arbitrary-learner performance claim. The test contains
**49 substitutions / 15 insertions / zero deletions**; Waleed has **90 substitutions /
19 insertions / 19 deletions**. Dokali remains the highest-error adaptation voice.

Only **three test clips** are >15s. Long ayahs, legitimate repeated recitation,
microphone noise and actual learner mistakes remain inadequately tested. Zero
test deletions on professional recordings does not establish sensitivity to a
learner omitting a word.

## Did LoRA improve the original model?

The same-protocol unadapted Tarteel development control scored **3.75% WER /
.90% CER**, with zero flags. The selected adapter scored **3.69% / .89%**:
**one fewer word error (59 vs 60 out of 1,599 words)**. Macro-reader WER changes
from **3.11% to 3.07%**. The paired surah-bootstrap WER-difference interval includes
zero (**approximately −.129 to 0 percentage points**).

This supports **stable recognition and removal of v1's observed catastrophic
behavior under the new recipe**, not a meaningful proven accuracy gain from
LoRA. There is no matched unadapted control for this v2 test/Waleed report. The
five-reader v2 test differs from v1's four-reader test; learning rate, targets and
decoding changed together, so the improvement cannot be attributed to one fix.
The causal effect of any one change remains unisolated.

Prior test/Waleed audio is now observed: these are **descriptive regression
results**, not fresh confirmatory tests. Existing full-model comparisons also
contain reconstructed training overlap, as documented in the v1 report. Do not
substitute those contaminated comparisons for original clean-protocol metrics.

## Use and remaining release requirements

For a **private, consented <=30s recording test**, not a certified recitation grade:

```powershell
python src/training_with_gpu/test_my_audio.py --model-path runs/tarteel_base_lora_private_v2/adapter --audio YOUR_RECORDING.wav --beams 3
```

The matching pinned base, processor, saved generation config and diagnostics are
loaded. Near-silence or decoder failures abstain; uncertain ASR text differences
are not labelled “true omissions” or pronunciation errors. Longer recordings are
rejected rather than silently cut. Full transcripts and repeated speech are not
silently repaired from an expected ayah.

Before deployment, obtain consented recordings from unfamiliar **learners**, with
qualified labels for correct recitation, real omissions/replacements, repeats,
noise and long passages; validate recognition, false learner penalties and
abstention. Preserve a new sealed confirmatory partition. Qaloon vowels/tajweed
need qualified acoustic/content review beyond normalized spelling agreement.

Taha's private admission records **user-directed research with unverified source-use
status**, not a fabricated rights-holder license. Trabulsi retains its verified
publisher public-material reuse-policy basis, not an explicit ML redistribution
grant. Original audio approval flags and production permission gates are unchanged.
Public release/source clearance and existing security-review requirements remain.

**164 Python tests**, compilation and whitespace checks pass. Deployed weights
remain unchanged. The personal-audio CLI also passed a live GPU check on a fixed
development recording: complete vowelled transcript, emitted EOS and no decode
flags. This verifies the loading/decoding path, not unfamiliar-learner accuracy.
No commit, push, model upload or public release was performed.

See [the measured repair investigation](TARTEEL_DECODE_REPAIR.md) and
[preserved v1 failure results](TARTEEL_LORA_V1_RESULTS.md).
