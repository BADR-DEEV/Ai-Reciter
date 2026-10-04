# Tarteel decoding repair and the five-reader private v2 experiment

## Assessment of the supplied Gemini explanation

**Supported:** sustained autoregressive repetition dominates v1's measured errors;
teacher-forced loss can fall while free-running generation deteriorates. Beam
search and conservative optimization are reasonable hypotheses to test.

**Not established:** a missing EOS ID, audio bleed as the cause, ordinary
overfitting as the sole mechanism, or a “true 7%” repaired score. Runtime v1 already
used the explicit modern generation configuration, despite PEFT not saving it.
Both train-probe and validation recognition deteriorated. Several predictions
failed, not just one clip. Trabulsi/Husary/Waleed's scores were not all the lowest
observed scores in this project: the old model scored lower on Husary/Waleed in
the contaminated same-WAV comparison. Neither comparison proves causal gains.

Direct inspection of the pinned Tarteel checkpoint confirms:

- Decoder start **50258** (`<|startoftranscript|>`).
- Arabic/task/no-timestamps prefix **50258, 50272, 50359, 50363**.
- EOS/padding **50257** (`<|endoftext|>`); this tokenizer's stop token is not
  literally named `<|endoftranscript|>`.
- Correct targets begin with the transcript prefix and end with EOS. The collator
  removes BOS once, masks only padding, and now asserts that supervised EOS survives.

The audit records actual clean training-fixture loss, unadapted development scores
and decoder ablations. It **does not open test predictions or Waleed audio**.

### Measured development findings, not projected test gains

On v1's unchanged 365-clip development set:

| Model/decoder | Development WER |
| --- | ---: |
| Unadapted pinned Tarteel, greedy | **4.12%** |
| Unadapted pinned Tarteel, beam3 | **3.98%** |
| v1 adapter, greedy | 42.36% |
| v1 adapter, beam3 | **8.72%** |
| v1 adapter, beam3 + penalty 1.1 | 8.93% |
| v1 adapter, beam5 + penalty 1.2 + trigram block | **16.12%** |

Beam3 avoids the large word-loop failure without blanket repetition suppression,
but **v1 still regresses against unadapted Tarteel**. Its beam3 CER remains high
(20.25%), so word-only improvement is not a sufficient acceptance criterion.
These are development results, not a new “true 7%” test score.

Eight fixed training clips show a substantial target-style shift: mean initial
loss **8.45** for stripped-text targets versus **2.15** for canonical Qaloon vowelled
targets. Tarteel generates vowelled text. This supports preserving the canonical
vowelled target style rather than forcing a high-loss output-format change with
LoRA. It does not alone prove the cause of all instability. Model-predicted targets
were inspected only diagnostically and are **never used as training labels**.

The first safety audit mistakenly inspected Whisper's postprocessed tensor return,
which strips prefix/EOS, and therefore flagged every clip. That flags subsection
is explicitly invalidated in its preserved audit directory. The corrected API
requests a structured generation ModelOutput with full sequences and reruns to
`runs/tarteel_base_lora_v1_development_audit_v2/`; do not mistake stripped EOS in a
wrapper's return for a model's failure to emit EOS.

The corrected audit confirms **three flagged v1 clips** under greedy/beam3/soft
penalty decoding: all three contain token cycles and lack EOS. Beam3 lowers word
WER but does **not eliminate those failures**, which contribute to its high CER.
Unadapted Tarteel has **zero flags** under greedy/beam3. The trigram-block ablation
has zero flags but worse word recognition and **16 development deletions**, versus
7 for v1 plain beam3 and 2 for unadapted beam3. No outputs are removed from WER.

### Independent recording observations

The fixed-threshold PCM audit at
`data/segmentation_review/core-pcm-facts-v1/engineering.json` found:

| Reader | Energetic starts | Energetic ends | Possible clipping |
| --- | ---: | ---: | ---: |
| Dokali | 310 / 569 | 301 / 569 | 0 / 569 |
| Huthaify | 43 / 569 | 35 / 569 | 0 / 569 |
| Husary | 23 / 569 | 18 / 569 | 95 / 569 |

These are measured 20ms edge-energy/saturation flags, **not proven cut words** or
listening decisions. They support reviewing Dokali's boundaries but do not prove
he “poisoned training”; the same recordings decode well with unadapted Tarteel.
Husary should not be assumed pristine merely from the “studio master” description.
No audio was edited, no reader was automatically excluded and none was near-silent.

## What changed

- `decoding_safety.py`: pinned adapter+matching processor loading, generation-control
  verification, reproducible decoding profiles, EOS/context-limit and sustained
  token-cycle diagnostics. Real repeats may also flag, so a flag means **abstain
  from grading**, not “the learner made a mistake.” Raw output remains auditable.
- PEFT checkpoints and final adapters now include **`generation_config.json`** and
  **`decoding_policy.json`**. Actual adapter reload is tested on fixed development
  WAVs and predictions must agree before claiming successful export.
- Development-only comparison of greedy, beam3, beam3 with mild repetition penalty,
  and the proposed beam5/trigram-block/repetition-penalty decoder. **Trigram blocking
  is an ablation, not silently made the default**: legitimate repetitions/restarts
  and valid Quran token patterns can otherwise be suppressed.
- Keep the verified **448-position decoder context**, not a universal 60-token cap.
  Whisper uses subword tokens; a valid long ayah can exceed 60. No expected-text
  prompts, missing-word completion, cropping, guessed labels or outlier removal.
- v2 uses **1e-5 LR**, a maximum of three epochs, seeded LoRA initialization,
  generated macro-reader WER selection, early stopping and matching evaluation/
  checkpoint intervals every **39 steps**, rather than only once per epoch.
- v2's teacher-forced target is **canonical `normalized_with_harakat`**, not a
  Tarteel/Hafs transcript or a label inferred from audio. The original normalized
  ASR references remain unchanged for selection/comparison. Both targets come
  from the same original Qaloon source; mark-cleaning and stripped normalization
  do not commute for six core rows, so targets are checked against the original
  source, not silently used to rewrite the benchmark reference.
- The unadapted pinned Tarteel model is measured under the **same development
  decoder** before fitting. Reports show macro-WER/deletion changes and flags as
  well as micro WER/CER, reader disparity and the clean train-probe gap.
- `test_my_audio.py` now loads Tarteel adapters from their own pinned base/processor,
  retains all audio ≤30s, rejects longer recordings rather than silently truncating,
  rejects expected/preamble prompts, diagnoses decoding flags and abstains. Its
  sequence-aware comparisons are labelled **ASR text differences**, never “true
  omissions” or pronunciation findings from fuzzy spelling similarity.

Flags do not remove examples from official WER. Coverage/flag rate is reported
separately. Unflagged output is **not certified correct** and is not a tajweed grade.

## Five-reader corpus and source status

The user explicitly requested private training with all five voices and accepted
the incomplete corrected candidates. This research instruction is now recorded
without manufacturing rights-holder approval:

- Huthaify, Dokali, Husary core audio.
- **256 Trabulsi** strict automatic-pass candidates from the verified direct
  publisher collection; source policy evidence stays intact.
- **263 Taha** strict automatic-pass candidates. A separate decision records
  **`user-directed-private-research`** and
  **`unverified-user-asserted-open-access`**, not a verified ML license. It identifies
  the actual local recording hashes and Assabile source page.
- Failed-content/boundary candidates and missing references remain excluded.
  Merely lowering sampling weight never fixes incomplete audio or a wrong label.
- Production `reviewed_audio.py` remains unchanged; original `usable_for_training`
  and human-approval flags are not rewritten. **Public release remains disallowed**
  in the experimental decisions. Free downloads/reuse descriptions alone are not
  represented as an explicit ML/model redistribution grant.

Dry-run counts: **1,409 train / 412 validation / 401 test**, with **568 Waleed** clips
outside adaptation. Every adaptation voice's same-surah clips share one partition;
audio/source hashes and canonical ayah IDs cannot leak across those partitions.
Reader-balanced sampling keeps both supplemental voices at relative weight **.5**.
Dokali is **not** excluded merely because one model loops on his recordings.

The v1 test/Waleed sets are now observed; v2 results on them are **descriptive
regression results**, not fresh confirmatory evidence. A new sealed, consented,
qualified learner/reader test remains necessary for broad human-use claims.

## Commands and release restraint

```powershell
python src/training_with_gpu/audit_tarteel_development.py --run runs/tarteel_base_lora_private_v1 --output-dir runs/NEW_DEVELOPMENT_AUDIT
python src/training_with_gpu/train_tarteel_lora.py --output-dir runs/NEW_FIVE_READER_RUN --supplement src/dataset_collection/dataset_qaloon_trabulsi src/dataset_collection/dataset_qaloon_trabulsi/experiment_decision.json --supplement src/dataset_collection/dataset_qaloon_taha src/dataset_collection/dataset_qaloon_taha/experiment_decision.json --learning-rate 1e-5 --epochs 3 --eval-steps 39 --decode-profile beam3 --label-field normalized_with_harakat
# Add --dry-run to validate membership without fitting. Never reuse a nonempty run.
python src/training_with_gpu/test_my_audio.py --model-path runs/NEW_FIVE_READER_RUN/adapter --audio YOUR_CONSENTED_WAV --beams 3
```

Run GPU audits/training serially on this 16-GB host. Previous v1 evidence and
existing deployed weights remain unchanged. Improved development metrics alone
do not authorize deployment: validate stability, long ayahs, legitimate repeats,
noise/silence, unfamiliar speakers, abstention and genuine learner mistakes first.

## Verification and operational status

The five-reader admission dry-run passes with the counts above. **164 Python
tests pass**, including BOS/EOS padding, preserved valid repeats, token-cycle
abstention, source uncertainty, vocalized canonical targets, non-truncating personal
audio and prior regressions; compilation and `git diff --check` pass.

The corrected full-sequence development audit completed. The GPU five-reader v2
smoke run at `runs/tarteel_base_lora_private_v2_smoke/` passed training, checkpoint
selection, generation-config export and real adapter reload: **eight fixed
development WAVs produced identical predictions**, with zero development decode
flags in its ten-clip smoke partition. One step is not accuracy improvement evidence.

The full five-reader fit at **`runs/tarteel_base_lora_private_v2/`** reached early
stopping at **step 156 / epoch 1.76** and exported the **step-78 / epoch .88**
checkpoint selected by generated development macro-reader WER. Adapter byte hashes
match that checkpoint. The original process then exited **code 1** after saving
complete validation/test predictions, before saving Waleed results and the final
report. No traceback was recorded; the cause is **unresolved**, not automatically
ascribed to memory pressure or training failure.

Preserved complete predictions yield **3.69% development WER / .89% CER** and
**3.80% test WER / 1.20% CER** with zero loop/EOS flags. Against the same-decoder
unadapted development control, **only one fewer word error** was observed (59 vs
60 of 1,599 words); the paired surah-bootstrap WER-difference interval includes
zero. This is stable recognition, not a proven meaningful adaptation gain.

An **evaluation-only recovery** in a fresh process checks the exact exported
weights/configuration and frozen PCM hashes, verifies eight saved development
fixtures on reload, and evaluates Waleed with unchanged beam3 controls. It never
fits, picks a new checkpoint or overwrites the original evidence. Results go to
**`runs/tarteel_base_lora_private_v2/recovery_evaluation_v1/`**. Recovery completed:
**568 Waleed clips score 5.49% WER / 2.80% CER**, with no loop/EOS flags. All eight
reload fixtures exactly match the original raw/normalized text, flags and token
counts. See [the final v2 results and limitations](TARTEEL_LORA_V2_RESULTS.md).
No upload, deployment or human-ready claim is made.

Future training code now persists selection/logs and split metrics **before**
expensive inference, streams prediction batches to exclusive-create files and
releases no-longer-needed PCM/optimizer caches. These are resource/evidence
safeguards, not a proven diagnosis of the original exit.
