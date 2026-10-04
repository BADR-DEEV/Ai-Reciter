# Tarteel-base LoRA v1: completed, not a release candidate

The private four-reader experiment **completed successfully**, stopping after
three epochs (**234 optimizer steps**) because generated macro-reader validation
WER failed to improve for two epochs. It restored **epoch 1 / checkpoint-78**.
No model was deployed, uploaded or substituted for the existing full model.

Evidence is preserved in `runs/tarteel_base_lora_private_v1/`: `metrics.json`,
`comparison.json`, frozen `experiment_manifest.json`, per-sample predictions,
`generalization_history.json`, checkpoints and `adapter/`. The adapter identifies
the pinned Tarteel revision. Dataset membership remained Huthaify/Dokali/Husary
plus automatically passing, user-accepted direct-source Trabulsi; **Taha was not
used**. These are private experimental weights, not certified Qālūn assessment.

## Selected checkpoint scores

All error rates below are percentages; **lower WER/CER is better**.

| Evaluation | Clips | WER | CER | Exact normalized ayah |
| --- | ---: | ---: | ---: | ---: |
| Recording/surah-held-out validation | 365 | **42.36%** | 27.43% | 70.68% |
| Recording/surah-held-out test | 353 | **61.46%** | 27.71% | 72.80% |
| Waleed, held out of adaptation | 568 | **9.43%** | 7.65% | 79.23% |
| Waleed, text also unseen during adaptation | 205 | **10.95%** | 4.32% | 79.02% |

Waleed's surah-clustered 95% WER interval is **6.63–12.74%**. Tarteel's upstream
exposure is unknown. This one professional voice does not establish performance
on arbitrary learners or genuine recitation mistakes. The test interval is very
wide (**5.45–83.43%**, six surah groups), reflecting severe outlier sensitivity.

### Reader-specific test performance

| Reader | Clips | New LoRA WER | Existing full model, same WAVs |
| --- | ---: | ---: | ---: |
| Dokali | 101 | **173.94%** | 9.39% |
| Husary | 101 | 9.39% | 4.93% |
| Huthaify | 101 | 27.70% | 3.05% |
| Trabulsi | 50 | 6.31% | 13.11% |
| **All four** | **353** | **61.46%** | **6.81%** |
| Waleed, separate adaptation-held-out test | 568 | 9.43% | 6.64% |

WER can exceed 100% when the model inserts more words than the reference has.
**The old full model has an unfair familiarity advantage:** reconstructed legacy
training-audio overlap is **261/353 test clips** and **463/568 Waleed clips**, and
Waleed was an adaptation voice for it. Its original checkpoint has no embedded
split manifest, so those counts are a reconstruction, not a provenance-certified
control. The paired comparisons are same-audio observations, **not proof that
one initialization or training method is intrinsically superior**. Trabulsi's
better observed score likewise is not a statistically established general gain.

The original full model's historical **26.39% WER / 7.96% CER** used a different
test partition. Do not substitute the newly contaminated 6.81% score for that
historical benchmark or advertise it as new clean generalization performance.

## Failure analysis

| Epoch | Validation loss | Validation micro WER | Validation macro-reader WER | Clean train-probe WER |
| --- | ---: | ---: | ---: | ---: |
| 1 | 3.476 | 42.36% | **37.28%** | 9.05% |
| 2 | 1.623 | **98.74%** | 95.72% | 71.43% |
| 3 | 1.277 | 76.97% | 75.49% | 65.71% |

Falling teacher-forced loss did **not** mean better free-running recognition.
Clean training-probe recognition also deteriorated, so this is **not enough
evidence to diagnose ordinary held-out-only overfitting**. Training/decoding
instability, repetition and compatibility warrant investigation; the causal
source is not yet isolated. There was no same-protocol unadapted Tarteel control.

- Selected validation predictions include **Dokali 109:6** generating an extended
  repetition of `د`, with **435 inserted words** on a four-word reference. This
  one clip contributes about **71.7% of all validation word errors**.
- The selected test set has **775 insertions / 129 substitutions / 8 deletions**.
  Insertions account for about **85.0% of its word errors**. Most are associated
  with Dokali; high exact-ayah accuracy does not cancel catastrophic failures.
- Trabulsi's test score is the strongest observed reader score, but this does not
  prove imperfect supplemental cuts are harmless or that they caused failures
  elsewhere. There was no supplement ablation. Do not blame or clear a dataset
  from this aggregate run alone.
- No outliers were removed, no expected-text prompts were added, and no repetition
  penalties were retroactively chosen to improve the reported test score.
- v1's adapter initialization was not seeded before model construction; future
  entry-point runs now do so. Preserve v1 as evidence rather than claiming an
  exactly reproducible fresh start from its seed alone.

## Decision and next experiment

**Keep the existing deployed model. Do not promote v1.** The selected adapter
and all results remain available for inspection. Do not train another run or
change decoding automatically based on these held-out scores.

Before a deliberately authorized next run:

1. Audit decoder prefixes, label shifting, EOS, attention masking, tokenizer
   vocabulary and generation metadata against pinned Tarteel; verify teacher-
   forced loss and blind decoding on fixed **training/development** fixtures.
   Saved PEFT adapters do not include a complete generation configuration here:
   reloading must use the pinned base/processor plus `configure_generation`, not
   silently inherit Tarteel's legacy decoding metadata. Nothing is deployed.
2. Establish an **unadapted pinned Tarteel** development control, and a fresh OpenAI
   control under identical recording-safe partitions. Add an explicit supplement
   ablation to separate data effects from initialization/optimization effects.
3. Test conservative learning rates/early checkpoint intervals on development
   data only. Inspect free-running predictions as well as loss, and validate any
   repetition/abstention rule without removing legitimate recitation repeats.
4. Preserve these now-observed test sets as descriptive regression suites; use a
   **new sealed reader/recording/learner test** for confirmatory future claims.
   Obtain consent, rights and qualified labels; include realistic microphone/noise
   conditions and actual mistakes rather than treating professional ASR errors
   as learner omissions or tajweed errors.

The current run establishes that the pipeline trains, selects, saves and produces
auditable comparisons—not that the resulting model improves Rattil.
