# Qālūn ASR — GPU experiment and measured results

**Goal:** improve the transcription that supports V1 (recognizing forgotten
words/verses). Production-readiness is *measured*, not inferred from a falling
training loss. The pre-existing trained tiny adapter and test predictions are
under `runs/`; the new GPU scripts do not overwrite them unless you explicitly
point `--output-dir` there. `python src/training_with_gpu/summarize_previous.py`
recalculates per-reciter scores directly from saved predictions.

## What the saved run actually shows

Data: Fātiḥah + surahs 78–114. The previous experiment used **three** reciters
and seed 42 (1,382 train / 141 validation / 180 test; four Husary clips over
30 s excluded). This table is **from existing saved outputs**:

| Model / partition | WER | CER |
| --- | ---: | ---: |
| Whisper tiny, zero-shot, test | **83.14%** | **43.28%** |
| Whisper tiny, LoRA, validation | **73.08%** | **25.32%** |
| Whisper tiny, LoRA, test | **64.37%** | **22.73%** |
| Whisper base, zero-shot, test | **57.98%** | **20.50%** |

Tiny LoRA gained **18.77 percentage points WER** and **20.55 points CER** over
zero-shot tiny on the same test set. However, *zero-shot base already beats
fine-tuned tiny* in this run. Nothing here establishes a production-grade
omission detector: WER counts substitutions and insertions as well as deletions;
CER counts character edits (including spaces), not the probability that a
specific recited phoneme was correct. A falling eval loss over five epochs
(1.0098 → 0.8228) does not prove zero overfitting, architectural saturation,
or that more epochs cannot help. The model was trained with only attention
q/v LoRA modules; the increased rank/target coverage below is a **hypothesis
to test**, not a promised accuracy gain.

The validation and test sets in that run **both contain every reciter**:
47 validation and 60 test clips *per reciter*. This is an ayah-level split,
not a speaker-level split. It is incorrect to attribute the 73% versus 64%
WER difference to Husary being only in validation. Saved per-reciter **tiny
LoRA** results (recomputed from prediction JSONL):

| Reciter | Validation WER / CER | Test WER / CER |
| --- | ---: | ---: |
| Dokali | 75.40% / 29.11% | 67.05% / 25.27% |
| Husary | 80.75% / 27.64% | 73.56% / 26.03% |
| Huthaify | 63.10% / 19.20% | 52.49% / 16.89% |

Examples like `وجعلنا نومكم سباتا` → `وجعلنا ومكم سباتا وجعلنا ومكم سباتا`
are repetitions/hallucinations; `والسابحات سبحا` → `وساب حيات سبحا` has
spacing and substitutions. Inspect the WAV itself before attributing an edit
to model size: Dokali sometimes bleeds into adjacent clips, and ayah boundary
mistakes make a correct transcript *impossible*. `الصور`/`السور` and
`فتاتون`/`فتقتون` are not merely hamza spelling differences. WER can be high
even with lower CER because token boundaries matter; primary scoring retains
medial **ئ/ؤ**, final **ى**, and standard Qālūn spelling. A *secondary* diagnostic
folds hamza seats symmetrically on reference/prediction; **do not** train on
that folded diagnostic to hide real differences.

## Data and split discipline

Four reciters are supported: `huthaify`, `husary`, `dokali`, `waleed`. Waleed
WAVs have no start/end milliseconds; duration comes from the WAV header, not
fabricated timestamps. Its metadata's `reciter` value is sometimes `cached`;
the canonical identity is its **folder key**. Waleed Fātiḥah has a separately
numbered basmalah and combines two textual segments at the end. **All of
Fātiḥah is grouped into one partition across reciters**, avoiding accidental
text leakage across train/test. This changes the old split; *new four-reciter
scores must not be directly compared with the three-reciter table above*.
Default four-reciter dry run: **1,851 train / 188 validation / 232 test**;
four Husary ayahs >30 s excluded rather than silently truncated with their
full-ayah label. Durations are checked against WAV contents for Waleed and
the stored millisecond timings for other reciters. Inspect suspect clips by
ear, compare neighboring ayahs, and repair true boundaries before adding
more examples. Down-weighting Dokali is a temporary bias-control, **not** a
substitute for relabeling/recutting his leaked clips.

## RTX 5070 (12 GB) setup and run

Use Python 3.11/3.12. NVIDIA RTX 50-series requires a recent NVIDIA driver
and a PyTorch CUDA build with Blackwell (`sm_120`) kernels (for example CUDA
12.8 builds); a CPU-only torch wheel will not use the GPU. From repo root:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r src/training_with_gpu/requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA')"
python src/training_with_gpu/train.py --dry-run --output-dir runs/gpu_base
python src/training_with_gpu/train.py --model base --output-dir runs/gpu_base --batch-size 1 --gradient-accumulation 16
python src/training_with_gpu/train.py --model tiny --output-dir runs/gpu_tiny --batch-size 1 --gradient-accumulation 16
```

`--include-reciter` / `--exclude-reciter` accept the four keys; `--data-root`
works with a Kaggle mount containing the four `dataset_qaloon_*` folders.
For Kaggle save outputs in `/kaggle/working`, e.g.:

```bash
python /kaggle/input/ai-reciter/src/training_with_gpu/train.py --model base --data-root /kaggle/input/qaloon-data --output-dir /kaggle/working/qaloon_base
```

Defaults: base, rank 16 on attention **and** feed-forward projections,
batch 1/accumulation 16, BF16 if supported, gradient checkpointing,
validation-loss best checkpoint/early stopping (patience 2), maximum eight
epochs and *train-only* Dokali sampling weight 0.35. Evaluation always uses
the actual, unweighted validation/test recordings. The explicit `--noise-prob`
defaults to **0**; if testing robust augmentation, try `0.1` (mild 25–35 dB
SNR noise on train only), and **keep the no-augmentation baseline**. Do not
apply noise, pitch/time stretch, cloned voices or extra silence to validation
or test. If 12 GB OOMs with batch 1, try tiny; do not truncate a 30+ second
ayah without a corresponding correctly segmented transcript. The script
refuses to silently run CPU training.

Evaluate a fresh base baseline on the **new** four-reciter split, then the
new adapter on that identical split:

```bash
python src/training_with_gpu/evaluate.py --model base --output-dir runs/new_base_zero_shot
python src/training_with_gpu/evaluate.py --model base --adapter runs/gpu_base --output-dir runs/new_base_lora
```

The metrics JSON contains overall and per-reciter WER/CER, *apparent*
deletions and the fraction of **correctly recited** ayahs that would trigger
an omission alert if you naively trusted Whisper deletions. This is a
**false-alarm proxy**, not omission recall: the dataset has no student
recitations with ground-truth missing words. To claim usable V1 quality,
record consented readers intentionally omitting words/entire ayahs; have
reviewers label missing spans; tune alert/abstention thresholds on validation
only; report precision, recall and false alarms per hour/reader on a separate
held-out test set. Acoustic verification against the expected ayah can be
investigated, but forcing Whisper to output the expected text and then
scoring that transcription would leak the answer.

## What is likeliest to improve the result

1. **Repair labels and boundaries first.** Listen to adjacent Dokali clips;
   fix leaked phonemes or drop ambiguous clips. Scrutinize basmalah, merges,
   prolonged recitations and misaligned source ayahs. Never train on a
   transcript that does not match the WAV.
2. **Expand clean Qālūn Quran coverage**: yes, more distinct surahs/words
   should help if source, segmentation and normalization are verified. More
   misaligned Quran merely adds noise. 30+ second ayahs need human-verified
   text-aligned subsegments or a longer-context model; do not clip and retain
   the original text.
3. **Add reciters strategically** and evaluate on a held-out *new voice* as
   well as held-out ayahs. The current test uses the same voices as training;
   it measures unseen ayahs, not generalization to an unfamiliar reader.
4. **Compare base against tiny**, rank/target choices and mild augmentation
   one change at a time, holding the splits fixed. Do not infer convergence
   from five epochs of eval loss; select models using validation ASR quality
   and false alarms, not test WER.
5. **Noise corpora such as MUSAN** can test robustness to background sound
   after clean alignment is established: small, realistic mixtures in training
   only, preserving intelligibility and labels. A quiet recitation benchmark
   does not justify noise that masks consonants.

## XTTS-v2 is not a shortcut to accurate Qālūn tajweed

XTTS takes text plus a short reference speaker sample. A speaker encoder
extracts voice characteristics, an autoregressive component predicts acoustic
tokens, and a vocoder renders speech; temperature, top-k/top-p and repetition
settings change synthetic diversity. It **clones style**, not verified
Qālūn-specific articulation or a scholar's tajweed assessment. Do not clone
named reciters without permission or assign synthesized Qur'an as authentic
ground truth. A cited **5.79% CER** from another setup (a Whisper-Small
diacritic evaluator on synthetic/plain speech) is *not* evidence of tajweed
correctness: the evaluator has its own errors and can share generative biases.
If exploring XTTS later, use clearly labelled, consented synthetic material
only as a supplementary controlled study, reviewed by qualified teachers,
never in held-out evaluation. Compare against clean real-audio baselines.

See `src/training_with_tajweed/README.md` for the separate V2 track.
