# Qālūn Whisper tiny/base: LoRA and zero-shot comparison

The scripts use `src/dataset_collection/dataset_qaloon_{hutafi,Husary,dokali}/metadata.jsonl`
and each directory's `audio/` WAV files. They train on `text_asr_normalized`, not
the vowelled Uthmani transcription. Source collection and transcription are
separate; these scripts do not download audio from nQuran.

## Dataset review

At the time of writing there are **569 ayahs per reciter**, 1,707 WAV/label
pairs in total: Fātiḥah 1:1–7 and surahs 78–114. All three folders have the
same `(surah, ayah)` keys. Approximate cumulative audio durations, calculated
from the metadata timings: Huthaify 0.86 h, Husary 1.26 h, Dokali 0.74 h.
The WAVs are expected to be **mono 16 kHz**. No ayah-0 basmalah is included
by default.

For example, Huthaify 78:1 has `078001.wav`, label `عم يتساءلون`, and a
timing of 6,080–11,700 ms (5.62 s). The following ayahs have the same
transcription but different reciter audio in the other folders. The training
label for 78:2 is `عن النبا العظيم` (initial-alif hamza folded); 78:3 turns
the Qālūn yeh barree into `الذي`. Evaluation uses the *same* Qālūn
normalizer for Whisper predictions and stored labels, preserving medial hamzas
(`الملائكة`), terminal `ى` (`موسى`), and Qālūn 1:3 `ملك`.

Whisper's audio window is at most 30 seconds. Four Husary clips exceed it:
78:40 (37.58 s), 98:5 (33.88 s), 98:6 (30.84 s), and 98:8 (35.62 s).
The loader reports and excludes these four rather than silently cropping
their audio against full-ayah labels: **1,703 eligible clips** for all three
reciters. Inspect timing and recitation alignment by listening before training;
metadata/API validation alone cannot prove word-level alignment.

## Setup (local)

From the repository root, use Python 3.10–3.12 with a supported PyTorch install
(CUDA PyTorch is recommended; choose the matching CUDA wheel from PyTorch's
installation instructions if needed):

```bash
python -m pip install -r src/training/requirements_qaloon.txt
python src/training/compare_qaloon_zero_shot.py --output-dir runs/zero_shot
python src/training/train_qaloon_lora.py --model tiny --output-dir runs/tiny_lora
python src/training/train_qaloon_lora.py --model base --output-dir runs/base_lora
```

Start with tiny; base needs more GPU memory. On CPU, reduce `--batch-size 1`
and `--eval-batch-size 1` (training will be slow). Training defaults: 5 epochs,
LoRA rank 8 on attention `q_proj` and `v_proj`, learning rate 1e-4,
batch size 4 and gradient accumulation 4. Use `--epochs`, `--lora-r`,
`--lora-alpha`, `--learning-rate`, and `--gradient-accumulation` to tune.
Output directories contain adapter weights, processor, validation/test
predictions, and `metrics.json`; baseline results are in `zero_shot_metrics.json`.
The training script evaluates loss each epoch, then computes WER/CER on the
validation and untouched test sets after training.

### Reciter selection

Both scripts accept `--include-reciter` and `--exclude-reciter` with keys
`huthaify`, `husary`, `dokali`. The default includes all three. For example:

```bash
python src/training/train_qaloon_lora.py --model tiny --include-reciter husary dokali --exclude-reciter dokali --output-dir runs/husary_tiny
python src/training/compare_qaloon_zero_shot.py --include-reciter husary --output-dir runs/husary_baseline
```

Each reciter uses its own metadata and audio directory; `relative_audio_path`
is resolved against that directory, never against the process working
directory. `--data-root PATH` overrides the parent directory for Kaggle or
another machine. If you intentionally generated separate ayah-0 basmalah
samples, `--include-bismillah` opts into them when present.

## Kaggle notebook cells

Upload the three `dataset_qaloon_*` directories as a Kaggle dataset, preserving
the metadata/audio structure. Enable **GPU** and **Internet** (for the first
model download and package installation); add this repository as a Kaggle
dataset too, or clone it. Replace the two paths below with your actual `/kaggle/input/`
mounts. `/kaggle/input` is read-only; save outputs under `/kaggle/working`.

```python
!pip -q install -r /kaggle/input/ai-reciter/src/training/requirements_qaloon.txt
DATA = "/kaggle/input/qaloon-audio"  # contains the three dataset_qaloon_* directories
CODE = "/kaggle/input/ai-reciter/src/training"
!python {CODE}/compare_qaloon_zero_shot.py --data-root {DATA} --output-dir /kaggle/working/zero_shot
!python {CODE}/train_qaloon_lora.py --model tiny --data-root {DATA} --output-dir /kaggle/working/tiny_lora --batch-size 2 --gradient-accumulation 8
!python {CODE}/train_qaloon_lora.py --model base --data-root {DATA} --output-dir /kaggle/working/base_lora --batch-size 2 --gradient-accumulation 8
```

If GPU memory is limited, use `--batch-size 1 --eval-batch-size 1` and
increase gradient accumulation. Save/download `/kaggle/working/*` after the
session. The same training and baseline code is used locally and on Kaggle;
no separate notebook implementation can silently diverge.

## Evaluation and interpretation

`compare_qaloon_zero_shot.py` evaluates untouched `openai/whisper-tiny` and
`openai/whisper-base` on the **test** set (default), with overall and
per-reciter word error rate (WER) and character error rate (CER). Its
`--split validation` option is for development; keep test results for final
reporting. Per-ayah predictions are saved as JSONL for inspection. The LoRA
script uses the same evaluation function and split seed (default `--seed 42`),
so trained and zero-shot scores are directly comparable when reciter selection
matches. Scores are fractions (0.12 = 12% error). No benchmark score is
claimed here: training and zero-shot inference must actually be run to obtain
one.

The deterministic split is keyed on `(surah, ayah)` rather than filenames:
all recordings of an ayah go to the *same* train/validation/test partition,
preventing the exact reference text from appearing in another reciter's
training partition. Split fractions are approximately 80/10/10. These tests
measure unseen ayahs from the current surahs, **not** generalization to unseen
surahs or reciters. For a future unseen-reciter study, train using two reciters
and evaluate the third *separately*; merely excluding a reciter from training
also excludes it from the current test loader. Recording artifacts, omitted
basmalahs and individual boundaries should be audited before interpreting
ASR error rates as recitation accuracy.

## Growing the corpus

Add Qālūn full-surah MP3s and matching MP3Quran read-ID timings for other
surahs, then run the dataset builder and validate the merged ayah numbering,
WAV format, audio/text agreement and 30-second limit. Keep new reciters in
independent directories and add their folder key to `RECITER_DIRS` in
`qaloon_data.py`. For longer ayahs, obtain verified shorter text-aligned
segments; do not crop a WAV while retaining the original full-ayah label.
After expanding, freeze a new split seed and rerun the zero-shot baselines
before comparing new adapters.
