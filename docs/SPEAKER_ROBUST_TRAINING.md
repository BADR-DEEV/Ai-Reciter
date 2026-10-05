# Speaker-robust training

All training reciters are adult male professional sheikhs, but Rattil's learners are men,
women and children recording on cheap phone microphones in ordinary rooms. Without help
the model also learns "what a sheikh's voice in a studio sounds like". The
`speaker-robust` augmentation profile changes the voice and the recording channel of
every training clip on the fly, so the model has to rely on what is being recited.

Code: `src/training_with_gpu/augment.py` (numpy/scipy/librosa only, no torchaudio),
used by `train_base_full.py` and, with `--augment-profile`, `train_tarteel_lora.py`.

## What each op simulates

| Op | Simulates | `speaker-robust` default |
| --- | --- | --- |
| tempo (phase vocoder, pitch kept) | faster or slower readers | p .5, x0.85–1.45 |
| pitch shift (librosa, duration kept) | higher or lower voices (F0 and formants) | p .3, ±3 semitones |
| VTLP: log-mel frequency warp, Jaitly & Hinton 2013 | women and children (shorter vocal tract), other men | p .4, warp 0.88–1.12 |
| synthetic room impulse response | bedrooms, halls, mosques | p .25, RT60 0.15–0.8 s, wet 15–60 % |
| random EQ (shelf and peaking biquads) | different microphone and phone colouring | p .3, 1–3 bands, ±8 dB |
| telephone band-pass or low-pass | calls (300–3400 Hz) and cheap mics (3–7 kHz) | p .1 |
| white/pink/brown noise at an SNR measured against the clip's power | fans, traffic, hiss | p .35, 12–35 dB |
| gain | quiet or loud recordings (Whisper features are not level-normalised) | p .4, ±6 dB |
| soft clipping | overdriven phone microphones | p .03 |
| codec (μ-law 8-bit, 8 kHz round trip, or both) | VoIP/GSM and compressed uploads | p .08 |
| background voice (another training reciter, ≥ 20 dB below) | someone else talking nearby | p 0 (off: it can confuse transcription) |
| SpecAugment (`model.config`, applied by HF in training only) | dropouts, general regularisation | time p .05 × 10 frames, freq p .05 × 10 bins |

10 % of clips skip every waveform op (`clean_prob`). The tempo op is skipped when it would
push a clip past 30 s, so speech is never cut. Output is float32 in [-1, 1].

## Profiles

- `speaker-robust` (default in `train_base_full.py`): the table above.
- `legacy`: exactly the earlier behaviour. `--speed-prob` (.40) applies tempo `--tempo-min`/`--tempo-max`
  (.95–1.05), and `--noise-prob` (.12) adds white noise at 25–35 dB on the global `np.random`.
  The checkpoint's SpecAugment settings are left as they are. This is the default in `train_tarteel_lora.py`.
- `none`: no augmentation, and SpecAugment is switched off.

To override any field, pass `--aug FIELD=VALUE`, which you can repeat, for example
`--aug pitch_prob=0.5 --aug noise_colors=pink,brown --aug other_voice_prob=0.05`.
The four legacy flags override `noise_prob`, `tempo_prob`, `tempo_min` and `tempo_max` in any
profile. `--mask-time-prob`, `--mask-time-length`, `--mask-feature-prob` and `--mask-feature-length`
set SpecAugment directly. The run's `experiment_manifest.json` records the full config that was used.

To hear the waveform ops (VTLP and SpecAugment act on features, so you cannot hear them), run:

```bash
python -m src.training_with_gpu.augment --input some_clip.wav --out-dir /tmp/aug --n 8 --profile speaker-robust
python -m src.training_with_gpu.augment --input some_clip.wav --out-dir /tmp/reverb --n 3 --profile none --aug reverb_prob=1
```

## Data and devices

- Readers are read from `src/dataset_collection/` (where `pull_hf_assets.py` links them), or from
  `data/hf/qaloon-reciter-dataset/` if that is missing. `--data-root` overrides both.
- `--reader-root DIR` registers every `DIR/dataset_qaloon_<name>/` folder as reciter `<name>`, in lower case.
  You can repeat it. Pass a wrong `--reciters` name with `--dry-run` to print every known key.
- `--device auto` picks CUDA (bf16, or fp16 when bf16 is unsupported), then MPS (fp32), then CPU.
  On the training PC, `--dataloader-workers 4` spreads the augmentation across processes, and each worker draws its own random stream.
  Keep `0` workers only to reproduce a `legacy` run bit for bit.
- On the Mac, prefix commands with `PYTHONPATH=src/deployment/mac_shim`. You also need `jiwer` for the metrics.

## (a) Retrain the plain model with speaker-robust augmentation (training PC)

```bash
python src/training_with_gpu/train_base_full.py \
  --init-model deepdml/whisper-base-ar-quran-mix-norm \
  --reader-root data/hf/qaloon-reciter-dataset --reader-root data/hf/qaloon-new-reciters \
  --reciters huthaify husary dokali abusnaina akri daawob deeban kshidan qeniwa \
  --augment-profile speaker-robust --learning-rate 1e-5 --epochs 4 --patience 2 \
  --dataloader-workers 4 --output-dir runs/rattil_qaloon_v4_robust
```

Check the reciter keys with `--dry-run` first: they come from the folder names (add `garu` if its folder is present). Waleed can never
enter training, because the trainer refuses it. A cheaper option is to adapt the current release
instead of starting over: use `--init-model runs/rattil_qaloon_v3 --learning-rate 5e-6 --epochs 2`.

## (b) Leave-one-reciter-out evaluation

`--holdout-reciter KEY` removes that reciter from training, validation and checkpoint selection.
The reciter is scored only at the end, and you can repeat the flag. Start from a checkpoint that
never saw the held-out reader. v3 saw all ten, so it is not a fair start.

```bash
for r in huthaify husary dokali abusnaina akri daawob deeban kshidan qeniwa; do
  for profile in legacy speaker-robust; do
    python src/training_with_gpu/train_base_full.py \
      --init-model deepdml/whisper-base-ar-quran-mix-norm \
      --reader-root data/hf/qaloon-reciter-dataset --reader-root data/hf/qaloon-new-reciters \
      --reciters huthaify husary dokali abusnaina akri daawob deeban kshidan qeniwa --holdout-reciter "$r" \
      --augment-profile "$profile" --learning-rate 1e-5 --epochs 4 \
      --output-dir "runs/loro/$profile/$r"
  done
done
```

Compare `metrics.json` → `per_reciter_wer.holdout` across the two profiles. Per-clip outputs are in
`holdout_predictions.jsonl`. To test against Waleed, a voice that never enters training, add
`--reader-root data/hf/qaloon-reciter-experiments --holdout-reciter waleed`. The run prints
`<split> per-reciter WER:` after evaluating each split.

## (c) The tajweed model (tag tokens in the targets)

Inputs:

- `data/tajweed/labels.jsonl`: one JSON object per line. Each holds the target text under the `--label-field`
  name, or under `"text"`, plus a join key:

  ```json
  {"reciter_key": "huthaify", "relative_audio_path": "audio/078014.wav", "text": "…"}
  {"relative_audio_path": "audio/078014.wav", "text": "…"}
  {"reciter_key": "husary", "surah": 78, "ayah": 14, "text": "…"}
  {"surah": 78, "ayah": 14, "text": "وانزلنا<tj:madd> من<tj:ghunna> …"}
  ```

  The most specific key wins, in this order: reciter + path, reciter + surah/ayah, path for any reciter,
  then surah/ayah for any reciter. `relative_audio_path` is the metadata field (`audio/SSSAAA.wav`), and it
  is the same in every reader folder. Use `reciter_key` only for targets that are specific to one recording.
  If tags come from the text alone, one `{"surah", "ayah", "text"}` line per ayah covers every reader.
  Identical duplicates are fine and conflicting ones are an error. Every training and validation clip
  needs a target. Write tags straight after their word with no space: `word<tj:a><tj:b>`.
- `src/tajweed/model_tokens.txt`: one token per line. Lines starting with `#` are ignored.
  The tokens are added as **non-special** tokens, so `skip_special_tokens=True` keeps them.
  Their embedding rows start at the mean of the existing rows. Tokens other than `<tj:NAME>` are
  allowed, but WER does not strip them.

```bash
python src/training_with_gpu/train_base_full.py \
  --init-model runs/rattil_qaloon_v3 \
  --label-field text_tajweed --labels-jsonl data/tajweed/labels.jsonl --extra-tokens src/tajweed/model_tokens.txt \
  --reader-root data/hf/qaloon-reciter-dataset --reader-root data/hf/qaloon-new-reciters \
  --reciters huthaify husary dokali abusnaina akri daawob deeban kshidan qeniwa \
  --augment-profile speaker-robust --learning-rate 5e-5 --epochs 10 --patience 3 \
  --output-dir runs/rattil_qaloon_tajweed_v1
```

Generate the two inputs with `python -m src.tajweed.targets` (see [`QALOON_TAJWEED.md`](QALOON_TAJWEED.md)).
Tagged runs select checkpoints by `tajweed_score` (tag F1 minus macro WER) unless `--select-by` says
otherwise. The new tag embeddings start from the mean row, so they need a higher learning rate than
plain fine-tuning; 2e-5 left tag F1 at 0.45 after five epochs on three readers. The serving preset
`rattil-tajweed-v1` looks for `runs/rattil_qaloon_tajweed_v1`.

- WER always strips `<tj:…>` tags first and is scored against `text_asr_normalized`.
  `--label-field` therefore must not be `text_asr_normalized`.
  Tags get their own per-word precision, recall and F1 (`eval_tag_*` during training, and
  `metrics.json` → `<split>.tags.overall` / `.per_reciter`). These come from aligning the tag-stripped
  words, and each tag belongs to the word it follows.
- The output folder holds the tokenizer with the tags, so serving loads them with the model.
  `generation_config.force_unique_generate_call` is set to `true`. Without it, Whisper's tensor-returning
  `generate()` treats two consecutive tags (their IDs sit above the timestamp range) as a timestamp segment
  and corrupts the output. Any consumer that compares text should run `strip_tags()` from
  `train_base_full.py` first.

## Smoke run (Mac, about 30 s)

```bash
PYTHONPATH=src/deployment/mac_shim python src/training_with_gpu/train_base_full.py \
  --data-root data/hf/qaloon-reciter-dataset --init-model runs/rattil_qaloon_v3 \
  --output-dir /tmp/smoke_run --max-steps 5 --max-train-clips 16 --max-samples-per-split 4 \
  --holdout-reciter dokali --batch-size 4 --gradient-accumulation 1 --logging-steps 1 --device mps
```

Tests: `TMPDIR=/private/tmp PYTHONPATH=src/deployment/mac_shim python -m unittest src.training_with_gpu.test_augment`.
