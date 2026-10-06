# Rattil — Training, Audio Processing, and Metrics

Rattil uses **fine-tuned Whisper speech recognition**, a **separate tajweed-token model**, and a **local FastAPI/WebSocket audio service** to turn Quran recitation into word-level practice feedback. The default recognition release is **`Mathani-Ayat/rattil-qaloon-v4`**.

The numbers below are recorded results from the repository's documentation, not newly run benchmarks.

## 1. Core technology

| Area | Technology and role |
| --- | --- |
| Speech recognition | Whisper-base, adapted from `deepdml/whisper-base-ar-quran-mix-norm` to Qālūn recitation. |
| Training | Python, PyTorch, and Hugging Face Transformers; full-model fine-tuning for the release lineage, with separate LoRA experiments in the repository. |
| Audio augmentation | NumPy, SciPy, and librosa for tempo, pitch, noise, reverberation, and recording-channel simulation. |
| Audio features | Whisper feature extraction converts 16 kHz audio to log-mel features before encoder-decoder inference. |
| Inference API | FastAPI and WebSockets for local audio processing and live updates. |
| Text matching | Arabic normalization and word alignment against Quran text, after speech recognition. |
| Tajweed | A text-based Qālūn rule engine plus a trained model that emits rule tokens alongside words. |
| Interface | Next.js, React, and TypeScript for recording, reference playback, and feedback. |

## 2. Dataset preparation and training

### Audio and label preparation

1. Collect Qālūn recordings from multiple reference reciters.
2. Segment full-surah recordings into ayah clips using forced alignment against the canonical Qālūn text (`segment_by_pauses.py`).
3. Audit clips against their labels. The v2 expansion retained clips only when three independent models agreed on the ayah match; rejected clips were kept for review.
4. Apply the project's ayah-numbering merges and Arabic text normalization.
5. Prepare mono 16 kHz audio for Whisper feature extraction.
6. Keep Waleed out of training for the documented unseen-reciter evaluation.

The early release work fixed a normalization bug affecting **840 words** and excluded **28 audio/label mismatch clips**. These changes matter because incorrect labels teach the model incorrect transcriptions.

### Release evolution

| Version | Main training change |
| --- | --- |
| v1 | Full fine-tuning of a Quran-pretrained Whisper-base model on Huthaify, Husary, and Dokali. Documented recipe: learning rate `1e-5`, cosine schedule, 3 epochs, effective batch size 16, bf16, and time masking. |
| v2 | Six additional Qālūn reciters; dataset expanded from 1,675 to 4,661 clips. Added pitch-preserving tempo augmentation at 0.85–1.45×. |
| v3 | Added 544 audited Garu clips, bringing the documented dataset to 5,205 clips. |
| v4 | Continued from v3 with single-ayah audio plus 2,648 joined clips containing 2–4 consecutive ayahs: **7,853 clips**, **1 epoch**, learning rate **`5e-6`**. Half of the joined clips had pauses shortened to **0.05–0.35 seconds**. |

Joining ayahs makes the training input closer to live use, where a learner may continue across several ayahs without waiting for recognition to finish.

### Robustness tools

The repository also implements a configurable **speaker-robust** augmentation profile. Its availability does not mean every historical release used all its operations.

| Augmentation | Default range or setting | Purpose |
| --- | --- | --- |
| Tempo, preserving pitch | 0.85–1.45× | Faster and slower recitation. |
| Pitch shift | ±3 semitones | Voice variation. |
| Vocal-tract length perturbation (VTLP) | Warp 0.88–1.12 | Frequency variation in log-mel features. |
| Reverberation | RT60 0.15–0.8 seconds | Room acoustics. |
| Noise | SNR 12–35 dB | Background noise. |
| EQ, filtering, gain, clipping, codec simulation | Configurable | Phone microphones and compressed audio. |
| SpecAugment | Time and frequency masking | Training regularization. |

The profile leaves 10% of clips free of waveform augmentation and avoids tempo changes that would push a clip beyond 30 seconds. Training utilities support CUDA, Apple MPS, and CPU, with precision chosen for the device.

## 3. Live audio processing

```text
Microphone / uploaded audio
          ↓
Mono 16 kHz PCM
          ↓
Bounded chronological audio queue
          ↓
Whisper log-mel feature extraction
          ↓
Arabic transcription
          ↓
Overlap stitching + Quran word matching
          ↓
Word progress, skipped-word feedback, and optional tajweed tags
```

- The WebSocket service accepts small **Float32 mono PCM frames**. PCM16 request paths convert samples to Float32 by dividing by 32,768.
- `AudioQueue` has a default **45-second capacity**; it reports overflow rather than silently discarding unprocessed speech.
- The current streaming code uses **8-second ordinary windows** and can expand to **24 seconds for recovery**. Individual Whisper inference inputs must be at most **30 seconds**.
- Transcript stitching uses recognized-word overlap. Ambiguous joins trigger recovery with retained audio rather than filling gaps from expected Quran text.
- Live partial decoding uses one beam; final decoding can use the configured beam count. Supported overrides are **1, 3, or 5 beams**.
- In tajweed mode, `PairedEngine` takes words from the plain recognition model and transfers aligned tags from the tajweed model. This avoids depending on the tag model's weaker word output.

Key implementation files: `src/streaming/server.py`, `live_buffer.py`, `matcher.py`, and `tajweed_tags.py`.

## 4. Recognition metrics

**WER (word error rate)** = `(substitutions + deletions + insertions) / reference words`. Lower is better. **CER** measures the equivalent errors at character level. Neither directly measures pronunciation quality or tajweed correctness.

The release comparison uses **Waleed, a voice excluded from training**. Faster-speed rows are accelerated-audio stress tests.

| Evaluation condition | v1 WER | v2 WER | v3 WER | **v4 WER** |
| --- | --- | --- | --- | --- |
| Normal speed | 2.6% | 2.0% | 2.2% | **2.0%** |
| 1.25× speed | 19.9% | 16.9% | 13.4% | **11.9%** |
| 1.5× speed | 30.4% | 21.5% | 21.2% | **20.4%** |

The same report records **100% on its selected Qālūn-word checks** and **0% invented skipped ayahs in its omission test**. These are scoped benchmark results, not guarantees for arbitrary recordings.

The v4 development report also records **21 wrongly marked missing words in each of two fast An-Naba app runs** with the matcher fix and three beams. This illustrates why offline WER and live user-visible feedback should be evaluated separately.

## 5. Tajweed training and metrics

The tajweed model adds tokens such as `<tj:madd>` and `<tj:ghunna>` after words. The v2 run started from rattil-v4, used **10 readers / 5,036 training clips**, plain synthetic-speech negatives, speaker-robust augmentation, and learning rate **`5e-5`**. Training ended after epoch 3; the installed model uses that epoch.

Recorded end-to-end results use **v4 words plus tajweed tokens**, with greedy decoding:

| Metric | Installed tajweed v2 result |
| --- | --- |
| Rule-token precision on Waleed | **0.99** |
| Rule-token recall on Waleed | **0.94** |
| False alarms per word, plain TTS | **0.006** |
| False alarms per word, unseen plain Majed voice | **0.019** |
| Word error in paired ahkam mode on Waleed | **2.1%** |
| Evaluation sizes | Waleed: **562 clips**; TTS: **182 takes**; Majed: **104 ayahs**. |

Precision measures how many emitted rule tokens belong there; recall measures how many expected rule tokens were detected. False alarms per word counts invented rule tokens on plain readings.

The standalone epoch-3 tajweed model had **59% validation WER** because some clips looped. The application therefore uses the plain model for words. Training includes correct professional recitation and synthetic plain negatives; these results do not establish accuracy on teacher-labelled learner mistakes.

## 6. Documented speed and model size

| Item | Recorded value | Scope |
| --- | --- | --- |
| Recognition model download | About **290 MB** | rattil-qaloon-v4 weights. |
| Tajweed model download | About **290 MB** | rattil-qaloon-tajweed-v2 weights. |
| Three-beam live decode | Reported worst decode **0.7 s**, cycle **1.2 s** | Historical v3-era report; hardware is not specified in that section. |
| Paired word/tag decoding | About **0.2 s per 9 s of audio** | Historical v1 tajweed report on a Mac CPU. |

These timings describe different historical configurations. They are not a measured latency benchmark for the current v4 streaming pipeline on this machine.

The evaluation tooling can record **real-time factor (RTF)** and **p50/p95 batch inference latency**. Its timing scope excludes file loading, feature extraction, and network latency. No fresh current-machine RTF, VRAM usage, or training duration was measured for this overview.

## Sources

- [Release training and WER comparison](docs/QALOON_MODEL_V2.md)
- [v1 recipe and benchmark methodology](docs/QALOON_MODEL_V1.md)
- [Speaker-robust augmentation](docs/SPEAKER_ROBUST_TRAINING.md)
- [Tajweed training and evaluation](docs/QALOON_TAJWEED.md)
- [Model assets and local setup](docs/LOCAL_SETUP.md)
- [Streaming implementation](src/streaming/server.py)
- [Audio queue and overlap stitching](src/streaming/live_buffer.py)
- [Evaluation performance instrumentation](src/training_with_gpu/gpu_evaluation.py)
