"""Speaker- and channel-robust augmentation for Whisper fine-tuning (numpy/scipy/librosa only).

The training reciters are adult male professionals; learners are men, women and
children on cheap phone microphones in untreated rooms. Waveform ops run on the
16 kHz clip before feature extraction, VTLP warps the log-mel features after it,
and SpecAugment is configured on the model (HF Whisper masks only in training).
Audio never exceeds 30 s (an op that would lengthen past it is skipped) and is
returned as float32 in [-1, 1].

Listen: python -m src.training_with_gpu.augment --input x.wav --out-dir dir --n 8
"""
import argparse
from dataclasses import asdict, dataclass, fields, replace
from functools import lru_cache
import math
from pathlib import Path

import numpy as np
from scipy import signal

SAMPLE_RATE = 16000
MAX_SECONDS = 30
NOISE_COLORS = ("white", "pink", "brown")
SPEC_FIELDS = ("mask_time_prob", "mask_time_length", "mask_feature_prob", "mask_feature_length")


@dataclass(frozen=True)
class AugmentConfig:
    legacy: bool = False  # the pre-2026 recipe: tempo then white noise, on global np.random
    legacy_speed: bool = False  # legacy only: resample speed .92-1.08 instead of tempo (needs torchaudio)
    clean_prob: float = 0.0  # chance a clip skips every waveform op
    tempo_prob: float = 0.0
    tempo_min: float = .85
    tempo_max: float = 1.45
    pitch_prob: float = 0.0
    pitch_semitones: float = 3.0
    vtlp_prob: float = 0.0
    vtlp_min: float = .88
    vtlp_max: float = 1.12
    other_voice_prob: float = 0.0
    other_voice_snr_min: float = 20.0
    other_voice_snr_max: float = 30.0
    reverb_prob: float = 0.0
    rt60_min: float = .15
    rt60_max: float = .8
    reverb_wet_min: float = .15
    reverb_wet_max: float = .6
    eq_prob: float = 0.0
    eq_gain_db: float = 8.0
    band_prob: float = 0.0
    telephone_share: float = .4
    noise_prob: float = 0.0
    snr_min: float = 12.0
    snr_max: float = 35.0
    noise_colors: tuple = NOISE_COLORS
    gain_prob: float = 0.0
    gain_db: float = 6.0
    clip_prob: float = 0.0
    codec_prob: float = 0.0
    # None keeps the checkpoint's own SpecAugment settings.
    mask_time_prob: float = None
    mask_time_length: int = None
    mask_feature_prob: float = None
    mask_feature_length: int = None

    def __post_init__(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name.endswith(("_prob", "_share")) and value is not None and not 0 <= value <= 1:
                raise ValueError(f"{item.name} must be in [0, 1]")
        ranges = [(self.tempo_min, self.tempo_max), (self.vtlp_min, self.vtlp_max), (self.rt60_min, self.rt60_max),
                  (self.reverb_wet_min, self.reverb_wet_max), (self.snr_min, self.snr_max),
                  (self.other_voice_snr_min, self.other_voice_snr_max)]
        if any(not (math.isfinite(low) and math.isfinite(high) and low <= high) for low, high in ranges):
            raise ValueError("Augmentation ranges must be finite and ordered")
        if min(self.tempo_min, self.vtlp_min, self.rt60_min) <= 0 or self.reverb_wet_min < 0 or self.reverb_wet_max > 1:
            raise ValueError("Tempo/VTLP/RT60 must be positive and reverb wet mix in [0, 1]")
        if min(self.pitch_semitones, self.eq_gain_db, self.gain_db) < 0 or not self.noise_colors \
                or set(self.noise_colors) - set(NOISE_COLORS):
            raise ValueError(f"Non-negative pitch/EQ/gain spans and noise colors from {NOISE_COLORS} required")
        if any(value is not None and value <= 0 for value in (self.mask_time_length, self.mask_feature_length)):
            raise ValueError("SpecAugment mask lengths must be positive")

    def active(self):
        """Whether any waveform/VTLP op can fire (SpecAugment lives on the model config)."""
        return any(getattr(self, item.name) for item in fields(self)
                   if item.name.endswith("_prob") and item.name not in (*SPEC_FIELDS, "clean_prob"))


PROFILES = {
    "none": AugmentConfig(mask_time_prob=0.0, mask_feature_prob=0.0),
    # Exactly the previous train_base_full defaults (speed .40 -> tempo .95-1.05, white noise .12 at 25-35 dB).
    "legacy": AugmentConfig(legacy=True, tempo_prob=.40, tempo_min=.95, tempo_max=1.05,
                            noise_prob=.12, snr_min=25.0, snr_max=35.0, noise_colors=("white",)),
    "speaker-robust": AugmentConfig(
        clean_prob=.1, tempo_prob=.5, pitch_prob=.3, vtlp_prob=.4, reverb_prob=.25, eq_prob=.3, band_prob=.1,
        noise_prob=.35, gain_prob=.4, clip_prob=.03, codec_prob=.08,
        mask_time_prob=.05, mask_time_length=10, mask_feature_prob=.05, mask_feature_length=10),
}


def with_overrides(config, items):
    """Apply 'field=value' strings, e.g. pitch_prob=0.5 or noise_colors=pink,brown."""
    types = {item.name: item.type for item in fields(config)}
    values = {}
    for item in items or ():
        key, separator, raw = item.partition("=")
        key = key.strip().replace("-", "_")
        if not separator or key not in types:
            raise ValueError(f"Unknown augmentation override {item!r}; fields: {', '.join(types)}")
        kind = types[key]
        values[key] = (tuple(v.strip() for v in raw.split(",") if v.strip()) if kind is tuple
                       else raw.strip().lower() in ("1", "true", "yes") if kind is bool else kind(raw))
    return replace(config, **values)


def build_config(profile="speaker-robust", overrides=(), **values):
    """Profile defaults, then explicit field values (None keeps the default), then override strings."""
    return with_overrides(replace(PROFILES[profile], **{k: v for k, v in values.items() if v is not None}), overrides)


def configure_spec_augment(model_config, config):
    """Set HF Whisper SpecAugment fields; returns what was set ({} keeps the checkpoint's settings)."""
    values = {name: getattr(config, name) for name in SPEC_FIELDS if getattr(config, name) is not None}
    if not values:
        return {}
    for name, value in values.items():
        setattr(model_config, name, value)
    model_config.apply_spec_augment = bool(model_config.mask_time_prob or model_config.mask_feature_prob)
    return {name: getattr(model_config, name) for name in ("apply_spec_augment", *SPEC_FIELDS)}


def phase_vocoder(spectrum, rate, phase_advance):
    """torchaudio.functional.phase_vocoder, ported op-for-op (Anaconda's torchaudio is broken on the Mac)."""
    import torch
    if rate == 1.0:
        return spectrum
    time_steps = torch.arange(0, spectrum.size(-1), rate, dtype=torch.real(spectrum).dtype)
    alphas = time_steps % 1.0
    phase_0 = spectrum[..., :1].angle()
    spectrum = torch.nn.functional.pad(spectrum, [0, 2])
    spectrum_0 = spectrum.index_select(-1, time_steps.long())
    spectrum_1 = spectrum.index_select(-1, (time_steps + 1).long())
    phase = spectrum_1.angle() - spectrum_0.angle() - phase_advance
    phase = phase - 2 * math.pi * torch.round(phase / (2 * math.pi))
    phase = torch.cat([phase_0, (phase + phase_advance)[..., :-1]], dim=-1)
    magnitude = alphas * spectrum_1.abs() + (1 - alphas) * spectrum_0.abs()
    return torch.polar(magnitude, torch.cumsum(phase, -1))


def tempo_perturb(audio, sample_rate, rate):
    """Pitch-preserving phase-vocoder tempo; retain all content, never crop."""
    import torch
    target_length = round(len(audio) / rate)
    if target_length > sample_rate * MAX_SECONDS:
        return audio.copy()  # Reject augmentation, not the terminal recitation.
    tensor = torch.from_numpy(audio).float()
    fft, hop = 512, 128
    window = torch.hann_window(fft)
    spectrum = torch.stft(tensor, n_fft=fft, hop_length=hop, window=window, return_complex=True)
    phase = torch.linspace(0, math.pi * hop, spectrum.shape[-2]).unsqueeze(-1)
    stretched = phase_vocoder(spectrum, rate, phase)
    return torch.istft(stretched, n_fft=fft, hop_length=hop, window=window,
                       length=target_length).numpy().astype("float32")


def limit_peak(audio, peak=.999):
    """A common gain keeps every relative level (and SNR) while avoiding saturation."""
    current = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if current > peak:
        audio = audio * (peak / current)
    return audio.astype("float32")


def match_rms(audio, reference):
    power = float(np.mean(np.square(audio, dtype=np.float64)))
    target = float(np.mean(np.square(reference, dtype=np.float64)))
    return audio * math.sqrt(target / power) if power > 0 else audio


def mix_at_snr(audio, noise, snr_db):
    signal_power = float(np.mean(audio.astype(np.float64) ** 2))
    noise_power = float(np.mean(noise.astype(np.float64) ** 2))
    if signal_power == 0 or noise_power == 0:
        return audio.copy()
    return limit_peak(audio + noise * math.sqrt(signal_power / (10 ** (snr_db / 10) * noise_power)))


def add_snr_noise(audio, snr_db, rng=np.random):
    return mix_at_snr(audio, rng.normal(size=audio.shape), snr_db)


def colored_noise(length, color, rng):
    white = rng.normal(size=length)
    if color == "white":
        return white
    spectrum = np.fft.rfft(white)
    frequency = np.fft.rfftfreq(length)
    spectrum[1:] /= frequency[1:] ** (.5 if color == "pink" else 1.0)  # 1/f power, or 1/f^2 for brown
    spectrum[0] = 0
    return np.fft.irfft(spectrum, length)


def pitch_shift(audio, semitones, sample_rate=SAMPLE_RATE):
    """Shift F0 and formants together, keeping duration (librosa phase vocoder + resample)."""
    import librosa
    shifted = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=semitones, n_fft=512, hop_length=128)
    return limit_peak(shifted)


def synthetic_rir(rt60, rng, sample_rate=SAMPLE_RATE):
    """Exponentially decaying Gaussian noise reaching -60 dB at rt60; unit energy."""
    t = np.arange(max(1, int(rt60 * sample_rate))) / sample_rate
    rir = rng.normal(size=len(t)) * np.exp(-math.log(1000) * t / rt60)
    return rir / np.sqrt(np.sum(rir ** 2))


def reverb(audio, rt60, wet, rng, sample_rate=SAMPLE_RATE):
    reverberant = signal.fftconvolve(audio, synthetic_rir(rt60, rng, sample_rate))[:len(audio)]
    return limit_peak(match_rms((1 - wet) * audio + wet * reverberant, audio))


def biquad(kind, frequency, gain_db, q=1 / math.sqrt(2), sample_rate=SAMPLE_RATE):
    """RBJ audio-EQ-cookbook peaking/shelving filter as one SOS row."""
    a = 10 ** (gain_db / 40)
    w = 2 * math.pi * frequency / sample_rate
    cos, alpha = math.cos(w), math.sin(w) / (2 * q)
    if kind == "peak":
        b, d = [1 + alpha * a, -2 * cos, 1 - alpha * a], [1 + alpha / a, -2 * cos, 1 - alpha / a]
    else:
        root = 2 * math.sqrt(a) * alpha
        sign = 1 if kind == "lowshelf" else -1
        b = [a * ((a + 1) - sign * (a - 1) * cos + root), sign * 2 * a * ((a - 1) - sign * (a + 1) * cos),
             a * ((a + 1) - sign * (a - 1) * cos - root)]
        d = [(a + 1) + sign * (a - 1) * cos + root, -sign * 2 * ((a - 1) + sign * (a + 1) * cos),
             (a + 1) + sign * (a - 1) * cos - root]
    return np.array(b + d) / d[0]


def random_eq(audio, rng, max_gain_db, sample_rate=SAMPLE_RATE):
    bands = [("lowshelf", rng.uniform(80, 400), 1 / math.sqrt(2)), ("highshelf", rng.uniform(2000, 6000), 1 / math.sqrt(2)),
             ("peak", rng.uniform(250, 4000), rng.uniform(.5, 2.0))]
    chosen = [bands[i] for i in sorted(rng.choice(3, size=int(rng.uniform(1, 4)), replace=False))]
    sos = np.stack([biquad(kind, f, rng.uniform(-max_gain_db, max_gain_db), q, sample_rate) for kind, f, q in chosen])
    return limit_peak(match_rms(signal.sosfilt(sos, audio), audio)), len(chosen)


def band_limit(audio, telephone, cutoff=None, sample_rate=SAMPLE_RATE):
    """Telephone 300-3400 Hz band-pass, or a cheap-mic low-pass; level restored like a phone's AGC."""
    sos = (signal.butter(4, [300, 3400], "bandpass", fs=sample_rate, output="sos") if telephone
           else signal.butter(6, cutoff, "lowpass", fs=sample_rate, output="sos"))
    return limit_peak(match_rms(signal.sosfilt(sos, audio), audio))


def soft_clip(audio, drive):
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak == 0:
        return audio.copy()
    return limit_peak(np.tanh(audio * (drive / peak)) / math.tanh(drive) * peak)


def mu_law(audio, bits=8):
    mu = 2 ** bits - 1
    encoded = np.sign(audio) * np.log1p(mu * np.abs(audio)) / math.log1p(mu)
    quantized = np.round((encoded + 1) / 2 * mu) / mu * 2 - 1
    return limit_peak(np.sign(quantized) * ((1 + mu) ** np.abs(quantized) - 1) / mu)


def narrowband(audio):
    """Down to 8 kHz and back: everything above 4 kHz is lost, as on a VoIP/GSM call."""
    return limit_peak(signal.resample_poly(signal.resample_poly(audio, 1, 2), 2, 1)[:len(audio)])


@lru_cache(maxsize=4)
def mel_centers(n_mels, sample_rate=SAMPLE_RATE):
    """Centre frequencies of Whisper's Slaney mel filters (0 Hz to Nyquist)."""
    import librosa
    return librosa.mel_frequencies(n_mels + 2, fmin=0.0, fmax=sample_rate / 2)[1:-1]


def vtlp_warp(frequency, alpha, nyquist, boundary=4800.0):
    """Jaitly & Hinton (2013) piecewise-linear warp; alpha > 1 raises formants (shorter vocal tract)."""
    knee = boundary * min(alpha, 1.0) / alpha
    upper = nyquist - (nyquist - boundary * min(alpha, 1.0)) / (nyquist - knee) * (nyquist - frequency)
    return np.where(frequency <= knee, frequency * alpha, upper)


def vtlp(features, alpha, sample_rate=SAMPLE_RATE, boundary=4800.0):
    """Warp (n_mels, frames) log-mel features along frequency; the shape is unchanged."""
    if alpha == 1.0:
        return features.copy()
    import librosa
    centers = mel_centers(features.shape[0], sample_rate)
    nyquist = sample_rate / 2
    grid = np.concatenate([[0.0], centers, [nyquist]])
    source = np.interp(centers, vtlp_warp(grid, alpha, nyquist, boundary), grid)  # inverse warp
    position = np.interp(librosa.hz_to_mel(source), librosa.hz_to_mel(centers), np.arange(len(centers)))
    lower = np.floor(position).astype(int)
    upper = np.minimum(lower + 1, len(centers) - 1)
    fraction = (position - lower)[:, None]
    return ((1 - fraction) * features[lower] + fraction * features[upper]).astype(features.dtype)


class Augmenter:
    """Picklable pipeline. In a DataLoader worker the stream is re-derived from the worker's seed,
    which differs per worker and per epoch, so workers never repeat each other's augmentations."""

    def __init__(self, config, seed=0):
        self.config, self.seed = config, seed
        self.applied = []
        self._rng, self._key = None, None

    def rng(self):
        if self.config.legacy:
            return np.random  # bit-for-bit the old recipe, seeded by transformers.set_seed
        import torch.utils.data
        info = torch.utils.data.get_worker_info()
        key = None if info is None else info.seed
        if self._rng is None or key != self._key:
            self._rng = np.random.default_rng([self.seed] if key is None else [self.seed, key])
            self._key = key
        return self._rng

    def waveform(self, audio, rng=None, other_voice=None):
        """other_voice(rng) -> a clip by a DIFFERENT reciter, only used by the other-voice op."""
        c, rng, self.applied = self.config, rng or self.rng(), []
        if c.legacy:
            return self._legacy(audio, rng)

        def hit(probability):
            return probability > 0 and rng.random() < probability

        if hit(c.clean_prob):
            return limit_peak(audio)
        long_enough = len(audio) >= 2048
        if long_enough and hit(c.tempo_prob):
            rate = rng.uniform(c.tempo_min, c.tempo_max)
            if round(len(audio) / rate) <= SAMPLE_RATE * MAX_SECONDS:
                audio = tempo_perturb(audio, SAMPLE_RATE, rate)
                self.applied.append(f"tempo x{rate:.2f}")
        if long_enough and hit(c.pitch_prob):
            semitones = rng.uniform(-c.pitch_semitones, c.pitch_semitones)
            audio = pitch_shift(audio, semitones)
            self.applied.append(f"pitch {semitones:+.1f} st")
        voice = other_voice(rng) if other_voice is not None and hit(c.other_voice_prob) else None
        if voice is not None and len(voice):
            snr = rng.uniform(c.other_voice_snr_min, c.other_voice_snr_max)
            tiled = np.tile(voice, len(audio) // len(voice) + 2)
            start = int(rng.uniform(0, len(voice)))
            audio = mix_at_snr(audio, tiled[start:start + len(audio)], snr)
            self.applied.append(f"other voice {snr:.0f} dB")
        if hit(c.reverb_prob):
            rt60, wet = rng.uniform(c.rt60_min, c.rt60_max), rng.uniform(c.reverb_wet_min, c.reverb_wet_max)
            audio = reverb(audio, rt60, wet, rng)
            self.applied.append(f"reverb rt60 {rt60:.2f}s wet {wet:.2f}")
        if hit(c.eq_prob):
            audio, bands = random_eq(audio, rng, c.eq_gain_db)
            self.applied.append(f"eq {bands} band(s)")
        if hit(c.band_prob):
            telephone = rng.random() < c.telephone_share
            cutoff = None if telephone else rng.uniform(3000, 7000)
            audio = band_limit(audio, telephone, cutoff)
            self.applied.append("telephone band" if telephone else f"lowpass {cutoff:.0f} Hz")
        if hit(c.noise_prob):
            color, snr = str(rng.choice(list(c.noise_colors))), rng.uniform(c.snr_min, c.snr_max)
            audio = mix_at_snr(audio, colored_noise(len(audio), color, rng), snr)
            self.applied.append(f"{color} noise {snr:.0f} dB")
        if hit(c.gain_prob):
            gain = rng.uniform(-c.gain_db, c.gain_db)
            audio = limit_peak(audio * 10 ** (gain / 20))
            self.applied.append(f"gain {gain:+.1f} dB")
        if hit(c.clip_prob):
            drive = rng.uniform(1.5, 4.0)
            audio = soft_clip(audio, drive)
            self.applied.append(f"soft clip {drive:.1f}")
        if hit(c.codec_prob):
            kind = ("mu-law", "8 kHz", "8 kHz + mu-law")[int(rng.uniform(0, 3))]
            audio = mu_law(narrowband(audio)) if "+" in kind else mu_law(audio) if kind == "mu-law" else narrowband(audio)
            self.applied.append(kind)
        return limit_peak(audio)

    def _legacy(self, audio, rng):
        c = self.config
        if c.tempo_prob and rng.random() < c.tempo_prob:
            if not c.legacy_speed:
                candidate = tempo_perturb(audio, SAMPLE_RATE, rng.uniform(c.tempo_min, c.tempo_max))
            else:
                import torch
                import torchaudio.functional as F  # retained solely for historical recipes
                speed_factor = rng.uniform(0.92, 1.08)
                candidate = F.resample(torch.from_numpy(audio).float(), int(SAMPLE_RATE * speed_factor), SAMPLE_RATE).numpy()
            # Slowing a 29s clip can make it >30s: reject the augmentation, never crop.
            if len(candidate) <= SAMPLE_RATE * MAX_SECONDS:
                audio = candidate
                self.applied.append("legacy tempo/speed")
        if c.noise_prob and rng.random() < c.noise_prob:
            audio = add_snr_noise(audio, rng.uniform(c.snr_min, c.snr_max), rng)
            self.applied.append("legacy white noise")
        return audio

    def features(self, features, rng=None):
        c = self.config
        if c.legacy or not c.vtlp_prob:
            return features
        rng = rng or self.rng()
        if rng.random() >= c.vtlp_prob:
            return features
        return vtlp(features, rng.uniform(c.vtlp_min, c.vtlp_max))


def main(argv=None):
    p = argparse.ArgumentParser(description="Write augmented copies of one clip for listening. VTLP and "
                                            "SpecAugment act on features during training and are not audible here.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--profile", choices=PROFILES, default="speaker-robust")
    p.add_argument("--aug", action="append", default=[], metavar="FIELD=VALUE", help="e.g. --aug reverb_prob=1")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    import librosa
    import soundfile as sf
    audio, _ = librosa.load(args.input, sr=SAMPLE_RATE, mono=True)
    if not len(audio) or len(audio) > SAMPLE_RATE * MAX_SECONDS:
        p.error("Input must be non-empty and at most 30 s")
    np.random.seed(args.seed)  # legacy profile draws from the global stream
    augmenter = Augmenter(build_config(args.profile, args.aug), args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(args.out_dir / f"{args.input.stem}_original.wav", audio, SAMPLE_RATE, subtype="PCM_16")
    for index in range(args.n):
        output = augmenter.waveform(audio.astype("float32"))
        path = args.out_dir / f"{args.input.stem}_{args.profile}_{index:02d}.wav"
        sf.write(path, output, SAMPLE_RATE, subtype="PCM_16")
        print(f"{path}  [{len(output) / SAMPLE_RATE:.2f}s] {', '.join(augmenter.applied) or 'clean'}")
    print("Config:", {k: v for k, v in asdict(augmenter.config).items() if v not in (None, 0.0, False)})


if __name__ == "__main__":
    main()
