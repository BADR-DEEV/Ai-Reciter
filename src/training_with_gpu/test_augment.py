import json
import pickle
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch

from .augment import (PROFILES, SAMPLE_RATE, Augmenter, build_config, colored_noise, configure_spec_augment,
                      mix_at_snr, synthetic_rir, tempo_perturb, add_snr_noise, vtlp, vtlp_warp)
from .train_base_full import (AugmentedAyahDataset, add_extra_tokens, apply_label_overrides, load_label_overrides,
                              strip_tags, tag_scores, word_tags)

EVERY_OP = replace(PROFILES["speaker-robust"], clean_prob=0.0, tempo_prob=1.0, pitch_prob=1.0, vtlp_prob=1.0, reverb_prob=1.0,
                   eq_prob=1.0, band_prob=1.0, noise_prob=1.0, gain_prob=1.0, clip_prob=1.0, codec_prob=1.0, other_voice_prob=1.0)


def voice(seconds=2.0, f0=120.0, peak=.9):
    """Harmonic, amplitude-modulated stand-in for a voiced recitation."""
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    audio = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 20)) * (.6 + .4 * np.sin(2 * np.pi * 3 * t))
    return (audio / np.max(np.abs(audio)) * peak).astype("float32")


def old_legacy_waveform(audio, noise_prob, speed_prob, tempo_range):
    """The pre-augment.py AugmentedAyahDataset.__getitem__ waveform path, on the global np.random."""
    if speed_prob and np.random.random() < speed_prob:
        candidate = tempo_perturb(audio, SAMPLE_RATE, np.random.uniform(*tempo_range))
        if len(candidate) <= SAMPLE_RATE * 30:
            audio = candidate
    if noise_prob and np.random.random() < noise_prob:
        audio = add_snr_noise(audio, np.random.uniform(25, 35))
    return audio


class FakeTokenizer:
    def __call__(self, text):
        return SimpleNamespace(input_ids=[1, *range(2, 2 + len(text.split())), 0])


def fake_processor():
    from transformers import WhisperFeatureExtractor
    return SimpleNamespace(tokenizer=FakeTokenizer(), feature_extractor=WhisperFeatureExtractor())


class SignalDataset(torch.utils.data.Dataset):
    def __init__(self, augmenter):
        self.augmenter = augmenter

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return torch.from_numpy(self.augmenter.waveform(np.full(4000, .1, dtype=np.float32)))


class WaveformOpTests(unittest.TestCase):
    def test_every_op_keeps_length_dtype_finite_and_range(self):
        audio = voice()
        augmenter = Augmenter(EVERY_OP, seed=3)
        for _ in range(6):
            output = augmenter.waveform(audio, other_voice=lambda rng: voice(1.3, f0=210, peak=.5))
            self.assertEqual(output.dtype, np.float32)
            self.assertTrue(np.isfinite(output).all())
            self.assertLessEqual(float(np.max(np.abs(output))), 1.0)
            self.assertLessEqual(len(output), 30 * SAMPLE_RATE)
            self.assertGreater(len(output), 0)
            self.assertEqual(len(augmenter.applied), 10)  # VTLP acts on features

    def test_overlong_tempo_is_skipped_never_cropped(self):
        audio = voice(29.5, peak=.5)
        slow = replace(PROFILES["none"], tempo_prob=1.0, tempo_min=.9, tempo_max=.9)
        self.assertEqual(len(Augmenter(slow).waveform(audio)), len(audio))
        fast = replace(slow, tempo_min=1.2, tempo_max=1.2)
        self.assertEqual(len(Augmenter(fast).waveform(audio)), round(len(audio) / 1.2))

    def test_snr_is_measured_against_signal_power_for_every_color(self):
        rng = np.random.default_rng(0)
        audio = voice(peak=.1)
        for color in ("white", "pink", "brown"):
            for snr in (12, 20, 35):
                noisy = mix_at_snr(audio, colored_noise(len(audio), color, rng), snr)
                measured = 10 * np.log10(np.mean(audio.astype(np.float64) ** 2) / np.mean((noisy - audio).astype(np.float64) ** 2))
                self.assertAlmostEqual(float(measured), snr, delta=.05, msg=color)

    def test_noise_colors_have_falling_spectra(self):
        rng = np.random.default_rng(1)
        slopes = {}
        for color in ("white", "pink", "brown"):
            power = np.abs(np.fft.rfft(colored_noise(2 ** 16, color, rng))) ** 2
            low, high = power[10:100].mean(), power[10000:20000].mean()
            slopes[color] = 10 * np.log10(low / high)
        self.assertLess(abs(slopes["white"]), 3)
        self.assertGreater(slopes["pink"], 15)
        self.assertGreater(slopes["brown"], slopes["pink"] + 15)

    def test_rir_has_unit_energy_and_decays_60_db_by_rt60(self):
        rir = synthetic_rir(.5, np.random.default_rng(0))
        self.assertEqual(len(rir), SAMPLE_RATE // 2)
        self.assertAlmostEqual(float(np.sum(rir ** 2)), 1.0, places=6)
        window = len(rir) // 10
        head, tail = np.sqrt(np.mean(rir[:window] ** 2)), np.sqrt(np.mean(rir[-window:] ** 2))
        self.assertLess(20 * np.log10(tail / head), -45)

    def test_profiles_and_overrides(self):
        self.assertFalse(PROFILES["none"].active())
        self.assertEqual(PROFILES["speaker-robust"].other_voice_prob, 0.0)
        config = build_config("speaker-robust", ["pitch_prob=0.9", "noise_colors=pink,brown", "legacy=false"], snr_min=15.0, tempo_max=None)
        self.assertEqual((config.pitch_prob, config.noise_colors, config.snr_min, config.tempo_max), (.9, ("pink", "brown"), 15.0, 1.45))
        for bad in (["pitch_prob=2"], ["no_such_field=1"], ["noise_colors=violet"], ["tempo_min=2"]):
            with self.assertRaises(ValueError):
                build_config("speaker-robust", bad)
        legacy = PROFILES["legacy"]
        self.assertEqual((legacy.tempo_prob, legacy.tempo_min, legacy.tempo_max, legacy.noise_prob), (.40, .95, 1.05, .12))

    def test_augmenter_pickles_and_dataloader_workers_draw_distinct_streams(self):
        augmenter = Augmenter(replace(PROFILES["none"], noise_prob=1.0), seed=5)
        augmenter.waveform(voice())
        clone = pickle.loads(pickle.dumps(augmenter))
        self.assertEqual(clone.config, augmenter.config)
        loader = torch.utils.data.DataLoader(SignalDataset(clone), batch_size=None, num_workers=2)
        outputs = [batch.numpy() for batch in loader]
        self.assertFalse(np.array_equal(outputs[0], outputs[1]))  # first item of each worker


class LegacyProfileTests(unittest.TestCase):
    def test_legacy_profile_reproduces_previous_waveforms_from_same_rng(self):
        audio = voice(3.0, peak=.5)
        config = build_config("legacy", noise_prob=.6, tempo_prob=.6)
        for seed in range(8):
            np.random.seed(seed)
            expected = old_legacy_waveform(audio.copy(), .6, .6, (.95, 1.05))
            actual = Augmenter(config).waveform(audio.copy(), rng=np.random.RandomState(seed))
            self.assertTrue(np.array_equal(actual, expected), seed)

    def test_dataset_legacy_arguments_and_vtlp_hook_keep_feature_shape(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = []
            for index, reciter in enumerate(("a", "a", "b")):
                path = Path(folder) / f"{index}.wav"
                sf.write(path, voice(1.5 + index, peak=.4), SAMPLE_RATE, subtype="FLOAT")
                rows.append({"path": str(path), "reciter_key": reciter, "text": "كلمة اخرى"})
            processor = fake_processor()
            legacy = AugmentedAyahDataset(rows, processor, "text", .7, .7, (.95, 1.05))
            np.random.seed(11)
            item = legacy[0]
            np.random.seed(11)
            reference = processor.feature_extractor(old_legacy_waveform(legacy.cached_samples[0]["audio"].copy(), .7, .7, (.95, 1.05)),
                                                    sampling_rate=SAMPLE_RATE).input_features[0]
            self.assertTrue(np.array_equal(item["input_features"], reference))
            self.assertIsNone(AugmentedAyahDataset(rows, processor, "text", augmenter=Augmenter(PROFILES["none"])).augmenter)
            robust = AugmentedAyahDataset(rows, processor, "text", augmenter=Augmenter(EVERY_OP, seed=2))
            for index in range(3):
                item = robust[index]
                self.assertEqual(item["input_features"].shape, (80, 3000))
                self.assertTrue(np.isfinite(item["input_features"]).all())
            other = robust.other_voice("a", np.random.default_rng(0))
            self.assertTrue(np.array_equal(other, robust.cached_samples[2]["audio"]))
            self.assertIsNone(AugmentedAyahDataset(rows[:2], processor, "text").other_voice("a", np.random.default_rng(0)))
            pickle.dumps(robust)


class FeatureTests(unittest.TestCase):
    def test_vtlp_identity_shape_and_direction(self):
        features = np.random.default_rng(0).normal(size=(80, 3000)).astype("float32")
        self.assertTrue(np.array_equal(vtlp(features, 1.0), features))
        for alpha in (.88, 1.12):
            warped = vtlp(features, alpha)
            self.assertEqual((warped.shape, warped.dtype), (features.shape, features.dtype))
            self.assertFalse(np.allclose(warped, features))
        peak = np.full((80, 10), -1.0, dtype="float32")
        peak[30] = 1.0
        self.assertGreater(int(np.argmax(vtlp(peak, 1.12)[:, 0])), 30)  # formants rise: woman/child
        self.assertLess(int(np.argmax(vtlp(peak, .88)[:, 0])), 30)
        constant = np.full((80, 5), .3, dtype="float32")
        self.assertTrue(np.allclose(vtlp(constant, 1.1), constant))  # padding frames are untouched

    def test_vtlp_warp_is_monotonic_and_keeps_band_edges(self):
        grid = np.linspace(0, 8000, 2001)
        for alpha in (.88, 1.0, 1.12):
            warped = vtlp_warp(grid, alpha, 8000)
            self.assertAlmostEqual(float(warped[0]), 0.0)
            self.assertAlmostEqual(float(warped[-1]), 8000.0)
            self.assertTrue(np.all(np.diff(warped) > 0))

    def test_spec_augment_configuration(self):
        from transformers import WhisperConfig
        config = WhisperConfig(apply_spec_augment=False, mask_time_prob=.05)
        self.assertEqual(configure_spec_augment(config, PROFILES["legacy"]), {})
        self.assertFalse(config.apply_spec_augment)
        applied = configure_spec_augment(config, PROFILES["speaker-robust"])
        self.assertTrue(config.apply_spec_augment)
        self.assertEqual((config.mask_time_prob, config.mask_feature_prob, config.mask_feature_length), (.05, .05, 10))
        self.assertEqual(applied["mask_time_length"], 10)
        configure_spec_augment(config, PROFILES["none"])
        self.assertFalse(config.apply_spec_augment)


class TargetTests(unittest.TestCase):
    def test_labels_jsonl_join_prefers_reciter_path_then_ayah(self):
        rows = [{"reciter_key": r, "surah": 2, "ayah": a, "relative_audio_path": f"audio/002{a:03d}.wav"}
                for r in ("huthaify", "husary") for a in (5, 6, 7)]
        entries = [{"reciter_key": "huthaify", "relative_audio_path": "audio\\002005.wav", "text": "exact"},
                   {"relative_audio_path": "audio/002005.wav", "text": "any-reciter-path"},
                   {"reciter_key": "husary", "surah": 2, "ayah": 6, "text_tajweed": "reciter-ayah"},
                   {"surah": 2, "ayah": 6, "text": "any-ayah"},
                   {"surah": 2, "ayah": 6, "text": "any-ayah"}]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "labels.jsonl"
            path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n\n", encoding="utf-8")
            joined = apply_label_overrides(rows, load_label_overrides(path, "text_tajweed"), "text_tajweed")
            path.write_text(json.dumps({"surah": 1, "ayah": 1, "text": "a"}) + "\n" + json.dumps({"surah": 1, "ayah": 1, "text": "b"}))
            with self.assertRaisesRegex(ValueError, "conflicting"):
                load_label_overrides(path, "text_tajweed")
            path.write_text(json.dumps({"reciter_key": "husary", "text": "a"}))
            with self.assertRaisesRegex(ValueError, "relative_audio_path or surah"):
                load_label_overrides(path, "text_tajweed")
        self.assertEqual([r.get("text_tajweed") for r in joined],
                         ["exact", "any-ayah", None, "any-reciter-path", "reciter-ayah", None])
        self.assertNotIn("text_tajweed", rows[0])

    def test_tags_are_stripped_for_wer_and_scored_per_word(self):
        self.assertEqual(strip_tags("<tj:x>انَّ<tj:ghunna> اللَّهَ <tj:madd>  غفور"), "انَّ اللَّهَ غفور")
        self.assertEqual(word_tags("a <tj:x> b<tj:y><tj:z>"), [("a", ["<tj:x>"]), ("b", ["<tj:y>", "<tj:z>"])])
        self.assertEqual(word_tags("<tj:x>a b"), [("a", ["<tj:x>"]), ("b", [])])
        scores = tag_scores([("a<tj:x> b c<tj:y><tj:z>", "a <tj:x> b<tj:y> c<tj:z>")])
        self.assertAlmostEqual(scores["precision"], 2 / 3)
        self.assertAlmostEqual(scores["recall"], 2 / 3)
        self.assertEqual(scores["per_tag"]["<tj:y>"], {"precision": 0.0, "recall": 0.0, "f1": 0.0, "reference_tags": 1, "predicted_tags": 1})
        deleted = tag_scores([("a<tj:x> b<tj:y>", "a<tj:x>")])
        self.assertEqual((deleted["precision"], deleted["recall"]), (1.0, .5))
        self.assertIsNone(tag_scores([("a b", "a b")])["precision"])


def tiny_whisper(folder):
    """Self-contained byte-level Whisper tokenizer with the real special-token layout, and a tiny model."""
    from transformers import (WhisperConfig, WhisperFeatureExtractor, WhisperForConditionalGeneration,
                              WhisperProcessor, WhisperTokenizer)
    from transformers.models.whisper.tokenization_whisper import LANGUAGES, bytes_to_unicode
    vocab = {c: i for i, c in enumerate(bytes_to_unicode().values())}
    vocab["<|endoftext|>"] = len(vocab)
    (folder / "vocab.json").write_text(json.dumps(vocab))
    (folder / "merges.txt").write_text("#version: 0.2\n")
    tokenizer = WhisperTokenizer(folder / "vocab.json", folder / "merges.txt")
    tokenizer.add_special_tokens({"additional_special_tokens": [
        "<|startoftranscript|>", *[f"<|{code}|>" for code in LANGUAGES], "<|translate|>", "<|transcribe|>",
        "<|startoflm|>", "<|startofprev|>", "<|nospeech|>", "<|notimestamps|>"]})
    tokenizer.add_tokens(["<|0.00|>", "<|0.02|>"])
    WhisperProcessor(feature_extractor=WhisperFeatureExtractor(), tokenizer=tokenizer).save_pretrained(folder / "base")
    processor = WhisperProcessor.from_pretrained(folder / "base", language="arabic", task="transcribe")
    tokenizer = processor.tokenizer
    eot, sot = tokenizer.convert_tokens_to_ids("<|endoftext|>"), tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    torch.manual_seed(0)
    model = WhisperForConditionalGeneration(WhisperConfig(
        vocab_size=len(tokenizer), d_model=16, encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
        decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32, num_mel_bins=80, max_source_positions=16,
        max_target_positions=32, pad_token_id=eot, bos_token_id=eot, eos_token_id=eot, decoder_start_token_id=sot))
    return processor, model


class ExtraTokenTests(unittest.TestCase):
    def test_add_resize_initialise_save_reload_and_decode(self):
        from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList, WhisperForConditionalGeneration, WhisperProcessor
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            processor, model = tiny_whisper(folder)
            tokenizer, old = processor.tokenizer, model.config.vocab_size
            mean = model.get_input_embeddings().weight[:old].mean(0).detach().clone()
            tags = ["<tj:ghunna>", "<tj:madd>"]
            ids = add_extra_tokens(model, tokenizer, tags)
            self.assertEqual(ids, [old, old + 1])
            self.assertEqual(model.config.vocab_size, old + 2)
            self.assertIs(model.get_output_embeddings().weight, model.get_input_embeddings().weight)
            self.assertTrue(torch.allclose(model.get_input_embeddings().weight[old:], mean.expand(2, -1)))
            with self.assertRaisesRegex(ValueError, "Already"):
                add_extra_tokens(model, tokenizer, ["<tj:madd>"])
            text = "انَّ<tj:ghunna> الله<tj:madd>"
            labels = tokenizer(text).input_ids
            self.assertIn(old, labels)
            self.assertEqual(tokenizer.decode(labels, skip_special_tokens=True), text)
            output = model(input_features=torch.zeros(1, 80, 32), labels=torch.tensor([labels[1:]]))
            self.assertEqual(output.logits.shape[-1], old + 2)
            self.assertTrue(torch.isfinite(output.loss))

            gen = dict(decoder_start_token_id=model.config.decoder_start_token_id, eos_token_id=tokenizer.eos_token_id,
                       pad_token_id=tokenizer.eos_token_id, no_timestamps_token_id=tokenizer.convert_tokens_to_ids("<|notimestamps|>"),
                       lang_to_id={"<|ar|>": tokenizer.convert_tokens_to_ids("<|ar|>")},
                       task_to_id={"transcribe": tokenizer.convert_tokens_to_ids("<|transcribe|>")}, is_multilingual=True, max_length=32)
            model.generation_config = GenerationConfig(**gen, force_unique_generate_call=True)
            model.save_pretrained(folder / "run")
            processor.save_pretrained(folder / "run")
            reloaded = WhisperProcessor.from_pretrained(folder / "run")
            again = WhisperForConditionalGeneration.from_pretrained(folder / "run")
            self.assertEqual(reloaded.tokenizer.convert_tokens_to_ids(tags), ids)
            self.assertEqual(reloaded.tokenizer.decode(labels, skip_special_tokens=True), text)
            self.assertTrue(torch.equal(again.get_input_embeddings().weight, model.get_input_embeddings().weight))
            self.assertTrue(again.generation_config.force_unique_generate_call)

            # Consecutive tag IDs sit above Whisper's timestamp range; one generate call must keep them intact.
            content = [*tokenizer("ان", add_special_tokens=False).input_ids, *ids, *tokenizer(" الله", add_special_tokens=False).input_ids]

            class Script(LogitsProcessor):
                def __call__(self, input_ids, scores):
                    forced = torch.full_like(scores, -float("inf"))
                    step = input_ids.shape[1] - 4
                    forced[:, (content + [tokenizer.eos_token_id])[min(step, len(content))]] = 0
                    return forced
            sequence = again.generate(torch.zeros(1, 80, 32), language="ar", task="transcribe",
                                      logits_processor=LogitsProcessorList([Script()]))
            self.assertEqual(reloaded.tokenizer.decode(sequence[0], skip_special_tokens=True), "ان<tj:ghunna><tj:madd> الله")

    def test_tokenizer_must_cover_the_whole_output_vocabulary(self):
        with tempfile.TemporaryDirectory() as folder:
            processor, model = tiny_whisper(Path(folder))
            model.resize_token_embeddings(model.config.vocab_size + 3)
            with self.assertRaisesRegex(ValueError, "collide"):
                add_extra_tokens(model, processor.tokenizer, ["<tj:ghunna>"])


if __name__ == "__main__":
    unittest.main()
