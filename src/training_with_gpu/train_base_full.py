"""Full Whisper base/tiny Qaloon adaptation from pinned OpenAI/Tarteel/DeepDML starts or a local checkpoint."""

import argparse
from collections import defaultdict
from dataclasses import asdict
import difflib
import hashlib
import json
import math
import re
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
sys.path.insert(0, str(TRAINING_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qaloon_data import RECITER_DIRS, default_data_root, load_splits, register_reader_root, speaker_disjoint_splits
from augment import PROFILES, Augmenter, build_config, configure_spec_augment, add_snr_noise, tempo_perturb  # noqa: F401

TARTEEL_MODEL = "tarteel-ai/whisper-base-ar-quran"
TARTEEL_REVISION = "5c3c53fdf9272c4f6ee0bee09a1e5a4a615ee25c"
OPENAI_REVISION = "e37978b90ca9030d5170a5c07aadb050351a65bb"
TARTEEL_TINY_MODEL = "tarteel-ai/whisper-tiny-ar-quran"
DEEPDML_BASE = "deepdml/whisper-base-ar-quran-mix-norm"
DEEPDML_SMALL = "deepdml/whisper-small-ar-quran-mix-norm"
MODEL_REVISIONS = {
    DEEPDML_BASE: "23758f3e6877ce46077962c2e31a14794e700f1f",
    DEEPDML_SMALL: "2d0929ee4d62cf3642e4184c16e9cb939d86ea78",
    TARTEEL_MODEL: TARTEEL_REVISION,
    "openai/whisper-base": OPENAI_REVISION,
    TARTEEL_TINY_MODEL: "c3d7e624af5c81bef25a10a2af3a5a84cb4dd0f0",
    "openai/whisper-tiny": "169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
}
TAG_PATTERN = re.compile(r"<tj:[^<>\s]+>")


def configure_generation(model, processor, template=None):
    """Upgrade legacy decoding metadata only; never substitute model weights."""
    if template is None:
        from transformers import GenerationConfig
        template = GenerationConfig.from_pretrained("openai/whisper-base", revision=OPENAI_REVISION)
    vocabulary_ids = set(processor.tokenizer.get_vocab().values())
    if not vocabulary_ids or min(vocabulary_ids) < 0 or max(vocabulary_ids) >= model.config.vocab_size:
        raise ValueError("Checkpoint/processor vocabulary IDs exceed model embeddings")
    for token, expected in (("<|ar|>", template.lang_to_id["<|ar|>"]),
                            ("<|transcribe|>", template.task_to_id["transcribe"]),
                            ("<|notimestamps|>", template.no_timestamps_token_id)):
        if processor.tokenizer.convert_tokens_to_ids(token) != expected:
            raise ValueError(f"Modern decoding template/processor token mismatch: {token}")
    if max(vocabulary_ids) + 1 != model.config.vocab_size:
        # This 2022 Tarteel processor contains all text/control tokens, but omits
        # the 1,501 timestamp tokens at the tail of Whisper's output vocabulary.
        # Keep its matching text tokenizer and suppress ONLY those absent IDs.
        # Any other mismatch fails; never resize/reinitialize trained embeddings.
        if (max(vocabulary_ids) != template.no_timestamps_token_id
                or vocabulary_ids != set(range(template.no_timestamps_token_id + 1))
                or model.config.vocab_size - len(vocabulary_ids) != 1501):
            raise ValueError("Unsupported checkpoint/processor vocabulary mismatch")
        template.suppress_tokens = sorted(set(template.suppress_tokens or []) | set(range(len(vocabulary_ids), model.config.vocab_size)))
    model.generation_config = template
    model.generation_config.language = "arabic"
    model.generation_config.task = "transcribe"
    model.generation_config.return_timestamps = False
    model.generation_config.max_length = model.config.max_target_positions
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    # Prevent save_pretrained from copying the legacy max_length=1024 back into
    # the modern generation config. 20 is the deprecated config-field default;
    # the ACTUAL decoding maximum lives above in generation_config (448).
    model.config.max_length = 20


def parse_args(argv=None, *, default_model="openai/whisper-base", default_reciters=None, default_output="runs/gpu_base_full", default_dokali_weight=.35):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, help="Folder holding dataset_qaloon_* readers (default: src/dataset_collection, else data/hf/qaloon-reciter-dataset)")
    p.add_argument("--reader-root", type=Path, action="append", default=[],
                   help="Also register every dataset_qaloon_<name> folder here as reciter <name> (repeatable)")
    p.add_argument("--output-dir", type=Path, default=Path(default_output))
    p.add_argument("--init-model", "--base-model", default=default_model,
                   help="Pinned hub id, or a local checkpoint directory such as runs/rattil_qaloon_v3")
    p.add_argument("--model-revision", help="Override pinned initialization revision deliberately")
    p.add_argument("--reciters", nargs="+", default=default_reciters)
    p.add_argument("--trabulsi-reviewed-manifest", type=Path)
    p.add_argument("--trabulsi-permission-file", type=Path)
    p.add_argument("--trabulsi-validation-report", type=Path, help="Complete strict WAV/text audit of the corrected dataset")
    p.add_argument("--dokali-weight", type=float, default=default_dokali_weight)
    p.add_argument("--augment-profile", choices=PROFILES, default="speaker-robust")
    p.add_argument("--aug", action="append", default=[], metavar="FIELD=VALUE",
                   help="Override one AugmentConfig field, e.g. --aug pitch_prob=0.5 (repeatable)")
    p.add_argument("--noise-prob", type=float, help="Legacy default .12; overrides noise_prob in any profile")
    p.add_argument("--speed-prob", type=float, help="Probability of tempo perturbation (legacy default .40)")
    p.add_argument("--tempo-min", type=float, help="Legacy default .95")
    p.add_argument("--tempo-max", type=float, help="Legacy default 1.05")
    p.add_argument("--mask-time-prob", type=float, help="SpecAugment; default from the profile")
    p.add_argument("--mask-time-length", type=int)
    p.add_argument("--mask-feature-prob", type=float)
    p.add_argument("--mask-feature-length", type=int)
    p.add_argument("--label-field", default="text_asr_normalized", help="Metadata field used as the decoder target")
    p.add_argument("--labels-jsonl", type=Path, help="Targets joined by reciter_key+relative_audio_path or surah+ayah into --label-field")
    p.add_argument("--extra-tokens", type=Path, help="One token per line, added as NON-special tokens (e.g. <tj:ghunna>)")
    p.add_argument("--epochs", type=int, default=7)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1.25e-5, help="Standard safe LR for full fine-tuning")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--validation-reciter")
    p.add_argument("--test-reciter")
    p.add_argument("--holdout-reciter", action="append", default=[],
                   help="Keep this reciter out of training/selection entirely; report its WER at the end (repeatable)")
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    p.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto",
                   help="auto: bf16/fp16 on CUDA, fp32 on MPS/CPU")
    p.add_argument("--dataloader-workers", type=int, default=0)
    p.add_argument("--logging-steps", type=int, default=25)
    p.add_argument("--dry-run", action="store_true", help="Validate splits without loading a model or training")
    p.add_argument("--max-steps", type=int, default=-1, help="Positive value for an explicitly limited smoke experiment")
    p.add_argument("--max-samples-per-split", type=int, help="Stratified smoke subset; not a benchmark")
    p.add_argument("--max-train-clips", type=int, help="Reader-balanced cap on training clips (smoke runs); overrides --max-samples-per-split for train")
    args = p.parse_args(argv)
    args.data_root = args.data_root or default_data_root()
    try:
        for root in args.reader_root:
            register_reader_root(root)
        augmentation_config(args)
    except ValueError as error:
        p.error(str(error))
    named = [*(args.reciters or []), *args.holdout_reciter, *(r for r in (args.validation_reciter, args.test_reciter) if r)]
    if set(named) - RECITER_DIRS.keys():
        p.error(f"Unknown reciters {sorted(set(named) - RECITER_DIRS.keys())}; known: {sorted(RECITER_DIRS)} (add folders with --reader-root)")
    if set(args.holdout_reciter) & {args.validation_reciter, args.test_reciter}:
        p.error("A held-out reciter cannot also be the validation/test reciter")
    if args.init_model not in MODEL_REVISIONS and not args.model_revision and not Path(args.init_model).is_dir():
        p.error("--init-model must be a pinned hub id, a local checkpoint directory, or come with --model-revision")
    if args.labels_jsonl and args.label_field == "text_asr_normalized":
        p.error("Give --labels-jsonl targets their own --label-field (e.g. text_tajweed); text_asr_normalized stays the WER reference")
    if args.device in ("mps", "cpu") and args.precision in ("fp16", "bf16"):
        p.error("fp16/bf16 training needs CUDA; MPS/CPU train in fp32")
    if any((args.trabulsi_reviewed_manifest, args.trabulsi_permission_file, args.trabulsi_validation_report)) and not all((args.trabulsi_reviewed_manifest, args.trabulsi_permission_file, args.trabulsi_validation_report)):
        p.error("Pass reviewed Trabulsi manifest, source permission receipt AND strict validation report")
    if not math.isfinite(args.dokali_weight) or not 0 < args.dokali_weight:
        p.error("Require positive Dokali weight")
    if any(value is not None and value <= 0 for value in (args.max_samples_per_split, args.max_train_clips)):
        p.error("max-samples-per-split and max-train-clips must be positive")
    if (min(args.batch_size, args.eval_batch_size, args.gradient_accumulation, args.epochs, args.patience, args.logging_steps) <= 0
            or args.dataloader_workers < 0
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or args.max_steps == 0 or args.max_steps < -1):
        p.error("Training sizes/epochs/patience/LR must be positive; max-steps is -1 or positive")
    return args


def augmentation_config(args):
    return build_config(args.augment_profile, args.aug, noise_prob=args.noise_prob, tempo_prob=args.speed_prob,
                        tempo_min=args.tempo_min, tempo_max=args.tempo_max,
                        mask_time_prob=args.mask_time_prob, mask_time_length=args.mask_time_length,
                        mask_feature_prob=args.mask_feature_prob, mask_feature_length=args.mask_feature_length)


def resolve_device(device, precision):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available() or device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(f"{device} requested but unavailable")
    if precision == "auto":
        precision = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if device == "cuda" else "fp32"
    if precision != "fp32" and device != "cuda" or precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"{precision} is unsupported on {device}")
    return device, precision


def round_robin(rows, limit):
    """Alternate readers so a smoke subset never covers only the first reader in metadata order."""
    groups = {reader: [r for r in rows if r["reciter_key"] == reader] for reader in sorted({r["reciter_key"] for r in rows})}
    subset = []
    while groups and len(subset) < limit:
        for reader in list(groups):
            subset.append(groups[reader].pop(0))
            if not groups[reader]:
                del groups[reader]
            if len(subset) == limit:
                break
    return subset


def hold_out_reciters(splits, reciters):
    """Move whole reciters out of fitting and selection into a final-only 'holdout' split."""
    held = set(reciters)
    rows = [row for partition in splits.values() for row in partition]
    result = {name: [r for r in partition if r["reciter_key"] not in held] for name, partition in splits.items()}
    holdout = [r for r in rows if r["reciter_key"] in held]
    missing = held - {r["reciter_key"] for r in holdout}
    if missing or any(not partition for partition in result.values()):
        raise ValueError(f"Holdout produced an empty partition (missing reciters: {sorted(missing)})")
    texts = {r["text_asr_normalized"] for r in result["train"]}
    result["holdout"] = [dict(r, text_seen_in_training=r["text_asr_normalized"] in texts) for r in holdout]
    return result


def load_label_overrides(path, field):
    """JSONL -> {join key: target}. Each line holds the target under `field` (or "text") plus
    relative_audio_path or surah+ayah, optionally narrowed by reciter_key."""
    table = {}
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            entry = json.loads(line)
            text = entry.get(field, entry.get("text"))
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}:{number}: needs a non-empty '{field}' or 'text'")
            reciter = entry.get("reciter_key")
            if entry.get("relative_audio_path"):
                key = ("path", reciter, entry["relative_audio_path"].replace("\\", "/"))
            elif entry.get("surah") is not None and entry.get("ayah") is not None:
                key = ("ayah", reciter, int(entry["surah"]), int(entry["ayah"]))
            else:
                raise ValueError(f"{path}:{number}: needs relative_audio_path or surah+ayah")
            if table.setdefault(key, text.strip()) != text.strip():
                raise ValueError(f"{path}:{number}: conflicting duplicate target for {key}")
    return table


def apply_label_overrides(rows, table, field):
    """Most specific key wins: reciter+path, reciter+ayah, path (any reciter), ayah (any reciter)."""
    result = []
    for row in rows:
        path = (row.get("relative_audio_path") or "").replace("\\", "/") or None
        reciter, ayah = row["reciter_key"], (row["surah"], row["ayah"])
        keys = (("path", reciter, path), ("ayah", reciter, *ayah), ("path", None, path), ("ayah", None, *ayah))
        text = next((table[key] for key in keys if key in table), None)
        result.append(row if text is None else dict(row, **{field: text}))
    return result


def read_extra_tokens(path):
    tokens = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    if not tokens or len(set(tokens)) != len(tokens) or any(any(c.isspace() for c in token) for token in tokens):
        raise ValueError(f"{path}: need unique, whitespace-free tokens, one per line")
    untagged = [token for token in tokens if not TAG_PATTERN.fullmatch(token)]
    if untagged:
        print(f"WARNING: {untagged[:5]} are not <tj:NAME> tags; WER will NOT strip them.")
    return tokens


def add_extra_tokens(model, tokenizer, tokens):
    """Add NON-special tokens (they survive skip_special_tokens=True); new rows start at the embedding mean."""
    if len(tokenizer) != model.config.vocab_size:
        raise ValueError(f"Extra tokens need a tokenizer covering the whole output vocabulary ({len(tokenizer)} != "
                         f"{model.config.vocab_size}); new IDs would collide with existing embedding rows")
    present = [token for token in tokens if token in tokenizer.get_vocab()]
    if present:
        raise ValueError(f"Already in the vocabulary: {present[:5]}")
    old = model.config.vocab_size
    tokenizer.add_tokens(tokens)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    with torch.no_grad():
        for embeddings in {id(e): e for e in (model.get_input_embeddings().weight, model.get_output_embeddings().weight)}.values():
            embeddings[old:] = embeddings[:old].mean(0)
    return tokenizer.convert_tokens_to_ids(tokens)


def strip_tags(text):
    """Tags never count as words: remove them before WER."""
    return " ".join(TAG_PATTERN.sub(" ", text).split())


def word_tags(text):
    """[(word, tags)]; a tag belongs to the word it follows (leading tags go to the first word)."""
    words, pending = [], []
    for token in re.findall(r"<tj:[^<>\s]+>|[^\s<]+", text):
        if TAG_PATTERN.fullmatch(token):
            (words[-1][1] if words else pending).append(token)
        else:
            words.append((token, pending))
            pending = []
    return words


def tag_scores(pairs, normalize=None):
    """Per-word tag precision/recall over (target, prediction) pairs, aligning tag-stripped words."""
    normalize = normalize or (lambda word: word)
    counts = defaultdict(lambda: [0, 0, 0])  # tag -> true positives, false positives, false negatives
    for target, prediction in pairs:
        ref, hyp = word_tags(target), word_tags(prediction)
        matcher = difflib.SequenceMatcher(None, [normalize(w) for w, _ in ref], [normalize(w) for w, _ in hyp], autojunk=False)
        for _, i1, i2, j1, j2 in matcher.get_opcodes():
            for k in range(max(i2 - i1, j2 - j1)):
                expected = set(ref[i1 + k][1]) if k < i2 - i1 else set()
                found = set(hyp[j1 + k][1]) if k < j2 - j1 else set()
                for tag in expected & found:
                    counts[tag][0] += 1
                for tag in found - expected:
                    counts[tag][1] += 1
                for tag in expected - found:
                    counts[tag][2] += 1

    def summary(tp, fp, fn):
        precision, recall = (tp / (tp + fp) if tp + fp else None), (tp / (tp + fn) if tp + fn else None)
        f1 = None if precision is None or recall is None else 2 * precision * recall / (precision + recall) if tp else 0.0
        return {"precision": precision, "recall": recall, "f1": f1, "reference_tags": tp + fn, "predicted_tags": tp + fp}
    totals = [sum(c[i] for c in counts.values()) for i in range(3)]
    return {**summary(*totals), "per_tag": {tag: summary(*c) for tag, c in sorted(counts.items())}}


class AugmentedAyahDataset:
    """In-memory cached dataset: waveform augmentation, feature extraction, then feature-level VTLP."""
    def __init__(self, rows, processor, label_field, noise_prob=0.0, speed_prob=0.0, tempo_range=None, augmenter=None):
        self.processor = processor
        self.label_field = label_field
        if tempo_range is not None and not 0 < tempo_range[0] <= tempo_range[1]:
            raise ValueError("Positive ordered tempo range required")
        if augmenter is None and (noise_prob or speed_prob):
            tempo = dict(zip(("tempo_min", "tempo_max"), tempo_range)) if tempo_range is not None else {}
            augmenter = Augmenter(build_config("legacy", noise_prob=noise_prob, tempo_prob=speed_prob,
                                               legacy_speed=tempo_range is None, **tempo))
        self.augmenter = augmenter if augmenter is not None and augmenter.config.active() else None
        self.cached_samples = []
        self.by_reciter = defaultdict(list)

        print(f"Pre-caching {len(rows)} audio files into RAM...")
        for row in rows:
            audio, sr = sf.read(row["path"], dtype="float32")
            if sr != 16000 or audio.ndim != 1:
                raise ValueError(f"Expected 16 kHz mono WAV: {row['path']}")
            if not len(audio) or len(audio) > sr * 30 or not np.isfinite(audio).all():
                raise ValueError(f"Invalid or >30s audio (never silently truncate): {row['path']}")

            labels = self.processor.tokenizer(row[self.label_field]).input_ids
            if len(labels) > 448:
                raise ValueError(f"Whisper target exceeds 448-token decoder context: {row['path']}")
            self.by_reciter[row.get("reciter_key", "")].append(len(self.cached_samples))
            self.cached_samples.append({
                "audio": audio,
                "sr": sr,
                "labels": labels,
                "reciter_key": row.get("reciter_key", "")
            })
        print("RAM pre-caching complete.")

    def __len__(self):
        return len(self.cached_samples)

    def other_voice(self, reciter, rng):
        """A random cached clip by a different reciter (background-voice augmentation)."""
        others = sorted(r for r in self.by_reciter if r != reciter)
        if not others:
            return None
        pool = self.by_reciter[others[int(rng.uniform(0, len(others)))]]
        return self.cached_samples[pool[int(rng.uniform(0, len(pool)))]]["audio"]

    def __getitem__(self, index):
        sample = self.cached_samples[index]
        audio = sample["audio"].copy()
        sr = sample["sr"]
        if self.augmenter is not None:
            audio = self.augmenter.waveform(audio, other_voice=lambda rng: self.other_voice(sample["reciter_key"], rng))
        extracted = self.processor.feature_extractor(audio, sampling_rate=sr, return_attention_mask=True)
        features = extracted.input_features[0]
        if self.augmenter is not None:
            features = self.augmenter.features(features)  # VTLP: frequency warp of the log-mel features
        return {
            "input_features": features,
            "attention_mask": extracted.attention_mask[0],
            "labels": sample["labels"],
        }


def jsonable(value):
    return str(value) if isinstance(value, Path) else [jsonable(v) for v in value] if isinstance(value, (list, tuple)) else value


def main(argv=None, **defaults):
    args = parse_args(argv, **defaults)
    augmentation = augmentation_config(args)
    readers = set(args.reciters or ["huthaify", "dokali", "husary"])
    readers.update(r for r in (args.validation_reciter, args.test_reciter, *args.holdout_reciter) if r)
    splits, skipped = load_splits(args.data_root, sorted(readers), None, args.seed)
    if args.trabulsi_reviewed_manifest:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src.training.reviewed_audio import load_reviewed_trabulsi
        reviewed = load_reviewed_trabulsi(args.trabulsi_reviewed_manifest, args.trabulsi_permission_file, args.seed,
                                         validation_report=args.trabulsi_validation_report)
        for partition in splits:
            splits[partition].extend(reviewed[partition])
    if bool(args.validation_reciter) != bool(args.test_reciter):
        raise ValueError("Pass both --validation-reciter and --test-reciter for speaker-disjoint training")
    if args.test_reciter:
        splits = speaker_disjoint_splits(splits, args.validation_reciter, args.test_reciter)
    if args.holdout_reciter:
        splits = hold_out_reciters(splits, args.holdout_reciter)
    if any(row["reciter_key"] == "waleed" for row in splits["train"]):
        raise ValueError("Waleed is adaptation-held-out and must never enter fitting")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.training.label_integrity import corrected_label_overlay
    from src.dataset_collection.qaloon_audio2text import normalize_quran_for_asr, NORMALIZER_VERSION
    label_changes = []
    for name in splits:
        splits[name], changes = corrected_label_overlay(splits[name])
        label_changes.extend(changes)
    if args.labels_jsonl:
        table = load_label_overrides(args.labels_jsonl, args.label_field)
        for name in splits:
            splits[name] = apply_label_overrides(splits[name], table, args.label_field)
        print("Labels JSONL:", {name: f"{sum(bool(r.get(args.label_field)) for r in rows)}/{len(rows)}" for name, rows in splits.items()})
    unlabeled = [f"{r['reciter_key']} {r['surah']}:{r['ayah']}" for name in ("train", "validation") for r in splits[name] if not r.get(args.label_field)]
    if unlabeled:
        raise ValueError(f"{len(unlabeled)} training/validation clips lack '{args.label_field}', e.g. {unlabeled[:5]}")
    for name, rows in splits.items():
        limit = args.max_train_clips if name == "train" and args.max_train_clips else args.max_samples_per_split
        if limit:
            splits[name] = round_robin(rows, limit)
    extra_tokens = read_extra_tokens(args.extra_tokens) if args.extra_tokens else []
    tagged = bool(extra_tokens) or any(TAG_PATTERN.search(r[args.label_field]) for r in splits["train"])
    counts = {k: len(v) for k, v in splits.items()}
    protocol = ("speaker-disjoint-adaptation-pretraining-overlap-unknown" if args.test_reciter else
                "mixed-existing-ayah-split-Trabulsi-recording-groups-not-unseen-ayah" if args.trabulsi_reviewed_manifest else "known-voice-unseen-ayah")
    if args.holdout_reciter:
        protocol += "+whole-reciter-holdout:" + ",".join(sorted(args.holdout_reciter))
    local_init = Path(args.init_model).is_dir()
    print("Split counts:", counts, "Excluded >30s:", skipped)
    print("Reciters:", {k: sorted({row["reciter_key"] for row in rows}) for k, rows in splits.items()})
    print("Initialization:", args.init_model, "revision:", args.model_revision or MODEL_REVISIONS.get(args.init_model, "local checkpoint"))
    print("Augmentation:", args.augment_profile, {k: v for k, v in asdict(augmentation).items() if v not in (None, 0.0, False)})
    if args.dry_run:
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Choose a NEW empty output directory; never overwrite existing runs/reviewer work")

    from torch.utils.data import WeightedRandomSampler
    from train_qaloon_lora import WhisperCollator
    from transformers import (EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments,
                              WhisperForConditionalGeneration, WhisperProcessor, set_seed)

    device, precision = resolve_device(args.device, args.precision)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), "VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), "GB", precision)
    else:
        print("Device:", device, precision)

    model_id = args.init_model
    revision = args.model_revision or MODEL_REVISIONS.get(model_id)
    set_seed(args.seed)
    processor = WhisperProcessor.from_pretrained(model_id, revision=revision, language="arabic", task="transcribe", trust_remote_code=False)
    processor.tokenizer.set_prefix_tokens(language="arabic", task="transcribe")

    # FULL FINE-TUNING: Load entire model without PEFT/LoRA
    model = WhisperForConditionalGeneration.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    model.config.use_cache = False
    extra_token_ids = add_extra_tokens(model, processor.tokenizer, extra_tokens) if extra_tokens else []
    configure_generation(model, processor)
    if extra_tokens:
        # New IDs sit above Whisper's timestamp range; one generate call keeps consecutive tags
        # from being read as timestamp segment boundaries in the tensor-returning API.
        model.generation_config.force_unique_generate_call = True
        print(f"Added {len(extra_tokens)} non-special tokens, ids {extra_token_ids[0]}-{extra_token_ids[-1]}")
    spec_augment = configure_spec_augment(model.config, augmentation)
    print("SpecAugment:", spec_augment or "checkpoint settings kept")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Training FULL MODEL: {trainable_params:,} learnable parameters; fixed sinusoidal positions remain fixed (no LoRA/adapters).")

    weights = [args.dokali_weight if row["reciter_key"] == "dokali" else 1.0 for row in splits["train"]]

    class WeightedTrainer(Seq2SeqTrainer):
        def _get_train_sampler(self, *sampler_args, **sampler_kwargs):
            return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    def normalize_word(word):
        return normalize_quran_for_asr(word).replace(" ", "")

    def generated_metrics(prediction):
        from src.training.asr_metrics import report_predictions
        ids = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
        ids = np.where(ids == -100, processor.tokenizer.pad_token_id, ids)
        hypotheses = processor.tokenizer.batch_decode(ids, skip_special_tokens=True)
        if len(hypotheses) != len(splits["validation"]):
            raise ValueError("Generated development predictions differ from frozen rows")
        details = [{"reciter": row["reciter_key"], "surah": row["surah"],
                    "reference": row["text_asr_normalized"], "prediction": normalize_quran_for_asr(strip_tags(text))}
                   for row, text in zip(splits["validation"], hypotheses)]
        scores = report_predictions(details)
        metrics = {"wer": scores["overall"]["wer"], "cer": scores["overall"]["cer"],
                   "macro_reciter_wer": scores["macro_reciter_wer"]}
        if tagged:
            tags = tag_scores([(row[args.label_field], text) for row, text in zip(splits["validation"], hypotheses)], normalize_word)
            metrics.update({f"tag_{k}": tags[k] for k in ("precision", "recall", "f1") if tags[k] is not None})
        return metrics

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.10,
        weight_decay=0.01,
        bf16=precision == "bf16",
        fp16=precision == "fp16",
        use_cpu=device == "cpu",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_reciter_wer",
        greater_is_better=False,
        remove_unused_columns=False,
        label_names=["labels"],
        predict_with_generate=True,
        generation_max_length=448,
        generation_num_beams=3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        max_grad_norm=1.0,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=device == "cuda",
        report_to="none",
        seed=args.seed,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=AugmentedAyahDataset(splits["train"], processor, args.label_field, augmenter=Augmenter(augmentation, args.seed)),
        eval_dataset=AugmentedAyahDataset(splits["validation"], processor, args.label_field),
        data_collator=WhisperCollator(processor, model.config.decoder_start_token_id),
        compute_metrics=generated_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pretraining_overlap = ("unknown-local-checkpoint" if local_init else "unknown-model-card-incomplete"
                           if model_id.startswith(("tarteel-ai/", "deepdml/")) else "not-a-quran-specialist-checkpoint")
    manifest = {"init_model": model_id, "init_revision": revision or getattr(model.config, "_commit_hash", None),
        "init_local_checkpoint": local_init,
        "fine_tuning": "full-all-learnable-layers", "seed": args.seed, "label_field": args.label_field,
        "labels_jsonl": {"path": str(args.labels_jsonl), "sha256": hashlib.sha256(args.labels_jsonl.read_bytes()).hexdigest()} if args.labels_jsonl else None,
        "extra_tokens": dict(zip(extra_tokens, extra_token_ids)), "tag_metrics": tagged,
        "augmentation": {"profile": args.augment_profile, **asdict(augmentation)}, "spec_augment": spec_augment,
        "device": device, "precision": precision,
        "normalizer_version": NORMALIZER_VERSION, "label_overlay_changes": label_changes,
        "selection": "generated-macro-reader-development-WER-beam3-not-cross-entropy",
        "fixed_parameters": [name for name, parameter in model.named_parameters() if not parameter.requires_grad],
        "generation_template": {"model": "openai/whisper-base", "revision": OPENAI_REVISION, "purpose": "language-task-token-metadata-only-not-weights"},
        "split_protocol": protocol, "holdout_reciters": sorted(args.holdout_reciter),
        "max_steps": args.max_steps, "max_samples_per_split": args.max_samples_per_split, "max_train_clips": args.max_train_clips,
        "is_smoke_run": args.max_steps > 0 or args.max_samples_per_split is not None or args.max_train_clips is not None,
        "tarteel_pretraining_data_overlap": pretraining_overlap,
        "trabulsi_admitted": bool(args.trabulsi_reviewed_manifest),
        "splits": {name: [{**{k: row.get(k) for k in ("surah", "ayah", "reciter_key", "path", "source_sha256", "permission_receipt_sha256", "validation_report_sha256", "source_provider", "source_url", "text_asr_normalized", args.label_field)},
                          "audio_sha256": hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()} for row in rows] for name, rows in splits.items()},
        "settings": {key: jsonable(value) for key, value in vars(args).items()}}
    (args.output_dir / "experiment_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)  # carries any extra tokens for serving
    model.config.use_cache = True

    # Test evaluation
    from gpu_evaluation import evaluate_rows, summarize_predictions
    result = {"model": model_id, "best_validation_macro_reciter_wer": trainer.state.best_metric, "seed": args.seed,
               "init_revision": manifest["init_revision"], "is_smoke_run": manifest["is_smoke_run"],
               "pretraining_overlap": manifest["tarteel_pretraining_data_overlap"],
              "label_field": args.label_field, "split_counts": counts,
               "split_protocol": protocol,
              "validation_reciter": args.validation_reciter, "test_reciter": args.test_reciter,
              "holdout_reciters": sorted(args.holdout_reciter),
              "split_reciters": {k: sorted({r["reciter_key"] for r in rows}) for k, rows in splits.items()},
              "augmentation": {**manifest["augmentation"], "max_seconds": 30}, "spec_augment": spec_augment,
              "per_reciter_wer": {}}
    for split in ("validation", "test", "holdout"):
        if not splits.get(split):
            continue
        scores, details = evaluate_rows(model, processor, splits[split], args.eval_batch_size,
                                        decode_profile="beam3", normalizer_version=NORMALIZER_VERSION)
        if tagged:
            # WER on tag-stripped words; tags scored per aligned word against the tagged target.
            for row, detail in zip(splits[split], details):
                detail["prediction"] = normalize_quran_for_asr(strip_tags(detail["raw_prediction"]))
                detail["target"] = row.get(args.label_field)
            rescored = summarize_predictions(details, "beam3")
            rescored["qaloon_lexical_fidelity"]["constraint_can_force_qaloon_spelling"] = False
            scores.update(rescored)
            labeled = [d for d in details if d["target"]]
            scores["tags"] = {"overall": tag_scores([(d["target"], d["raw_prediction"]) for d in labeled], normalize_word),
                              "per_reciter": {reader: tag_scores([(d["target"], d["raw_prediction"]) for d in labeled if d["reciter"] == reader], normalize_word)
                                              for reader in sorted({d["reciter"] for d in labeled})}}
        result[split] = scores
        result["per_reciter_wer"][split] = {reader: scores[reader]["wer"] for reader in sorted({d["reciter"] for d in details})}
        print(f"\n{split.upper()} RESULTS:", json.dumps(scores, ensure_ascii=False, indent=2))
        print(f"{split} per-reciter WER:", {k: round(v, 4) for k, v in result["per_reciter_wer"][split].items()})
        if tagged:
            print(f"{split} tag P/R/F1:", {k: scores["tags"]["overall"][k] for k in ("precision", "recall", "f1")})
        with (args.output_dir / f"{split}_predictions.jsonl").open("w", encoding="utf-8") as handle:
            for row in details:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    (args.output_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
