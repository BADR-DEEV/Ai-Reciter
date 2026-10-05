"""Measure whether a checkpoint "autocorrects" omissions from Quran memory.

The live omission detector relies on blind transcripts containing only what
was actually recited. A model with a strong Quran prior can instead insert
words that were never spoken. Two probes, built from real reference clips:

* Skipped ayah: ayah N, a short pause, then ayah N+2. Count transcripts that
  contain most of the distinctive words of the unspoken ayah N+1.
* Early stop: the first 60% of an ayah's audio. Count transcripts that end
  with the ayah's final word, which was never spoken.

    python src/training_with_gpu/benchmark_autocorrect.py --data-root D:/data/all_reciters \
        --model ours=runs/gpu_base_full --model tarteel=D:/models/whisper-base-ar-quran
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_models import load_model, transcribe  # noqa: E402
from qaloon_data import load_splits  # noqa: E402
from qaloon_audio2text import normalize_quran_for_asr  # noqa: E402

KEEP_FRACTION = 0.6
PAUSE_SECONDS = 0.6


def skeleton(word):
    """Spelling-insensitive word form, so orthography conventions are not counted."""
    word = re.sub(r"[ءاأإآ]", "", word).replace("ى", "ي").replace("ة", "ه")
    return re.sub(r"(.)\1+", r"\1", word)


def words(text):
    return [skeleton(w) for w in normalize_quran_for_asr(text).split()]


def build_probes(rows, seed, max_skip):
    by_key = {(r["reciter_key"], r["surah"], r["ayah"]): r for r in rows}
    skips = []
    for (reciter, surah, ayah), first in by_key.items():
        middle, last = by_key.get((reciter, surah, ayah + 1)), by_key.get((reciter, surah, ayah + 2))
        if middle and last and first["duration_seconds"] + last["duration_seconds"] + PAUSE_SECONDS <= 28:
            spoken = set(words(first["text_asr_normalized"])) | set(words(last["text_asr_normalized"]))
            distinctive = set(words(middle["text_asr_normalized"])) - spoken
            if len(distinctive) >= 2:
                skips.append((first, middle, last, distinctive))
    random.Random(seed).shuffle(skips)
    stops = [r for r in rows if len(words(r["text_asr_normalized"])) >= 4]
    return skips[:max_skip], stops


def run(name, path, skips, stops, dtype, batch_size):
    processor, model = load_model(path, dtype)
    pause = np.zeros(int(16000 * PAUSE_SECONDS), dtype="float32")

    def decode(signals):
        out = []
        for i in range(0, len(signals), batch_size):
            out += transcribe(model, processor, signals[i:i + batch_size], False, dtype)
        return [[skeleton(w) for w in normalize_quran_for_asr(t).split()] for t in out]

    audio = lambda row: sf.read(row["path"], dtype="float32")[0]
    skip_out = decode([np.concatenate([audio(a), pause, audio(c)]) for a, _, c, _ in skips])
    invented = [len(d & set(o)) / len(d) >= 0.5 for (_, _, _, d), o in zip(skips, skip_out)]

    stop_out = decode([audio(r)[: int(len(audio(r)) * KEEP_FRACTION)] for r in stops])
    finished = [bool(o) and o[-1] == words(r["text_asr_normalized"])[-1] for r, o in zip(stops, stop_out)]
    del model
    torch.cuda.empty_cache()
    return {"model": name, "skipped_ayah_probes": len(skips), "invented_skipped_ayah": sum(invented) / len(skips),
            "early_stop_probes": len(stops), "finished_unspoken_ayah": sum(finished) / len(stops),
            "examples": [{"skipped": " ".join(sorted(d)), "heard": " ".join(o)}
                         for (_, _, _, d), o, bad in zip(skips, skip_out, invented) if bad][:5]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--model", action="append", required=True, help="NAME=PATH")
    p.add_argument("--split", choices=("train", "validation", "test", "all"), default="all",
                   help="Clips to build probes from; 'all' maximizes consecutive-ayah triples")
    p.add_argument("--max-skip-probes", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    splits, _ = load_splits(args.data_root)
    rows = [r for part in splits.values() for r in part] if args.split == "all" else splits[args.split]
    stop_rows = splits["test"]  # Early-stop probes always use the fixed test clips.
    skips, stops = build_probes(rows, args.seed, args.max_skip_probes)
    stops = [r for r in stops if r in stop_rows]
    print(f"{len(skips)} skipped-ayah probes, {len(stops)} early-stop probes", flush=True)
    results = []
    for spec in args.model:
        name, _, path = spec.partition("=")
        result = run(name, path, skips, stops, torch.float16, args.batch_size)
        results.append(result)
        print(f"{name:22} invented skipped ayah {result['invented_skipped_ayah']:6.1%}   "
              f"finished unspoken ayah {result['finished_unspoken_ayah']:6.1%}", flush=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
