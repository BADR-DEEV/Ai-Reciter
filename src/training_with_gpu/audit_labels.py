"""Flag clips whose audio and label disagree at the clip edges, before training on them.

Several strong independent checkpoints transcribe every clip. A clip is flagged when at least
--min-votes of them see extra or missing words at the start or end of the clip, the signature
of neighbouring-ayah audio bleeding in or a cut-off ayah. Spelling conventions are ignored
(skeleton forms) and substitutions are never counted, so Qaloon-specific readings that a
Hafs-trained checkpoint writes differently cannot cause a flag. Flagged clips should be
excluded from training and reviewed by a person; they are not deleted or relabelled.

    python src/training_with_gpu/audit_labels.py --data-root D:/data/all_reciters \
        --include-reciter huthaify husary dokali --output flagged.json \
        --model tarteel=D:/models/whisper-base-ar-quran --model deepdml=... --model turbo=...
"""
import argparse
import json
import re
import sys
from pathlib import Path

import jiwer
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_models import load_model, transcribe  # noqa: E402
from qaloon_data import load_splits  # noqa: E402
from qaloon_audio2text import normalize_quran_for_asr  # noqa: E402


def skeleton(text):
    words = []
    for word in normalize_quran_for_asr(text).split():
        word = re.sub(r"[ءاأإآ]", "", word).replace("ى", "ي").replace("ة", "ه")
        words.append(re.sub(r"(.)\1+", r"\1", word) or "ا")
    return " ".join(words)


def edge_mismatch(reference, hypothesis):
    """Insertions or deletions touching the first or last word of the clip."""
    if not hypothesis.strip():
        return ["empty transcript"]
    chunks = jiwer.process_words(reference, hypothesis).alignments[0]
    found = []
    for edge, chunk in (("start", chunks[0]), ("end", chunks[-1])):
        if chunk.type in ("insert", "delete"):
            found.append(f"{chunk.type} at {edge}")
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--include-reciter", nargs="+", required=True)
    p.add_argument("--model", action="append", required=True, help="NAME=PATH")
    p.add_argument("--min-votes", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    splits, _ = load_splits(args.data_root, include=args.include_reciter)
    rows = [dict(row, split=name) for name, part in splits.items() for row in part]
    print(f"Auditing {len(rows)} clips from {args.include_reciter}", flush=True)
    votes = {i: [] for i in range(len(rows))}
    transcripts = {i: {} for i in range(len(rows))}
    for spec in args.model:
        name, _, path = spec.partition("=")
        processor, model = load_model(path, torch.float16)
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            outputs = transcribe(model, processor, [sf.read(r["path"], dtype="float32")[0] for r in batch], False, torch.float16)
            for offset, (row, text) in enumerate(zip(batch, outputs)):
                index = start + offset
                transcripts[index][name] = normalize_quran_for_asr(text)
                issues = edge_mismatch(skeleton(row["text_asr_normalized"]), skeleton(text))
                if issues:
                    votes[index].append(f"{name}: {', '.join(issues)}")
        del model
        torch.cuda.empty_cache()
        print(f"  {name} done", flush=True)

    flagged = [{"key": f"{r['reciter_key']}:{r['surah']}:{r['ayah']}", "split": r["split"],
                "label": r["text_asr_normalized"], "votes": votes[i], "heard": transcripts[i]}
               for i, r in enumerate(rows) if len(votes[i]) >= args.min_votes]
    by_reciter = {}
    for item in flagged:
        reciter = item["key"].split(":")[0]
        by_reciter[reciter] = by_reciter.get(reciter, 0) + 1
    totals = {}
    for r in rows:
        totals[r["reciter_key"]] = totals.get(r["reciter_key"], 0) + 1
    summary = {reciter: f"{by_reciter.get(reciter, 0)}/{count}" for reciter, count in sorted(totals.items())}
    args.output.write_text(json.dumps({"min_votes": args.min_votes, "models": args.model, "flagged_per_reciter": summary,
                                       "flagged": flagged}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Flagged per reciter:", summary)


if __name__ == "__main__":
    main()
