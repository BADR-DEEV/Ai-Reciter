"""Evaluate ahkam mode exactly as the app runs it: words from the plain model,
tajweed tokens from the tajweed model aligned onto those words.

    PYTHONPATH=src/deployment/mac_shim python -m src.tajweed.evaluate \\
        --plain runs/rattil_qaloon_v4 --tajweed runs/rattil_qaloon_tajweed_v2 \\
        --reader-root data/tajweed --reciters waleed ttsplain

Per reader it reports word error, token precision/recall on recognised words,
and for plain readings (rows with "tajweed": false) the false-alarm rate:
tokens the model invents per word where nothing should be heard. Plain takes
are scored only on ayahs outside the training split.
"""
import argparse
import json
from pathlib import Path

import jiwer
import soundfile as sf

from src.dataset_collection.qaloon_audio2text import normalize_quran_for_asr
from src.streaming.matcher import words as matcher_words
from src.streaming.tajweed_tags import tagged_words as split_tagged, transfer_tags
from src.training.qaloon_data import split_name

from .feedback import compare


def load(model_dir, device="cpu"):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    processor = WhisperProcessor.from_pretrained(model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir).to(device).eval()
    return processor, model


def decode(bundle, audio, keep_tags):
    import torch
    processor, model = bundle
    features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(model.device)
    with torch.no_grad():
        ids = model.generate(features, language="arabic", task="transcribe", max_new_tokens=300)
    if not keep_tags:
        return processor.batch_decode(ids, skip_special_tokens=True)
    special = set(processor.tokenizer.all_special_ids) - set(processor.tokenizer.get_added_vocab().values())
    return [processor.tokenizer.decode([t for t in row.tolist() if t not in special]).strip() for row in ids]


def rows_for(root, reciter, limit):
    meta = Path(root) / f"dataset_qaloon_{reciter}" / "metadata.jsonl"
    rows = [json.loads(line) for line in meta.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r.get("tajweed") is not False or split_name(r["surah"], r["ayah"], 42) != "train"]
    return [{**r, "path": meta.parent / r["relative_audio_path"]} for r in rows[:limit] if r["ayah"] and r["surah"] != 1]


def evaluate(rows, plain, tajweed, batch=8):
    refs, hyps = [], []
    expected = applied = extra = recognised = 0
    for start in range(0, len(rows), batch):
        part = rows[start:start + batch]
        audio = [sf.read(r["path"], dtype="float32")[0] for r in part]
        for row, words_text, tagged in zip(part, decode(plain, audio, False), decode(tajweed, audio, True)):
            merged = transfer_tags(words_text, tagged, matcher_words)
            heard, tags = split_tagged(merged, matcher_words)
            refs.append(row["text_asr_normalized"])
            hyps.append(normalize_quran_for_asr(words_text) or "-")
            plain_reading = row.get("tajweed") is False
            if plain_reading:
                matched = [w for w in heard if w in row["text_asr_normalized"].split()]
                recognised += len(matched)
                extra += sum(len(t) for w, t in zip(heard, tags) if w in matched)
                continue
            result = compare(row["text_raw_uthmani"], heard, tags, row["surah"], row["ayah"]) or {}
            expected += result.get("expected", 0)
            applied += result.get("applied", 0)
            extra += sum(len(w["extra"]) for w in result.get("words", []))
    out = {"clips": len(rows), "wer": round(jiwer.wer(refs, hyps), 4)}
    if expected:
        out.update(token_recall=round(applied / expected, 3), token_precision=round(applied / max(1, applied + extra), 3))
    if recognised:
        out["false_alarms_per_word"] = round(extra / recognised, 3)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plain", type=Path, required=True)
    parser.add_argument("--tajweed", type=Path, required=True)
    parser.add_argument("--reader-root", type=Path, required=True)
    parser.add_argument("--reciters", nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--device", default="cpu", help="cpu or mps (greedy decoding works on mps; batched beams hang there)")
    args = parser.parse_args()
    plain, tajweed = load(args.plain, args.device), load(args.tajweed, args.device)
    report = {reciter: evaluate(rows_for(args.reader_root, reciter, args.limit), plain, tajweed) for reciter in args.reciters}
    print(json.dumps({"plain": str(args.plain), "tajweed": str(args.tajweed), **report}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
