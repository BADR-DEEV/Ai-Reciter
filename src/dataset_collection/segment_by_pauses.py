"""Cut full-surah murattal recordings into verified single-ayah clips by forced alignment.

The text of every surah is known (QaloonData_v10), so each word is located in the audio with
CTC forced alignment (torchaudio's MMS aligner, ~20 ms precision). Each ayah is cut at the quietest
moment inside the gap between its last word and the next ayah's first word, so no word is cut.
The opening isti'adha/basmalah is aligned only if a transcript of the opening shows it was recited
(for surahs other than al-Fatiha, hearing الرحمن الرحيم in the opening also counts as a basmalah).

Every clip is then re-transcribed by independent checkpoints. It is kept only if none of them sees
extra or missing words at its edges, all of them broadly agree with its text, the best one matches
closely, and its length is plausible for its word count; everything else goes to rejected.jsonl.

    python src/dataset_collection/segment_by_pauses.py --input-dir D:/data/new_reciters/daawob \
        --output-dir D:/data/new_qaloon/dataset_qaloon_daawob --reciter "Tareq Daawob (Qaloon)" \
        --reciter-key daawob --source-url https://cdn.mp3quran.net/audio/tareq-daawob/r1/ \
        --opening-model D:/runs/rattil_qaloon_v1 --verify-model D:/models/deepdml --verify-model D:/models/tarteel
"""
import argparse
import json
import re
import sys
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as F
import uroman

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "dataset_collection"))
sys.path.insert(0, str(ROOT / "src" / "training_with_gpu"))
from qaloon_audio2text import normalize_quran_for_asr, normalize_with_harakat  # noqa: E402
from benchmark_models import load_model, transcribe  # noqa: E402

SR = 16000
FRAME = 0.02
CHUNK, CONTEXT = 20.0, 1.0
ISTIADHA, BASMALAH = "اعوذ بالله من الشيطان الرجيم", "بسم الله الرحمن الرحيم"


def skeleton(word):
    word = re.sub(r"[ءاأإآ]", "", word).replace("ى", "ي").replace("ة", "ه")
    return re.sub(r"(.)\1+", r"\1", word) or "ا"


def load_mono(path):
    audio, rate = sf.read(path, dtype="float32")
    audio = audio.mean(axis=1) if audio.ndim > 1 else audio
    return F.resample(torch.from_numpy(audio), rate, SR).numpy()


class Aligner:
    def __init__(self):
        bundle = torchaudio.pipelines.MMS_FA
        self.model = bundle.get_model(with_star=False).cuda().eval()
        self.tokenizer, self.aligner = bundle.get_tokenizer(), bundle.get_aligner()
        self.romanizer = uroman.Uroman()

    def romanize(self, word):
        text = self.romanizer.romanize_string(word).lower()
        return "".join(c for c in text if c in self.tokenizer.dictionary) or "a"

    def emission(self, audio):
        """CTC emissions over long audio, computed in overlapping chunks to bound GPU memory."""
        parts, step = [], int(CHUNK * SR)
        for start in range(0, len(audio), step):
            lo, hi = max(0, start - int(CONTEXT * SR)), min(len(audio), start + step + int(CONTEXT * SR))
            with torch.inference_mode():
                out, _ = self.model(torch.from_numpy(audio[lo:hi])[None].cuda())
            rate = out.shape[1] / ((hi - lo) / SR)
            keep_from = round((start - lo) / SR * rate)
            keep_to = keep_from + round(min(step, len(audio) - start) / SR * rate)
            parts.append(out[0, keep_from:keep_to])
        return torch.cat(parts)

    def word_spans(self, audio, words):
        """(start, end, score) in seconds for each word, in order."""
        emission = self.emission(audio)
        spans = self.aligner(emission, self.tokenizer([self.romanize(w) for w in words]))
        ratio = len(audio) / SR / emission.shape[0]
        return [(s[0].start * ratio, s[-1].end * ratio, sum(t.score for t in s) / len(s)) for s in spans]


def quietest(audio, lo_s, hi_s):
    """Time of the lowest smoothed energy between two moments (widened to at least 0.2 s)."""
    if hi_s - lo_s < 0.2:
        mid = (lo_s + hi_s) / 2
        lo_s, hi_s = mid - 0.1, mid + 0.1
    lo, hi = max(0, int(lo_s * SR)), min(len(audio), int(hi_s * SR))
    hop = int(SR * FRAME)
    energy = np.convolve(audio[lo:hi] ** 2, np.ones(hop * 3) / (hop * 3), mode="same")
    return (lo + int(np.argmin(energy[::hop]) * hop)) / SR


def edge_clean(reference, hypothesis):
    if not hypothesis.strip():
        return False
    chunks = jiwer.process_words(reference, hypothesis).alignments[0]
    return chunks[0].type not in ("insert", "delete") and chunks[-1].type not in ("insert", "delete")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, required=True, help="Folder of SSS.mp3 full-surah files")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--reciter", required=True)
    p.add_argument("--reciter-key", required=True)
    p.add_argument("--source-url", required=True)
    p.add_argument("--opening-model", required=True, help="Checkpoint used to detect a recited isti'adha/basmalah")
    p.add_argument("--verify-model", action="append", required=True)
    p.add_argument("--max-wer", type=float, default=0.2, help="Best verifier's spelling-insensitive WER")
    p.add_argument("--max-worst-wer", type=float, default=0.4, help="Every verifier's spelling-insensitive WER")
    args = p.parse_args()

    quran = json.loads((ROOT / "src/dataset_collection/QaloonData_v10(1).json").read_text(encoding="utf-8"))
    texts = {}
    for row in quran:
        texts.setdefault(int(row["sura_no"]), []).append((int(row["aya_no"]), row["aya_text"]))
    (args.output_dir / "audio").mkdir(parents=True, exist_ok=True)

    aligner = Aligner()
    opening = load_model(args.opening_model, torch.float16)
    verifiers = [load_model(path, torch.float16) for path in args.verify_model]
    kept, rejected, skipped = [], [], []
    for path in sorted(args.input_dir.glob("*.mp3")):
        surah = int(path.stem)
        audio = load_mono(path)
        heard = {skeleton(w) for w in normalize_quran_for_asr(
            transcribe(opening[1], opening[0], [audio[:15 * SR]], False, torch.float16)[0]).split()}
        basmalah = skeleton("بسم") in heard or (surah != 1 and {skeleton("الرحمن"), skeleton("الرحيم")} <= heard)
        preamble = ([ISTIADHA] if skeleton("اعوذ") in heard else []) + ([BASMALAH] if basmalah else [])
        ref = [(0, w) for t in preamble for w in normalize_quran_for_asr(t).split()]
        ref += [(ayah, w) for ayah, raw in texts[surah] for w in normalize_quran_for_asr(raw).split()]
        spans = aligner.word_spans(audio, [w for _, w in ref])
        bounds = {}
        for (ayah, _), (start, end, score) in zip(ref, spans):
            first, last, scores = bounds.get(ayah, (start, end, []))
            bounds[ayah] = (min(first, start), max(last, end), scores + [score])
        order = sorted(bounds)
        for ayah, raw in texts[surah]:
            k = order.index(ayah)
            first, last, scores = bounds[ayah]
            prev_end = bounds[order[k - 1]][1] if k > 0 else None
            next_start = bounds[order[k + 1]][0] if k + 1 < len(order) else None
            start = quietest(audio, prev_end, first) if prev_end is not None else max(0.0, first - 0.3)
            end = quietest(audio, last, next_start) if next_start is not None else min(len(audio) / SR, last + 0.5)
            clip = audio[int(start * SR):int(end * SR)]
            label = normalize_quran_for_asr(raw)
            if len(clip) > 30 * SR:
                skipped.append({"surah": surah, "ayah": ayah, "reason": "longer than 30 s"})
                continue
            transcripts = [normalize_quran_for_asr(transcribe(m, pr, [clip], False, torch.float16)[0]) for pr, m in verifiers]
            ref_sk = " ".join(skeleton(w) for w in label.split())
            hyp_sk = [" ".join(skeleton(w) for w in h.split()) for h in transcripts]
            clean = all(edge_clean(ref_sk, h) for h in hyp_sk)
            wers = [jiwer.wer(ref_sk, h) if h else 1.0 for h in hyp_sk]
            pace = len(clip) / SR / len(label.split())
            row = {"surah": surah, "ayah": ayah, "audio_filename": f"{surah:03d}{ayah:03d}.wav",
                   "relative_audio_path": f"audio/{surah:03d}{ayah:03d}.wav", "text": label,
                   "text_asr_normalized": label, "text_raw_uthmani": raw, "normalized_with_harakat": normalize_with_harakat(raw),
                   "source_ayahs": [ayah], "reciter": args.reciter, "reciter_key": args.reciter_key,
                   "start_time": round(start * 1000), "end_time": round(end * 1000),
                   "duration_seconds": round(len(clip) / SR, 3), "source_url": args.source_url,
                   "source_license": "not_established", "preamble_detected": [t.split()[0] for t in preamble],
                   "alignment_min_score": round(min(scores), 2), "verifier_transcripts": transcripts,
                   "verifier_best_wer": round(min(wers), 3), "verifier_worst_wer": round(max(wers), 3),
                   "seconds_per_word": round(pace, 2),
                   "alignment_status": "ctc_forced_aligned_auto_verified_not_human_reviewed"}
            if clean and min(wers) <= args.max_wer and max(wers) <= args.max_worst_wer and 0.25 <= pace <= 3.0:
                sf.write(args.output_dir / row["relative_audio_path"], clip, SR, subtype="PCM_16")
                kept.append(row)
            else:
                rejected.append(row)
        print(f"{surah:03d}: kept {sum(r['surah'] == surah for r in kept)}, rejected {sum(r['surah'] == surah for r in rejected)}, "
              f"skipped {sum(r['surah'] == surah for r in skipped)} of {len(texts[surah])} (preamble: {[t.split()[0] for t in preamble]})", flush=True)

    for name, rows in (("metadata.jsonl", kept), ("rejected.jsonl", rejected), ("skipped.jsonl", skipped)):
        (args.output_dir / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    total = sum(len(texts[int(p.stem)]) for p in args.input_dir.glob("*.mp3"))
    print(f"{args.reciter_key}: kept {len(kept)}, rejected {len(rejected)}, skipped {len(skipped)} of {total} ayahs")


if __name__ == "__main__":
    main()
