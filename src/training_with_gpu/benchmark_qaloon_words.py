"""Does a checkpoint write what a Qaloon reciter actually recites where Qaloon differs from Hafs?

Differences were extracted by aligning the KFGQPC Qaloon v10 and Hafs v18 texts word by word
over al-Fatiha and surahs 78-114 (22 segments). The classification below is a draft that must
be confirmed by a qualified Qaloon teacher. Only differences in the recited *words* can be
judged from plain-text ASR; tashil (79:10, 79:27, 96:9, 96:11, 96:13, 107:1) is a pronunciation
quality and spelling-only differences are not reading differences, so neither is scored here.

A clip is scored only if the rest of its ayah is recognised (>= 60% of the other words);
otherwise it is reported as unusable rather than guessed.

    python src/training_with_gpu/benchmark_qaloon_words.py --data-root D:/data/all_reciters \
        --output out.json --model ours=runs/gpu_base_full --model tarteel=D:/models/whisper-base-ar-quran
"""
import argparse
import json
import re
import sys
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_models import load_model, transcribe  # noqa: E402
from qaloon_data import RECITER_DIRS  # noqa: E402
from qaloon_audio2text import normalize_quran_for_asr  # noqa: E402

# (surah, Qaloon ayah): Qaloon ayah text (ASR-normalized), Qaloon word, Hafs spellings of the Hafs reading.
POSITIONS = {
    (1, 3): ("ملك يوم الدين", "ملك", {"مالك"}),
    (79, 11): ("اذا كنا عظاما نخرة", "اذا", {"اءذا", "ائذا", "ااذا"}),
    (80, 22): ("ثم اذا شا انشره", "شا", {"شاء"}),
    (83, 31): ("واذا انقلبوا الى اهلهم انقلبوا فاكهين", "فاكهين", {"فكهين"}),
    (89, 20): ("ولا تحضون على طعام المسكين", "تحضون", {"تحاضون"}),
    (90, 20): ("عليهم نار موصدة", "موصدة", {"مؤصدة"}),
    (91, 15): ("فلا يخاف عقباها", "فلا", {"ولا"}),
    (104, 8): ("انها عليهم موصدة", "موصدة", {"مؤصدة"}),
    (112, 4): ("ولم يكن له كفؤا احد", "كفؤا", {"كفوا"}),
}
CONTEXT_THRESHOLD = 0.6


def skeleton(word):
    word = re.sub(r"[ءاأإآ]", "", word).replace("ى", "ي").replace("ة", "ه")
    return re.sub(r"(.)\1+", r"\1", word)


def verify_positions(qaloon_json):
    """Fail loudly if the hard-coded Qaloon ayah texts drift from the KFGQPC source."""
    rows = {(r["sura_no"], r["aya_no"]): r["aya_text"] for r in json.loads(Path(qaloon_json).read_text(encoding="utf-8"))}
    for key, (text, word, _) in POSITIONS.items():
        source = normalize_quran_for_asr(re.sub(r"[\u0660-\u0669\d]+", "", rows[key]))
        if source != text or word not in source.split():
            raise ValueError(f"{key}: KFGQPC Qaloon text {source!r} does not match test definition {text!r}")


def judge(transcript, text, word, hafs):
    heard = normalize_quran_for_asr(transcript).split()
    context = [skeleton(w) for w in text.split() if w != word]
    found = {skeleton(w) for w in heard}
    if sum(w in found for w in context) < CONTEXT_THRESHOLD * len(context):
        return "unusable"
    if word in heard:
        return "qaloon"
    return "hafs" if hafs & set(heard) else "other"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--qaloon-json", type=Path, required=True, help="KFGQPC QaloonData_v10.json")
    p.add_argument("--model", action="append", required=True, help="NAME=PATH")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    verify_positions(args.qaloon_json)

    clips = []
    for reciter, folder in sorted(RECITER_DIRS.items()):
        for (surah, ayah) in POSITIONS:
            path = args.data_root / folder / "audio" / f"{surah:03d}{ayah:03d}.wav"
            if path.is_file():
                clips.append((reciter, surah, ayah, path))
    print(f"{len(clips)} clips: {len(POSITIONS)} Qaloon word positions x reciters", flush=True)

    report = []
    for spec in args.model:
        name, _, path = spec.partition("=")
        processor, model = load_model(path, torch.float16)
        outputs = transcribe(model, processor, [sf.read(c[3], dtype="float32")[0] for c in clips], False, torch.float16)
        del model
        torch.cuda.empty_cache()
        rows = []
        for (reciter, surah, ayah, _), out in zip(clips, outputs):
            text, word, hafs = POSITIONS[(surah, ayah)]
            rows.append({"reciter": reciter, "ayah": f"{surah}:{ayah}", "expected_qaloon": word,
                         "verdict": judge(out, text, word, hafs), "heard": normalize_quran_for_asr(out)})
        scored = [r for r in rows if r["verdict"] != "unusable"]
        counts = {v: sum(r["verdict"] == v for r in scored) for v in ("qaloon", "hafs", "other")}
        summary = {"model": name, "scored": len(scored), "unusable": len(rows) - len(scored), **counts,
                   "qaloon_rate": counts["qaloon"] / max(1, len(scored)), "clips": rows}
        report.append(summary)
        print(f"{name:20} Qaloon form {counts['qaloon']:2}/{len(scored)} ({summary['qaloon_rate']:.0%})  "
              f"Hafs form {counts['hafs']:2}  other {counts['other']:2}  unusable {summary['unusable']}", flush=True)
        for r in rows:
            if r["verdict"] in ("hafs", "other"):
                print(f"    {r['verdict']:6} {r['reciter']:9} {r['ayah']:6} expected {r['expected_qaloon']:8} heard: {r['heard']}")
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
