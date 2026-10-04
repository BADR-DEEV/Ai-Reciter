"""Independent core-reader PCM facts; energy flags do NOT prove leaked words.

No clipping/trimming, ASR, automatic boundary approval or wholesale reader exclusion.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.training.qaloon_data import DATA_ROOT, RECITER_DIRS


def acoustic_facts(audio, sr):
    if sr != 16000 or audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Expected nonempty finite 16kHz mono audio")
    rms = float(np.sqrt(np.mean(audio ** 2)))
    edge = min(len(audio), 320)  # Fixed 20ms observation, never a proposed crop.
    start_rms = float(np.sqrt(np.mean(audio[:edge] ** 2)))
    end_rms = float(np.sqrt(np.mean(audio[-edge:] ** 2)))
    flags = []
    if start_rms > .005 and start_rms > rms * .5:
        flags.append("energetic_start_review_only")
    if end_rms > .005 and end_rms > rms * .5:
        flags.append("energetic_end_review_only")
    if float(np.mean(np.abs(audio) >= .999)) > .001:
        flags.append("possible_clipping")
    if rms < 1e-5:
        flags.append("near_silent")
    return {"duration_seconds": len(audio) / sr, "rms": rms, "start_20ms_rms": start_rms, "end_20ms_rms": end_rms,
            "peak": float(np.max(np.abs(audio))), "dc_offset": float(np.mean(audio)),
            "saturated_sample_fraction": float(np.mean(np.abs(audio) >= .999)), "flags": flags}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Preserve previous audits")
    report = {"scope": "core dataset engineering, not model-test tuning", "readers": {},
        "boundaries_and_content_not_certified": True, "automatic_quarantine_from_energy": False,
        "thresholds": {"edge_ms": 20, "absolute_rms": .005, "relative_rms": .5}}
    for reader in ("dokali", "huthaify", "husary"):
        folder = DATA_ROOT / RECITER_DIRS[reader]
        results = []
        for line in (folder / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            path = folder / row["relative_audio_path"]
            audio, sr = sf.read(path, dtype="float32")
            results.append({"surah": row["surah"], "ayah": row["ayah"], **acoustic_facts(audio, sr)})
        flags = {name: sum(name in r["flags"] for r in results) for name in ("energetic_start_review_only", "energetic_end_review_only", "possible_clipping", "near_silent")}
        report["readers"][reader] = {"clips": len(results), "flag_counts": flags, "details": results}
        print(reader, len(results), flags)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
