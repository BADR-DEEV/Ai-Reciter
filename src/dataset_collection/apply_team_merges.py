"""Convert a KFGQPC-numbered clip folder to the team's ayah numbering (AYAH_MERGES).

The team datasets and the studio merge four KFGQPC ayah pairs into one ayah and renumber the
rest of the surah (see qaloon_audio2text.AYAH_MERGES). A pair is merged only when both of its
clips passed verification; the two clips share one cut point, so joining them restores the
continuous recording. If either half is missing, both are dropped. The KFGQPC-numbered clips
are kept in audio_kfgqpc/ with metadata.kfgqpc.jsonl.

    python src/dataset_collection/apply_team_merges.py D:/data/new_qaloon/dataset_qaloon_daawob
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qaloon_audio2text import AYAH_MERGES, normalize_quran_for_asr, normalize_with_harakat  # noqa: E402


def convert(folder):
    folder = Path(folder)
    if (folder / "metadata.kfgqpc.jsonl").exists():
        raise SystemExit(f"{folder} was already converted")
    rows = [json.loads(line) for line in (folder / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    (folder / "audio").rename(folder / "audio_kfgqpc")
    (folder / "metadata.jsonl").rename(folder / "metadata.kfgqpc.jsonl")
    (folder / "audio").mkdir()
    by_key = {(r["surah"], r["ayah"]): r for r in rows}
    out, dropped = [], []
    for r in rows:
        surah, ayah = r["surah"], r["ayah"]
        pairs = AYAH_MERGES.get(surah, [])
        if any(ayah == second for _, second in pairs):
            continue  # Emitted together with its first half.
        new_ayah = ayah - sum(1 for _, second in pairs if second < ayah)
        source = folder / "audio_kfgqpc" / r["audio_filename"]
        row = dict(r, ayah=new_ayah, audio_filename=f"{surah:03d}{new_ayah:03d}.wav",
                   relative_audio_path=f"audio/{surah:03d}{new_ayah:03d}.wav")
        second = next((b for a, b in pairs if a == ayah), None)
        if second is not None:
            other = by_key.get((surah, second))
            if other is None:
                dropped.append(f"{surah}:{ayah}+{second}")
                continue
            first_audio, rate = sf.read(source, dtype="float32")
            second_audio, _ = sf.read(folder / "audio_kfgqpc" / other["audio_filename"], dtype="float32")
            raw = f"{r['text_raw_uthmani']} {other['text_raw_uthmani']}"
            label = normalize_quran_for_asr(raw)
            audio = np.concatenate([first_audio, second_audio])
            row.update(text=label, text_asr_normalized=label, text_raw_uthmani=raw,
                       normalized_with_harakat=normalize_with_harakat(raw), source_ayahs=[ayah, second],
                       end_time=other["end_time"], duration_seconds=round(len(audio) / rate, 3))
            sf.write(folder / row["relative_audio_path"], audio, rate, subtype="PCM_16")
        else:
            shutil.copy2(source, folder / row["relative_audio_path"])
        out.append(row)
    (folder / "metadata.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print(f"{folder.name}: {len(rows)} KFGQPC clips -> {len(out)} team-numbered clips; dropped incomplete pairs: {dropped}")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        convert(path)
