import os
import json
import shutil
import argparse
import requests

from qaloon_audio2text import load_quran, AYAH_MERGES, normalize_quran_for_asr


# =========================================================
# SETTINGS
# =========================================================

END_SURAH = 114

RECITERS = {
    "husary": (270, "Mahmoud Al-Husary (Qaloon)",
               "https://server13.mp3quran.net/husr/Rewayat-Qalon-A-n-Nafi", "dataset_qaloon_Husary"),
    "dokali": (208, "Al-Dokali Muhammad Al-Alim (Qaloon)",
               "https://server7.mp3quran.net/dokali", "dataset_qaloon_dokali"),
    # The supplied hthfi/Rewayat-Sho-bah-A-n-Asim URL is Shu'bah, NOT Qaloon.
    "huthaify": (75, "Ali Al-Huthaifi (Qaloon)",
                 "https://server9.mp3quran.net/huthifi_qalon", "dataset_qaloon_hutafi"),
}

TIMING_URL = "https://www.mp3quran.net/api/v3/ayat_timing"


# =========================================================
# OUTPUT
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================================================
# LOAD QALOON TEXT
# =========================================================

# Text is loaded when the script runs, so importing this module is safe.


# =========================================================
# FIX MP3QURAN TIMING TO MATCH FRIEND'S QALOON MERGES
# =========================================================

def fix_timings(surah, timings):

    merges = {
        start: end
        for start, end
        in AYAH_MERGES.get(surah, [])
    }

    if not merges:
        return timings

    timings = sorted(
        timings,
        key=lambda x: int(x["ayah"])
    )

    fixed = []

    i = 0
    new_ayah = 1

    while i < len(timings):

        item = timings[i].copy()

        old_ayah = int(
            item["ayah"]
        )

        # Example:
        # Surah 80: timing 24 + 25
        # becomes corrected ayah 24
        if old_ayah in merges:

            merge_end = merges[
                old_ayah
            ]

            j = i

            while (
                j < len(timings)
                and
                int(timings[j]["ayah"])
                <= merge_end
            ):
                j += 1

            item["ayah"] = new_ayah

            item["start_time"] = int(
                timings[i]["start_time"]
            )

            item["end_time"] = int(
                timings[j - 1]["end_time"]
            )

            fixed.append(item)

            i = j

        else:

            item["ayah"] = new_ayah

            fixed.append(item)

            i += 1

        new_ayah += 1

    return fixed


# =========================================================
# DOWNLOAD FULL SURAH FROM MATCHING MP3QURAN MOSHAF
# =========================================================

def download_surah(surah, full_surah_dir, audio_base_url):

    filename = f"{surah:03d}.mp3"

    path = os.path.join(
        full_surah_dir,
        filename
    )

    # Already downloaded
    if (
        os.path.exists(path)
        and
        os.path.getsize(path) > 1000
    ):

        print(
            f"Surah {surah}: audio already downloaded"
        )

        return path

    url = (
        f"{audio_base_url}/"
        f"{filename}"
    )

    print(
        f"Surah {surah}: downloading audio..."
    )

    r = requests.get(
        url,
        timeout=120
    )

    r.raise_for_status()

    with open(
        path,
        "wb"
    ) as f:

        f.write(
            r.content
        )

    return path


# =========================================================
# GET TIMING
# =========================================================

def get_timing(surah, read_id):

    print(
        f"Surah {surah}: getting timing..."
    )

    r = requests.get(
        TIMING_URL,
        params={
            "surah": surah,
            "read": read_id
        },
        timeout=60
    )

    r.raise_for_status()

    timings = r.json()

    if not isinstance(
        timings,
        list
    ):
        raise ValueError(
            f"Invalid timing response for Surah {surah}"
        )

    return fix_timings(
        surah,
        timings
    )


# =========================================================
# PROCESS
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Build Qaloon Whisper WAV clips from MP3Quran timings")
    parser.add_argument("--reciter", choices=RECITERS, default="husary")
    parser.add_argument("--include-bismillah", action="store_true",
                        help="Include the pre-ayah Fatiha introduction as a separate ayah 0 clip")
    args = parser.parse_args()
    read_id, reciter, audio_base_url, output_name = RECITERS[args.reciter]
    out_dir = os.path.join(BASE_DIR, output_name)
    audio_dir = os.path.join(out_dir, "audio")
    full_surah_dir = os.path.join(out_dir, f"full_surahs_read_{read_id}")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(full_surah_dir, exist_ok=True)
    # pydub needs both executables to decode the cached full-surah MP3s.
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        try:
            import static_ffmpeg
            static_ffmpeg.add_paths()
        except ImportError:
            pass
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        raise RuntimeError(
            "ffmpeg and ffprobe are required. Install them on PATH or run "
            "'python -m pip install static-ffmpeg' to use bundled binaries."
        )
    from pydub import AudioSegment
    AudioSegment.converter = shutil.which("ffmpeg")

    quran = load_quran()
    rows = []
    missing = []
    surahs = [1, *range(78, END_SURAH + 1)]

    for surah in surahs:
        print(f"Processing Surah {surah}")
        try:
            full_audio_path = download_surah(surah, full_surah_dir, audio_base_url)
            timings = get_timing(surah, read_id)
            audio = AudioSegment.from_mp3(full_audio_path)
            audio_duration_ms = len(audio)
            seen = set()

            if surah == 1 and args.include_bismillah and timings:
                # MP3Quran's Fatiha #1 starts after the introductory basmalah.
                intro_end = min(audio_duration_ms, int(timings[0]["start_time"]))
                if intro_end > 0:
                    filename = "001000.wav"
                    audio[:intro_end].set_channels(1).set_frame_rate(16000).set_sample_width(2).export(
                        os.path.join(audio_dir, filename), format="wav"
                    )
                    raw = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
                    rows.append({"surah": 1, "ayah": 0, "audio_filename": filename,
                                 "relative_audio_path": f"audio/{filename}", "text": "بسم الله الرحمن الرحيم",
                                 "text_asr_normalized": normalize_quran_for_asr(raw),
                                 "text_raw_uthmani": raw, "source_ayahs": [], "reciter": reciter,
                                 "start_time": 0, "end_time": intro_end})

            for item in timings:
                ayah = int(item["ayah"])
                data = quran.get((surah, ayah))
                if data is None:
                    missing.append({"surah": surah, "ayah": ayah, "reason": "missing_text"})
                    continue

                source_ayahs = data.get("source_ayahs", [ayah])
                seen.update(source_ayahs)
                start_time = max(0, int(item["start_time"]))
                end_time = min(audio_duration_ms, int(item["end_time"]))
                if end_time <= start_time:
                    missing.append({"surah": surah, "ayah": ayah, "reason": "invalid_timing"})
                    continue

                filename = f"{surah:03d}{ayah:03d}.wav"
                clip = audio[start_time:end_time]
                clip.set_channels(1).set_frame_rate(16000).set_sample_width(2).export(
                    os.path.join(audio_dir, filename), format="wav"
                )
                rows.append({
                    "surah": surah,
                    "ayah": ayah,
                    "audio_filename": filename,
                    "relative_audio_path": f"audio/{filename}",
                    "text": data["text"],
                    "text_asr_normalized": normalize_quran_for_asr(data["raw"]),
                    "text_raw_uthmani": data["raw"],
                    "source_ayahs": source_ayahs,
                    "reciter": reciter,
                    "start_time": start_time,
                    "end_time": end_time,
                })

            for expected in sorted(a for s, a in quran if s == surah):
                if expected not in seen:
                    missing.append({"surah": surah, "ayah": expected, "reason": "missing_timing"})
            print(f"Surah {surah} finished.")
        except Exception as exc:
            print(f"ERROR Surah {surah}: {exc}")
            missing.append({"surah": surah, "reason": str(exc)})

    if not rows:
        print(f"No clips created; leaving {out_dir} metadata untouched. Problems: {missing}")
        return

    with open(os.path.join(out_dir, "metadata.jsonl"), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "missing_audio.json"), "w", encoding="utf-8") as f:
        json.dump(missing, f, ensure_ascii=False, indent=2)
    print(f"Created: {len(rows)} samples | Problems: {len(missing)} | Dataset: {out_dir}")


if __name__ == "__main__":
    main()
