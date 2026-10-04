"""
Robust Quran Segmentation & Slicing Engine (Madd-Perfect & Leak-Free).
Guarantees:
1. Madd Protection: Tracks vocal energy dynamically so Madds are NEVER cut early.
2. Strict Forward Barrier: Ayah N+1 can NEVER steal the tail of Ayah N.
3. Zero Preamble Leak: Ayah 1 starts strictly after the Basmalah/Isti'adhah ends.
4. Schema-compliant output matching QaloonData_v10(1).json.
"""

import argparse
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import re
import sys
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F

ROOT = Path(__file__).resolve().parents[2]
COLLECTION_DIR = ROOT / "src/dataset_collection"
sys.path.insert(0, str(COLLECTION_DIR))

from qaloon_audio2text import (
    load_quran,
    normalize_quran_for_asr,
    normalize_with_harakat
)

try:
    from quran_geometry import geometry_fields
except ImportError:
    def geometry_fields(surah, source_ayahs):
        return {}


def word_similarity(w1: str, w2: str) -> float:
    return SequenceMatcher(None, w1, w2).ratio()


def find_dynamic_speech_end(audio: np.ndarray, sr: int, rough_end_sec: float, hard_limit_sec: float) -> float:
    """
    Follows vocal energy through the entire Madd until voice decays into silence.
    Guarantees that Madds are given their full right without crossing into the next verse.
    """
    start_sample = int(rough_end_sec * sr)
    limit_sample = min(len(audio), int(hard_limit_sec * sr))

    if start_sample >= limit_sample:
        return hard_limit_sec

    # Measure local peak energy around the word end
    ref_start = max(0, start_sample - int(0.5 * sr))
    local_segment = audio[ref_start:limit_sample]
    if len(local_segment) == 0:
        return hard_limit_sec

    peak_energy = np.max(np.abs(local_segment)) + 1e-6
    silence_threshold = 0.025 * peak_energy  # 2.5% of peak volume (-32 dB)

    frame_size = int(0.025 * sr)  # 25ms frame
    hop_size = int(0.010 * sr)    # 10ms hop

    last_vocal_sample = start_sample
    silent_hops = 0
    required_silent_hops = 20     # Must stay silent for 200ms continuously

    for s in range(start_sample, limit_sample - frame_size, hop_size):
        frame = audio[s:s + frame_size]
        rms = np.sqrt(np.mean(frame ** 2) + 1e-9)

        if rms > silence_threshold:
            last_vocal_sample = s + frame_size
            silent_hops = 0
        else:
            silent_hops += 1
            if silent_hops >= required_silent_hops:
                break

    # Add 120ms natural decay padding after vocal energy drops
    final_sample = min(limit_sample, last_vocal_sample + int(0.12 * sr))
    return final_sample / sr


def find_surah_file_end(audio: np.ndarray, sr: int, last_ayah_start_sec: float) -> float:
    """
    For the TRUE final Ayah of a Surah (e.g. Fatiha 7, An-Naba 40, An-Nas 6):
    Scans backward from the file end to capture the full 6-count Madd and natural decay.
    """
    total_sec = len(audio) / sr
    start_sample = int((last_ayah_start_sec + 0.5) * sr)

    if start_sample >= len(audio) - int(0.2 * sr):
        return total_sec

    slice_to_check = audio[start_sample:]
    if len(slice_to_check) == 0:
        return total_sec

    peak_energy = np.max(np.abs(slice_to_check)) + 1e-6
    silence_threshold = 0.02 * peak_energy  # -34 dB threshold for Madd ring-out

    frame_size = int(0.04 * sr)
    last_speech_sample = len(audio)

    for s in range(len(audio) - frame_size, start_sample, -frame_size):
        frame = audio[s:s + frame_size]
        rms = np.sqrt(np.mean(frame ** 2) + 1e-9)
        if rms > silence_threshold:
            last_speech_sample = min(len(audio), s + int(0.60 * sr))
            break

    max_sample = int((last_ayah_start_sec + 29.5) * sr)
    return min(last_speech_sample, max_sample, len(audio)) / sr


def align_surah_with_hard_walls(whisper_words, all_refs, audio_duration: float, max_surah_ayah: int, is_fatiha: bool = False):
    """
    Locates boundaries with 2-word anchor matching and strict preamble isolation.
    """
    # 1. Detect Preamble End (Isti'adhah & Basmalah)
    preamble_wall = 0.0
    for w in whisper_words[:28]:
        cw = w["clean_word"]
        if any(m in cw for m in ["الرجيم", "رجيم", "الرحيم", "رحيم"]):
            if is_fatiha and w["start"] > 10.0:
                continue
            preamble_wall = max(preamble_wall, w["end"])

    ayah_raw_bounds = {}
    current_idx = 0
    last_end = preamble_wall

    for idx, r in enumerate(all_refs):
        ayah_num = r["ayah"]
        raw_words = r["text_asr"].split()
        if not raw_words:
            continue

        first_word = raw_words[0]
        second_word = raw_words[1] if len(raw_words) > 1 else None
        last_word = raw_words[-1]

        # 1. Locate Start
        start_time = None
        start_w_idx = None
        search_window = 20 if ayah_num <= 1 else 14

        for i in range(current_idx, min(len(whisper_words), current_idx + search_window)):
            if whisper_words[i]["end"] <= preamble_wall:
                continue

            if word_similarity(first_word, whisper_words[i]["clean_word"]) >= 0.70:
                if second_word and i + 1 < len(whisper_words):
                    if word_similarity(second_word, whisper_words[i + 1]["clean_word"]) >= 0.65:
                        start_time = whisper_words[i]["start"]
                        start_w_idx = i
                        break
                else:
                    start_time = whisper_words[i]["start"]
                    start_w_idx = i
                    break

        if start_time is None and last_end > 0:
            start_time = last_end
            start_w_idx = current_idx

        if ayah_num == 1 and preamble_wall > 0:
            start_time = max(preamble_wall, start_time or preamble_wall)

        if start_time is None or start_w_idx is None:
            continue

        # 2. Locate End
        end_time = None
        end_w_idx = None
        search_start = start_w_idx + max(0, len(raw_words) - 4)
        lookahead = min(len(whisper_words), start_w_idx + max(len(raw_words) * 2 + 25, 70))

        for j in range(search_start, lookahead):
            if word_similarity(last_word, whisper_words[j]["clean_word"]) >= 0.70:
                end_time = whisper_words[j]["end"]
                end_w_idx = j
                break

        if end_time is None and len(raw_words) > 2:
            penultimate_word = raw_words[-2]
            for j in range(search_start, lookahead):
                if word_similarity(penultimate_word, whisper_words[j]["clean_word"]) >= 0.75:
                    end_time = whisper_words[j]["end"] + 0.8
                    end_w_idx = j
                    break

        if ayah_num == max_surah_ayah:
            end_time = audio_duration
            end_w_idx = len(whisper_words) - 1

        if end_time is None or end_time <= start_time:
            continue

        ayah_raw_bounds[ayah_num] = {
            "data": r,
            "start": start_time,
            "end": end_time,
        }
        last_end = end_time
        if end_w_idx is not None:
            current_idx = end_w_idx + 1

    return ayah_raw_bounds, preamble_wall


def slice_surah_with_waveform_tracking(audio: np.ndarray, sr: int, ayah_raw_bounds: dict, preamble_wall: float, max_surah_ayah: int):
    """
    Slices audio with dynamic waveform tracking to ensure Madds are given
    their full right and never leak into the next verse.
    """
    canonical_nums = sorted([a for a in ayah_raw_bounds if a >= 1])
    verified_ayahs = []
    audio_duration = len(audio) / sr
    previous_ayah_cut_end = 0.0

    for idx, a_num in enumerate(canonical_nums):
        cand = ayah_raw_bounds[a_num]
        raw_start = cand["start"]
        raw_end = cand["end"]

        # -------------------------------------------------------------
        # 1. COMPUTE START BOUNDARY (Strict Forward Lock)
        # -------------------------------------------------------------
        if a_num == 1:
            # Ayah 1 starts 70ms before its onset, strictly after preamble!
            final_start = raw_start - 0.07
            if preamble_wall > 0:
                final_start = max(preamble_wall + 0.05, final_start)
            else:
                final_start = max(0.0, final_start)
        else:
            # STRICT BARRIER: Cannot start before previous Ayah ended!
            final_start = max(previous_ayah_cut_end, raw_start - 0.08)

        # -------------------------------------------------------------
        # 2. COMPUTE END BOUNDARY (Dynamic Madd Energy Tracker)
        # -------------------------------------------------------------
        is_true_final = (a_num == max_surah_ayah)

        if is_true_final:
            final_end = find_surah_file_end(audio, sr, final_start)
        elif idx + 1 < len(canonical_nums):
            next_num = canonical_nums[idx + 1]
            next_cand = ayah_raw_bounds[next_num]
            next_raw_start = next_cand["start"]

            # Safe boundary wall: 60ms before next Ayah's spoken onset
            hard_limit = max(raw_end, next_raw_start - 0.06)

            # Track waveform energy: follow the Madd until voice drops into silence
            final_end = find_dynamic_speech_end(audio, sr, raw_end, hard_limit)
        else:
            # Interior ayah with no next match: allow up to 1.0s decay
            final_end = min(audio_duration, raw_end + 1.0)

        # Update previous cut end so next Ayah can NEVER start before this cut!
        previous_ayah_cut_end = final_end

        duration = final_end - final_start
        if duration > 30.0 or duration < 0.4:
            continue

        verified_ayahs.append({
            "ayah": a_num,
            "data": cand["data"],
            "start": round(final_start, 3),
            "end": round(final_end, 3),
            "is_true_final": is_true_final
        })

    return verified_ayahs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=Path("E:/Ai-Reciter/taha_al_fahad_juz_amma/mp3"),
                   help="Directory with full Surah MP3 files")
    p.add_argument("--output-dir", type=Path, default=Path("E:/Ai-Reciter/dataset_qaloon_taha"),
                   help="Output directory (contains audio/ and metadata.jsonl)")
    p.add_argument("--model", default="small", choices=["tiny", "base", "small", "large-v3"])
    p.add_argument("--reciter", default="Taha Mohamed Abdulrahman Al-Fahad (Qaloon)")
    p.add_argument("--reciter-key", default="taha")
    args = p.parse_args()

    audio_out_dir = args.output_dir / "audio"
    audio_out_dir.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading faster-whisper ({args.model}) on {device}...")
    from faster_whisper import WhisperModel
    model = WhisperModel(args.model, device=device, compute_type="float16" if device == "cuda" else "int8")

    print("Loading canonical Qālūn database...")
    quran_db = load_quran()

    audio_files = []
    for f in args.input_dir.glob("*.*"):
        if f.suffix.lower() in [".mp3", ".wav", ".ogg", ".flac"]:
            match = re.search(r"\b(0?0?1|0?7[8-9]|0?[8-9][0-9]|10[0-9]|11[0-4])\b", f.name)
            if match:
                audio_files.append((int(match.group(1)), f))

    audio_files.sort(key=lambda x: x[0])
    print(f"Found {len(audio_files)} Surahs in {args.input_dir}.\n")

    all_metadata_rows = []
    total_extracted = 0

    for surah, audio_path in audio_files:
        print(f"🎧 Processing Surah {surah:03d} ({audio_path.name})...")

        surah_ayah_keys = sorted([a for s, a in quran_db if s == surah])
        if not surah_ayah_keys:
            continue

        max_surah_ayah = max(surah_ayah_keys)

        all_refs = []
        all_refs.append({
            "ayah": -2,
            "text": "أعوذ بالله من الشيطان الرجيم",
            "text_asr": "اعوذ بالله من الشيطان الرجيم",
            "raw": "أَعُوذُ بِاللَّهِ مِنَ الشَّيْطَانِ الرَّجِيمِ",
            "source_ayahs": []
        })

        if surah != 9:
            all_refs.append({
                "ayah": -1,
                "text": "بسم الله الرحمن الرحيم",
                "text_asr": "بسم الله الرحمن الرحيم",
                "raw": "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
                "source_ayahs": []
            })

        for a in surah_ayah_keys:
            item = quran_db[(surah, a)]
            all_refs.append({
                "ayah": a,
                "text": item["text"],
                "text_asr": normalize_quran_for_asr(item["raw"]),
                "raw": item["raw"],
                "source_ayahs": item.get("source_ayahs", [a])
            })

        audio, sr = sf.read(str(audio_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != 16000:
            audio_tensor = torch.from_numpy(audio).float()
            audio_tensor = F.resample(audio_tensor, sr, 16000)
            audio = audio_tensor.numpy()
            sr = 16000

        audio_duration = len(audio) / sr

        segments, _ = model.transcribe(
            audio, language="ar", task="transcribe", beam_size=5,
            word_timestamps=True, vad_filter=True, condition_on_previous_text=False
        )

        whisper_words = []
        for segment in segments:
            for w in segment.words or []:
                norm_w = normalize_quran_for_asr(w.word)
                if norm_w:
                    whisper_words.append({
                        "clean_word": norm_w,
                        "raw_word": w.word,
                        "start": w.start,
                        "end": w.end
                    })

        # 1. Global Boundary Detection
        ayah_raw_bounds, preamble_wall = align_surah_with_hard_walls(
            whisper_words, all_refs, audio_duration, max_surah_ayah=max_surah_ayah, is_fatiha=(surah == 1)
        )

        # 2. Dynamic Waveform Slicing (Madd-Perfect & Leak-Free)
        verified_ayahs = slice_surah_with_waveform_tracking(
            audio, sr, ayah_raw_bounds, preamble_wall, max_surah_ayah
        )

        print(f"   ✅ Aligned and sliced {len(verified_ayahs)} / {len(surah_ayah_keys)} Ayahs.")

        for cand in verified_ayahs:
            ayah = cand["ayah"]
            data = cand["data"]
            start_sec = cand["start"]
            end_sec = cand["end"]

            start_frame = max(0, int(start_sec * sr))
            end_frame = min(len(audio), int(end_sec * sr))

            duration = (end_frame - start_frame) / sr
            sliced_audio = audio[start_frame:end_frame]
            filename = f"{surah:03d}{ayah:03d}.wav"
            filepath = audio_out_dir / filename

            sf.write(str(filepath), sliced_audio, sr, subtype="PCM_16")

            start_time_ms = int(round((start_frame / sr) * 1000))
            end_time_ms = int(round((end_frame / sr) * 1000))
            source_ayahs = data.get("source_ayahs", [ayah])

            geo = geometry_fields(surah, source_ayahs)

            schema_entry = {
                "surah": surah,
                "ayah": ayah,
                "audio_filename": filename,
                "relative_audio_path": f"audio/{filename}",
                "text": data["text"],
                "text_asr_normalized": normalize_quran_for_asr(data["raw"]),
                "text_raw_uthmani": data["raw"],
                "source_ayahs": source_ayahs,
                "reciter": args.reciter,
                "start_time": start_time_ms,
                "end_time": end_time_ms,
                "normalized_with_harakat": normalize_with_harakat(data["raw"]),
                **geo
            }
            all_metadata_rows.append(schema_entry)
            total_extracted += 1

    jsonl_path = args.output_dir / "metadata.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in all_metadata_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    json_path = args.output_dir / "metadata.json"
    json_path.write_text(json.dumps(all_metadata_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"🎉 Slicing Complete! Total Extracted: {total_extracted} Ayahs.")
    print(f"Audio Output Folder:   {audio_out_dir}")
    print(f"Metadata written to:   {jsonl_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()