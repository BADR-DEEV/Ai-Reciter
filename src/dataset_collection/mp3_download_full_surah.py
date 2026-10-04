"""
Downloader for Sheikh Ahmad Al-Tarabulsi (Qaloon 'an Nafi') from MP3Quran CDN.
URL: https://cdn.mp3quran.net/audio/marwan-akri/r1/
"""

import argparse
import os
from pathlib import Path
import sys
import time
import requests

BASE_URL = "https://cdn.mp3quran.net/audio/marwan-akri/r1"

# Backup mirror on MP3Quran server in case CDN has rate-limiting
FALLBACK_URL = "https://cdn.mp3quran.net/audio/marwan-akri/r1"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Referer": "https://www.mp3quran.net/",
}

SURAH_NAMES = {
    1: "Al-Fatihah", 2: "Al-Baqarah", 3: "Ali 'Imran", 4: "An-Nisa", 5: "Al-Ma'idah",
    6: "Al-An'am", 7: "Al-A'raf", 8: "Al-Anfal", 9: "At-Tawbah", 10: "Yunus",
    78: "An-Naba", 79: "An-Naziat", 80: "Abasa", 81: "At-Takwir", 82: "Al-Infitar",
    83: "Al-Mutaffifin", 84: "Al-Inshiqaq", 85: "Al-Buruj", 86: "At-Tariq", 87: "Al-Ala",
    88: "Al-Ghashiyah", 89: "Al-Fajr", 90: "Al-Balad", 91: "Ash-Shams", 92: "Al-Layl",
    93: "Ad-Duha", 94: "Ash-Sharh", 95: "At-Tin", 96: "Al-Alaq", 97: "Al-Qadr",
    98: "Al-Bayyinah", 99: "Az-Zalzalah", 100: "Al-Adiyat", 101: "Al-Qariah", 102: "At-Takathur",
    103: "Al-Asr", 104: "Al-Humazah", 105: "Al-Fil", 106: "Quraish", 107: "Al-Maun",
    108: "Al-Kawthar", 109: "Al-Kafirun", 110: "An-Nasr", 111: "Al-Masad", 112: "Al-Ikhlas",
    113: "Al-Falaq", 114: "An-Nas"
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("E:/Ai-Reciter/ahmad_tarabulsi/mp3"),
                   help="Directory where MP3s will be saved")
    p.add_argument("--start", type=int, default=1, help="Start Surah (1 to 114)")
    p.add_argument("--end", type=int, default=114, help="End Surah (1 to 114)")
    p.add_argument("--juz-amma", action="store_true",
                   help="Quick shortcut: download Sūrat al-Fātiḥah (1) + Juz' 'Amma (78-114)")
    return p.parse_args()


def download_file(url: str, dest_path: Path) -> bool:
    """Streams download with resume capability and file size check."""
    temp_path = dest_path.with_suffix(".tmp")
    try:
        with requests.get(url, headers=HEADERS, stream=True, timeout=30) as r:
            if r.status_code != 200:
                return False

            total_size = int(r.headers.get("content-length", 0))
            downloaded = 0
            start_time = time.time()

            with open(temp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

            # Move temp file to destination if valid
            if temp_path.exists() and temp_path.stat().st_size > 5000:
                temp_path.replace(dest_path)
                return True
            else:
                if temp_path.exists():
                    temp_path.unlink()
                return False

    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        return False


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.juz_amma:
        surah_list = [1, *range(78, 115)]
    else:
        surah_list = list(range(args.start, args.end + 1))

    print("=" * 70)
    print("🚀 MP3Quran Downloader: Sheikh Ahmad Al-Tarabulsi (Qālūn)")
    print(f"📁 Destination: {args.output_dir.resolve()}")
    print(f"📖 Total Surahs to download: {len(surah_list)}")
    print("=" * 70 + "\n")

    successful = 0
    failed = []

    for i, s in enumerate(surah_list, start=1):
        filename = f"{s:03d}.mp3"
        dest_file = args.output_dir / filename
        surah_title = SURAH_NAMES.get(s, f"Surah {s}")

        # Skip already downloaded files
        if dest_file.exists() and dest_file.stat().st_size > 50000:
            size_mb = dest_file.stat().st_size / (1024 * 1024)
            print(f"[{i}/{len(surah_list)}] ⏩ {filename} ({surah_title}) already exists ({size_mb:.2f} MB). Skipping.")
            successful += 1
            continue

        print(f"[{i}/{len(surah_list)}] 📥 Downloading {filename} ({surah_title})...", end="", flush=True)

        primary_url = f"{BASE_URL}/{filename}"
        ok = download_file(primary_url, dest_file)

        # Fallback to secondary server if primary CDN drops
        if not ok:
            backup_url = f"{FALLBACK_URL}/{filename}"
            print(" (retrying via mirror)...", end="", flush=True)
            ok = download_file(backup_url, dest_file)

        if ok:
            size_mb = dest_file.stat().st_size / (1024 * 1024)
            print(f" ✅ Done ({size_mb:.2f} MB)")
            successful += 1
        else:
            print(" ❌ Failed!")
            failed.append(s)

        time.sleep(0.3)  # Gentle rate-limiting

    print("\n" + "=" * 70)
    print(f"🎉 Download Summary: {successful}/{len(surah_list)} Surahs ready.")
    if failed:
        print(f"⚠️ Failed Surahs: {failed}")
    print(f"📁 Files saved in: {args.output_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()