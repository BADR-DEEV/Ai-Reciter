"""
Dataset Integrity Auditor: Detects missing Ayahs, orphaned WAV files,
and coverage percentages against the standard Quran text.
"""

import argparse
import json
from pathlib import Path
import re

# Standard Quranic Ayah counts for Juz' 'Amma (Surahs 78 to 114) + Al-Fatiha
SURAH_DATA = {
    1: {"name": "Al-Fatihah", "ayahs": 7},
    78: {"name": "An-Naba", "ayahs": 40},
    79: {"name": "An-Naziat", "ayahs": 46},
    80: {"name": "Abasa", "ayahs": 42},
    81: {"name": "At-Takwir", "ayahs": 29},
    82: {"name": "Al-Infitar", "ayahs": 19},
    83: {"name": "Al-Mutaffifin", "ayahs": 36},
    84: {"name": "Al-Inshiqaq", "ayahs": 25},
    85: {"name": "Al-Buruj", "ayahs": 22},
    86: {"name": "At-Tariq", "ayahs": 17},
    87: {"name": "Al-Ala", "ayahs": 19},
    88: {"name": "Al-Ghashiyah", "ayahs": 26},
    89: {"name": "Al-Fajr", "ayahs": 30},
    90: {"name": "Al-Balad", "ayahs": 20},
    91: {"name": "Ash-Shams", "ayahs": 15},
    92: {"name": "Al-Layl", "ayahs": 21},
    93: {"name": "Ad-Duha", "ayahs": 11},
    94: {"name": "Ash-Sharh", "ayahs": 8},
    95: {"name": "At-Tin", "ayahs": 8},
    96: {"name": "Al-Alaq", "ayahs": 19},
    97: {"name": "Al-Qadr", "ayahs": 5},
    98: {"name": "Al-Bayyinah", "ayahs": 8},
    99: {"name": "Az-Zalzalah", "ayahs": 8},
    100: {"name": "Al-Adiyat", "ayahs": 11},
    101: {"name": "Al-Qariah", "ayahs": 11},
    102: {"name": "At-Takathur", "ayahs": 8},
    103: {"name": "Al-Asr", "ayahs": 3},
    104: {"name": "Al-Humazah", "ayahs": 9},
    105: {"name": "Al-Fil", "ayahs": 5},
    106: {"name": "Quraish", "ayahs": 4},
    107: {"name": "Al-Maun", "ayahs": 7},
    108: {"name": "Al-Kawthar", "ayahs": 3},
    109: {"name": "Al-Kafirun", "ayahs": 6},
    110: {"name": "An-Nasr", "ayahs": 3},
    111: {"name": "Al-Masad", "ayahs": 5},
    112: {"name": "Al-Ikhlas", "ayahs": 4},
    113: {"name": "Al-Falaq", "ayahs": 5},
    114: {"name": "An-Nas", "ayahs": 6}
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path, default=Path("E:/Ai-Reciter/dataset_qaloon_taha"),
                   help="Directory containing sliced WAV files and metadata.json")
    p.add_argument("--start-surah", type=int, default=78, help="Start Surah (default: 78)")
    p.add_argument("--end-surah", type=int, default=114, help="End Surah (default: 114)")
    return p.parse_args()


def main():
    args = parse_args()
    meta_json_path = args.dataset_dir / "metadata.json"
    meta_jsonl_path = args.dataset_dir / "metadata.jsonl"

    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")

    # 1. Load Metadata
    metadata_entries = []
    if meta_json_path.exists():
        metadata_entries = json.loads(meta_json_path.read_text(encoding="utf-8"))
    elif meta_jsonl_path.exists():
        with meta_jsonl_path.open("r", encoding="utf-8") as f:
            metadata_entries = [json.loads(line) for line in f if line.strip()]
    else:
        raise FileNotFoundError(f"No metadata.json or metadata.jsonl found in {args.dataset_dir}")

    print("=" * 75)
    print(f"📊 DATASET INTEGRITY & MISSING AYAHS REPORT")
    print(f"📁 Target Folder: {args.dataset_dir}")
    print(f"📄 Loaded {len(metadata_entries)} records from metadata.")
    print("=" * 75)

    # Index metadata by (surah, ayah)
    meta_indexed = {(entry["surah"], entry["ayah"]): entry for entry in metadata_entries}

    # 2. Scan physical WAV files on disk
    disk_files = list(args.dataset_dir.glob("*.wav"))
    disk_indexed = {}
    for f in disk_files:
        # Match SSSAAA format (e.g. 078001.wav)
        m = re.search(r"(\d{3})(\d{3})\.wav$", f.name)
        if m:
            disk_indexed[(int(m.group(1)), int(m.group(2)))] = f
        else:
            # Fallback format (e.g. taha_078_001.wav)
            m2 = re.search(r"(\d{3})_(\d{3})\.wav$", f.name)
            if m2:
                disk_indexed[(int(m2.group(1)), int(m2.group(2)))] = f

    # 3. Check Disk vs Metadata Integrity
    missing_on_disk = []
    for (s, a), entry in meta_indexed.items():
        if (s, a) not in disk_indexed and not Path(entry.get("path", "")).exists():
            missing_on_disk.append((s, a))

    orphaned_on_disk = []
    for (s, a), path in disk_indexed.items():
        if (s, a) not in meta_indexed:
            orphaned_on_disk.append((s, a, path.name))

    # 4. Check Missing Ayahs against Expected Mushaf
    total_expected = 0
    total_found = 0
    surah_reports = []
    all_missing_ayahs_list = []

    for surah_num in range(args.start_surah, args.end_surah + 1):
        if surah_num not in SURAH_DATA:
            continue

        info = SURAH_DATA[surah_num]
        surah_name = info["name"]
        expected_count = info["ayahs"]
        total_expected += expected_count

        found_ayahs = []
        missing_ayahs = []

        for ayah_num in range(1, expected_count + 1):
            if (surah_num, ayah_num) in meta_indexed:
                found_ayahs.append(ayah_num)
                total_found += 1
            else:
                missing_ayahs.append(ayah_num)
                all_missing_ayahs_list.append({
                    "surah": surah_num,
                    "surah_name": surah_name,
                    "ayah": ayah_num,
                    "code": f"{surah_num:03d}{ayah_num:03d}"
                })

        coverage_pct = (len(found_ayahs) / expected_count) * 100
        surah_reports.append({
            "surah": surah_num,
            "name": surah_name,
            "expected": expected_count,
            "found": len(found_ayahs),
            "missing_count": len(missing_ayahs),
            "missing_ayahs": missing_ayahs,
            "coverage_pct": coverage_pct
        })

    # 5. Print Summary Table
    print(f"\n{'SURAH':<22} | {'FOUND':<9} | {'MISSING':<9} | {'COVERAGE':<10} | {'MISSING AYAHS'}")
    print("-" * 75)

    for rep in surah_reports:
        missing_str = ", ".join(map(str, rep["missing_ayahs"])) if rep["missing_ayahs"] else "✅ None"
        if len(missing_str) > 25:
            missing_str = missing_str[:22] + "..."
        
        status_color = "✅" if rep["missing_count"] == 0 else "⚠️ "
        print(f"{status_color} {rep['surah']:03d} {rep['name']:<15} | {rep['found']:>3}/{rep['expected']:<3}   | {rep['missing_count']:<9} | {rep['coverage_pct']:>6.1f}%   | {missing_str}")

    print("-" * 75)
    overall_coverage = (total_found / max(1, total_expected)) * 100
    print(f"🏆 OVERALL JUZ' 'AMMA COVERAGE: {total_found} / {total_expected} Ayahs ({overall_coverage:.2f}%)")
    print(f"❌ TOTAL MISSING AYAHS:         {len(all_missing_ayahs_list)}")

    # 6. Disk Warning Checks
    if missing_on_disk:
        print(f"\n🚨 WARNING: {len(missing_on_disk)} files listed in metadata DO NOT exist on disk!")
        for s, a in missing_on_disk[:5]:
            print(f"   - Surah {s:03d}, Ayah {a:03d}")
    if orphaned_on_disk:
        print(f"\n⚠️  WARNING: {len(orphaned_on_disk)} WAV files exist on disk but are MISSING from metadata:")
        for s, a, name in orphaned_on_disk[:5]:
            print(f"   - {name}")

    # 7. Save Detailed JSON Report
    output_report_path = args.dataset_dir / "missing_ayahs_report.json"
    report_data = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "total_expected": total_expected,
        "total_found": total_found,
        "total_missing": len(all_missing_ayahs_list),
        "overall_coverage_pct": round(overall_coverage, 2),
        "surah_breakdown": surah_reports,
        "missing_ayahs_list": all_missing_ayahs_list
    }

    output_report_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n💾 Full missing ayahs report saved to:\n   {output_report_path.resolve()}\n")


if __name__ == "__main__":
    main()