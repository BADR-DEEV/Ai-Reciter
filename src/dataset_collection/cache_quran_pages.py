"""Cache MP3Quran geometry once; enrich datasets without changing audio timings.

Run from the repository root. Existing metadata is backed up before atomic updates.
Geometry is keyed by source ayah (before this project's Qaloon text merges).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET

import requests

from qaloon_audio2text import load_quran, normalize_quran_for_asr, normalize_with_harakat

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "web" / "public" / "quran"
API = "https://www.mp3quran.net/api/v3/ayat_timing"
SVG_BASE = "https://www.mp3quran.net/api/quran_pages_svg/"
READ_ID = 5  # Hafs-numbered geometry matches the page artwork, unlike Qaloon timing IDs.


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def get(url, **kwargs):
    for attempt in range(4):
        try:
            response = requests.get(url, timeout=60, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def cache_timing(surah):
    path = PUBLIC / "geometry" / f"{surah:03d}.json"
    if path.exists():
        return surah, json.loads(path.read_text(encoding="utf-8"))
    rows = get(API, params={"surah": surah, "read": READ_ID}).json()
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Missing geometry for surah {surah}: {rows}")
    # Do not store the chosen sheikh's timings as shared geometry.
    rows = [{k: row[k] for k in ("ayah", "polygon", "x", "y", "page")} for row in rows]
    atomic_json(path, rows)
    return surah, rows


def cache_page(url):
    if not re.fullmatch(re.escape(SVG_BASE) + r"\d{3}\.svg", url):
        raise ValueError(f"Unexpected page URL: {url}")
    filename = url.rsplit("/", 1)[1]
    path = PUBLIC / "pages" / filename
    if not path.exists():
        content = get(url).content
        root = ET.fromstring(content)
        if root.tag != "{http://www.w3.org/2000/svg}svg":
            raise ValueError(f"Invalid SVG: {url}")
        # Assets are served as images, not injected into the document.
        for node in root.iter():
            if node.tag.rsplit("}", 1)[-1] in {"script", "foreignObject"}:
                raise ValueError(f"Unsafe SVG: {url}")
            if any(key.lower().startswith("on") for key in node.attrib):
                raise ValueError(f"Unsafe SVG attributes: {url}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".svg.tmp")
        temp.write_bytes(content)
        temp.replace(path)
    root = ET.parse(path).getroot()
    return filename, {"path": f"/quran/pages/{filename}", "viewBox": root.attrib["viewBox"]}


def load_hafs_reference():
    path = PUBLIC / "hafs-reference.json"
    if not path.exists():
        atomic_json(path, get("https://raw.githubusercontent.com/risan/quran-json/main/dist/quran.json").json())
    return json.loads(path.read_text(encoding="utf-8"))


def build_text_mapping(raw, hafs):
    """Align nearly-identical text streams, not inconsistent riwayah numbering.

    Shared whole-ayah regions are flagged; we cannot pretend a whole Hafs
    polygon is a precise polygon for a differently divided Qaloon ayah.
    """
    mapping = {}
    for surah in hafs:
        s = int(surah["id"])
        qrows = [row for row in raw if int(row["sura_no"]) == s]
        qtokens, qowners, htokens, howners = [], [], [], []
        for row in qrows:
            tokens = normalize_quran_for_asr(row["aya_text"]).split()
            qtokens.extend(tokens)
            qowners.extend([int(row["aya_no"])] * len(tokens))
        for verse in surah["verses"]:
            tokens = normalize_quran_for_asr(verse["text"]).split()
            htokens.extend(tokens)
            howners.extend([int(verse["id"])] * len(tokens))
        linked = {}
        for block in SequenceMatcher(None, qtokens, htokens, autojunk=False).get_matching_blocks():
            for offset in range(block.size):
                linked.setdefault(qowners[block.a + offset], set()).add(howners[block.b + offset])
        for row in qrows:
            ayah = int(row["aya_no"])
            mapping[(s, ayah)] = sorted(linked.get(ayah, []))
        # Basmalah in Hafs Fatiha is numbered; in Qaloon it is an introduction.
        mapping[(s, 0)] = [1] if s == 1 else []
    return mapping


def regions_for(surah, source_ayahs, source, mapping):
    indexed = {int(row["ayah"]): row for row in source[surah]}
    regions = []
    for ayah in source_ayahs:
        for display_ayah in mapping.get((surah, ayah), []):
            row = indexed.get(display_ayah)
            if row is None or not row.get("page") or not row.get("polygon"):
                continue
            if any(r["display_ayah"] == display_ayah for r in regions):
                continue
            regions.append({"source_ayah": ayah, "display_ayah": display_ayah,
                            "polygon": row["polygon"], "x": row["x"], "y": row["y"],
                            "page": "/quran/pages/" + row["page"].rsplit("/", 1)[1]})
    return regions


def enrich_metadata(path, source, quran, mapping):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        surah, ayah = int(row["surah"]), int(row["ayah"])
        if surah not in source:
            continue
        sources = row.get("source_ayahs")
        if sources is None:
            # The legacy Waleed copy numbers its audio, not Qaloon text.
            if "walid" in str(path).lower() and surah == 1:
                sources = [0] if ayah == 1 else [6, 7] if ayah == 7 else [ayah - 1]
            else:
                sources = quran.get((surah, ayah), {}).get("source_ayahs", [ayah])
        regions = regions_for(surah, sources, source, mapping)
        row["quran_regions"] = regions
        row["geometry_source"] = f"mp3quran:read{READ_ID}:text-aligned"
        for key in ("polygon", "x", "y", "page"):
            row.pop(key, None)
        # Convenience fields for existing metadata consumers. Multi-region
        # ayahs MUST use quran_regions, not just the first polygon.
        if len(regions) == 1:
            row.update({key: regions[0][key] for key in ("polygon", "x", "y", "page")})
        # Deliberately leave original start_time/end_time and labels untouched.
    backup = path.with_suffix(path.suffix + ".before-geometry.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temp.replace(path)
    print(f"Enriched {path.relative_to(ROOT)} ({len(rows)} rows)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained-only", action="store_true", help="Only Fatiha and surahs 78–114")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--skip-metadata", action="store_true")
    args = parser.parse_args()
    surahs = [1, *range(78, 115)] if args.trained_only else list(range(1, 115))
    quran = load_quran()
    raw = json.loads((Path(__file__).parent / "QaloonData_v10(1).json").read_text(encoding="utf-8"))
    mapping = build_text_mapping(raw, load_hafs_reference())
    names = {int(row["sura_no"]): (row["sura_name_en"], row["sura_name_ar"]) for row in raw}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        source = dict(pool.map(cache_timing, surahs))
        urls = sorted({row["page"] for rows in source.values() for row in rows if row.get("page")})
        print(f"Caching {len(urls)} unique pages for {len(surahs)} surahs...", flush=True)
        pages = dict(pool.map(cache_page, urls))
    catalog = []
    for surah in surahs:
        ayahs = []
        for (s, a), row in sorted(quran.items()):
            if s != surah:
                continue
            ayahs.append({"ayah": a, "text": row["raw"],
                          "displayText": normalize_with_harakat(row["raw"]),
                          "normalized": normalize_quran_for_asr(row["raw"]),
                          "regions": regions_for(s, row["source_ayahs"], source, mapping)})
        display_owners = {}
        for ayah in ayahs:
            for region in ayah["regions"]:
                display_owners.setdefault(region["display_ayah"], set()).add(ayah["ayah"])
        precise = all(ayah["regions"] for ayah in ayahs) and all(len(owners) == 1 for owners in display_owners.values())
        entry = {"id": surah, "name": names[surah][0], "arabic": names[surah][1],
                 "trained": surah == 1 or surah >= 78, "ayahCount": len(ayahs),
                 "preciseGeometry": precise}
        atomic_json(PUBLIC / "surahs" / f"{surah:03d}.json", {**entry, "ayahs": ayahs})
        catalog.append(entry)
    atomic_json(PUBLIC / "text-mapping.json", {f"{s}:{a}": v for (s, a), v in mapping.items()})
    atomic_json(PUBLIC / "manifest.json", {"source": f"MP3Quran (read {READ_ID}, text-aligned shared geometry)",
                                           "surahs": catalog, "pages": pages})
    if not args.skip_metadata:
        paths = list(Path(__file__).parent.glob("dataset_qaloon_*/metadata.jsonl"))
        paths += list((ROOT / "walid_qaloon").glob("metadata.jsonl"))
        for path in paths:
            enrich_metadata(path, source, quran, mapping)
    print("Quran assets ready. Subsequent runs reuse cached geometry and SVGs.", flush=True)


if __name__ == "__main__":
    main()
