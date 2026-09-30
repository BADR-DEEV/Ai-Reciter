"""Shared, text-aligned geometry for dataset builders (never reciter timings)."""
from functools import lru_cache
import json
from pathlib import Path

PUBLIC = Path(__file__).resolve().parents[2] / "web" / "public" / "quran"


@lru_cache(maxsize=114)
def surah_regions(surah):
    path = PUBLIC / "surahs" / f"{surah:03d}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {ayah["ayah"]: ayah["regions"] for ayah in data["ayahs"]}


def geometry_fields(surah, canonical_ayahs):
    regions = []
    cached = surah_regions(surah)
    for ayah in canonical_ayahs:
        for region in cached.get(ayah, []):
            if not any(r["display_ayah"] == region["display_ayah"] for r in regions):
                regions.append(region)
    fields = {"quran_regions": regions, "geometry_source": "mp3quran:read5:text-aligned"}
    if len(regions) == 1:
        fields.update({key: regions[0][key] for key in ("polygon", "x", "y", "page")})
    return fields
