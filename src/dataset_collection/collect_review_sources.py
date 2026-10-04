"""Download ONLY requested scope into a quarantined review inventory.

No imported recording enters training or the learner audio selector automatically.
Hub credentials come from the existing login, never command-line tokens.
"""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import time
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser

import requests

ROOT = Path(__file__).resolve().parents[2]
SCOPE = {1, *range(78, 115)}
REPO = "Mathani-Ayat/qaloon-reciter-experiments"
COLLECTION = "https://ar.assabile.com/taha-mohamed-abdulrahman-al-fahad-382/collection/al-mus-haf-al-murattal-416"
AGENT = "RattilResearch/1.0"


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def trabulsi(output):
    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi().dataset_info(REPO)
    files = [f.rfilename for f in info.siblings]
    selected = [f for f in files if f in {"README.md", "LICENSE", "sources.json", "release_manifest.json"}
                or f.startswith("dataset_qaloon_trabulsi/") and not f.endswith(".wav")
                or re.fullmatch(r"dataset_qaloon_trabulsi/audio_source/(?:001|07[89]|08\d|09\d|10\d|11[0-4])\d{3}\.wav", f)]
    if any(int(Path(f).stem[:3]) not in SCOPE for f in selected if f.endswith(".wav")):
        raise ValueError("Scope guard failed")
    output.mkdir(parents=True, exist_ok=True)
    # Per-file retry supports interrupted downloads without a broad snapshot.
    for number, file in enumerate(selected, 1):
        for attempt in range(3):
            try:
                hf_hub_download(REPO, file, repo_type="dataset", revision=info.sha, local_dir=output)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        if number % 50 == 0:
            print(f"Trabulsi: {number}/{len(selected)} files", flush=True)
    source = output / "dataset_qaloon_trabulsi/audio_source"
    rows = [json.loads(line) for line in (source / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    inventory = []
    for row in rows:
        if row["surah"] not in SCOPE:
            continue
        audio = source / Path(row["relative_audio_path"]).name
        if not audio.exists():
            continue
        digest = hashlib.sha256(audio.read_bytes()).hexdigest()
        if digest != row.get("audio_sha256", row.get("sha256")):
            raise ValueError(f"Hash mismatch: {audio.name}")
        inventory.append({**row, "relative_audio_path": audio.relative_to(output).as_posix(),
                          "usable_for_training": False, "alignment_status": "pending_review"})
    write_json(output / "inventory.json", {"repo": REPO, "revision": info.sha,
        "scope": sorted(SCOPE), "authorization_status": "unauthorized", "usable_for_training": False,
        "downloaded_clips": len(inventory), "clips": inventory})
    print(f"Verified {len(inventory)} scoped Trabulsi recordings. Quarantine: {output}", flush=True)


class CollectionParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tracks = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and attrs.get("data-collection") == "416" and attrs.get("data-default", "").isdigit():
            surah = int(attrs["data-default"])
            track = attrs.get("href", "").lstrip("#")
            if surah in SCOPE and track.isdigit():
                self.tracks[surah] = track


def assabile(output):
    session = requests.Session()
    session.headers["User-Agent"] = AGENT
    robots_url = "https://ar.assabile.com/robots.txt"
    r = session.get(robots_url, timeout=30)
    r.raise_for_status()
    robots = RobotFileParser(robots_url)
    robots.parse(r.text.splitlines())
    if not robots.can_fetch(AGENT, COLLECTION):
        raise ValueError("Collection is disallowed by robots.txt")
    page = session.get(COLLECTION, timeout=60)
    page.raise_for_status()
    if "قالون" not in page.text or "416" not in page.text:
        raise ValueError("Collection identity could not be confirmed")
    parser = CollectionParser()
    parser.feed(page.text)
    if set(parser.tracks) != SCOPE:
        raise ValueError("Expected exactly Fatiha + surahs 78–114")
    blocked = [p for p in ("/recitations/getUrl/", "/people/player/")
               if not robots.can_fetch(AGENT, urljoin(COLLECTION, p))]
    write_json(output / "inventory.json", {"source": COLLECTION, "scope": sorted(SCOPE),
        "reciter": "Taha Mohamed Abdulrahman Al-Fahad", "riwayah_from_source": "Qaloon an Nafi",
        "authorization_status": "needs-permission-review", "usable_for_training": False,
        "robots_sha256": hashlib.sha256(r.content).hexdigest(), "blocked_endpoints": blocked,
        "downloaded_audio": 0, "status": "audio-resolution-blocked-by-robots" if blocked else "needs-public-download-urls",
        "tracks": [{"surah": s, "source_track_id": parser.tracks[s],
                    "public_download_page": f"https://ar.assabile.com/download-surat-{parser.tracks[s]}.htm"} for s in sorted(SCOPE)],
        "next_step": "Obtain permitted direct audio URLs/permission. Public download pages currently redirect to HTML, not MP3. Do not automate disallowed resolution endpoints."})
    print(f"Assabile: 38 scoped tracks catalogued, 0 audio downloads; resolution endpoints blocked. {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=("trabulsi", "assabile"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    target = args.output or ROOT / "data/review_sources" / args.source
    (trabulsi if args.source == "trabulsi" else assabile)(target)
