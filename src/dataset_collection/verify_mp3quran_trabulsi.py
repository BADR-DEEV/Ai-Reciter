"""Verify local full-surah recordings against discovered MP3Quran Qaloon links.

No existing audio is overwritten. This is source/reuse-policy evidence, not an
automatic human boundary approval or a legal certification of ML/public release.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import requests

PAGE = "https://www.mp3quran.net/eng/trabulsi-qalon"
POLICY = "https://www.mp3quran.net/eng/privacy"
QUOTE = "All rights are available to everyone, and we allow any visitor or developer to copy any material or use any link on the websites"


def allowed(url):
    origin = urlsplit(url)
    robots = f"{origin.scheme}://{origin.netloc}/robots.txt"
    response = requests.get(robots, timeout=30)
    if response.status_code in {404, 410}:
        return True
    response.raise_for_status()
    parser = RobotFileParser()
    parser.parse(response.text.splitlines())
    return parser.can_fetch("RattilSourceAudit", url)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("ahmad_tarabulsi/mp3"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--surahs", help="Subset; default Fatiha/Juz Amma")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Choose a new output directory; sources/reviewer work are never overwritten")
    if not allowed(PAGE) or not allowed(POLICY):
        parser.error("Publisher source/policy paths are robots-disallowed")
    page, policy = requests.get(PAGE, timeout=60), requests.get(POLICY, timeout=60)
    page.raise_for_status()
    policy.raise_for_status()
    links = {int(number): url for url, number in re.findall(r'(https://cdn\.mp3quran\.net/audio/ahmad-tarabulsi/r1/(\d{3})\.mp3)', page.text)}
    scope = {int(s) for s in args.surahs.split(",")} if args.surahs else {1, *range(78, 115)}
    if not scope.issubset({1, *range(78, 115)}) or scope - links.keys():
        parser.error("Every requested scoped Qaloon link must be discovered on the publisher page")
    if not all(allowed(links[s]) for s in scope):
        parser.error("Published audio path is robots-disallowed")
    def verify(surah):
        local = args.input_dir / f"{surah:03d}.mp3"
        response = requests.get(links[surah], timeout=120)
        response.raise_for_status()
        remote_hash = hashlib.sha256(response.content).hexdigest()
        local_hash = hashlib.sha256(local.read_bytes()).hexdigest() if local.is_file() else None
        return {"surah": surah, "source_file": str(local.resolve()), "source_url": links[surah],
                "source_sha256": local_hash, "publisher_sha256": remote_hash,
                "publisher_bytes": len(response.content), "byte_identical": local_hash == remote_hash,
                "reciter_key": "trabulsi", "riwayah": "Qaloon", "source_provider": "mp3quran-direct"}
    with ThreadPoolExecutor(max_workers=4) as pool:
        sources = list(pool.map(verify, sorted(scope)))
    plain = re.sub(r"<[^>]*>", " ", policy.text)
    reuse_verified = QUOTE in " ".join(plain.split())
    evidence = {"checked_at": datetime.now(timezone.utc).isoformat(), "source_page": PAGE,
                "policy_url": POLICY, "policy_sha256": hashlib.sha256(policy.content).hexdigest(),
                "public_material_reuse_clause_found": reuse_verified, "reuse_quote": QUOTE if reuse_verified else None,
                "ML_specific_clause_found": False, "public_release_legal_certification": False,
                "all_sources_byte_identical": all(s["byte_identical"] for s in sources), "sources": sources}
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "source_inventory.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "publisher-policy.html").write_bytes(policy.content)
    print(f"Verified {sum(s['byte_identical'] for s in sources)}/{len(sources)} local Trabulsi Qaloon recordings; public reuse clause found: {reuse_verified}")
    if not evidence["all_sources_byte_identical"] or not reuse_verified:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
