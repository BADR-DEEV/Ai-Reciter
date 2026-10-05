"""Pull the team's private Mathani-Ayat model and datasets at pinned revisions.

Needs a Hugging Face login with access to the Mathani-Ayat org (`hf auth login`).
Everything lands in Git-ignored folders; existing reciter folders are never replaced.
Progress is written to data/hf/pull-status.json (the web app shows it).
"""
import argparse
from datetime import datetime, timezone
from fnmatch import fnmatch
import json
import os
from pathlib import Path
import time

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

ROOT = Path(__file__).resolve().parents[2]
STATUS = ROOT / "data/hf/pull-status.json"
MODELS = {
    "v4": ("Mathani-Ayat/rattil-qaloon-v4", "eb62c35", ROOT / "runs/rattil_qaloon_v4"),  # default (joined-ayah training)
    "v3": ("Mathani-Ayat/rattil-qaloon-v3", "e9e59ac3db6cb096a0657f81f22365be189dda50", ROOT / "runs/rattil_qaloon_v3"),
}
DATASETS = {
    # Approved readers used by the web app (reference audio, challenges) plus Garu.
    "qaloon-reciter-dataset": "d0d2bdbbc757d09c05d1f0cdfabd67cc19420a83",
    # Waleed (held-out test voice) and Trabulsi: sources.json marks them unauthorized; private research only.
    "qaloon-reciter-experiments": "06cebc9ed98ef5e308da92aa4f09240460a760c7",
    # Six mp3quran readers added for v2/v3 training, plus an `unapproved/` folder.
    "qaloon-new-reciters": "ac53eb3c989d666c2cf6f68b9d008417356b23ab",
    # All ten approved readers relabelled with one normalizer (training the tajweed model); unapproved/ is skipped.
    "qaloon-all-reciters": "b5d51bcf2bdb7b53a0cd3bb580741e9304ed6183",
}
SKIP = {"qaloon-all-reciters": ["dataset_qaloon_*/*", "dataset_qaloon_*/*/*", "README.md"]}
# The web app reads these from src/dataset_collection/<folder> (web/lib/reciters.ts).
APP_READERS = ["dataset_qaloon_hutafi", "dataset_qaloon_Husary", "dataset_qaloon_dokali"]
RATE_WAIT = 300  # the Hub allows 1000 requests per 5 minutes


def status(state, **fields):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    data = {"state": state, "pid": os.getpid(), "updated": datetime.now(timezone.utc).isoformat(), **fields}
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(data), encoding="utf-8")
    temporary.replace(STATUS)


def denied(error):
    return getattr(getattr(error, "response", None), "status_code", None) in (401, 403, 404)


def hub_call(function, *args, **kwargs):
    while True:
        try:
            return function(*args, **kwargs)
        except (HfHubHTTPError, LocalEntryNotFoundError) as error:
            if denied(error):
                raise
            print(f"Hub unavailable or rate limited; retrying in {RATE_WAIT // 60} min", flush=True)
            time.sleep(RATE_WAIT)


def pull(repo_id, out, revision, repo_type="model", patterns=None):
    """Download until every expected file exists; snapshot_download alone returns partial folders when rate limited."""
    files = [f for f in hub_call(HfApi().list_repo_files, repo_id, repo_type=repo_type, revision=revision)
             if not f.startswith(".") and (patterns is None or any(fnmatch(f, p) for p in patterns))]
    stalled = 0
    while True:
        missing = [f for f in files if not (out / f).is_file()]
        status("downloading", repo=repo_id, done=len(files) - len(missing), total=len(files))
        if not missing:
            return
        try:
            snapshot_download(repo_id, repo_type=repo_type, revision=revision, local_dir=out, max_workers=4,
                              allow_patterns=missing if len(missing) <= 200 else patterns)
        except (HfHubHTTPError, LocalEntryNotFoundError) as error:
            if denied(error):
                raise
        remaining = sum(not (out / f).is_file() for f in missing)
        if remaining:
            stalled = stalled + 1 if remaining == len(missing) else 0
            if stalled >= 6:
                raise RuntimeError(f"No progress on {repo_id} after {stalled} attempts; {remaining} files missing")
            status("waiting", repo=repo_id, done=len(files) - remaining, total=len(files),
                   message="Hugging Face rate limit; resuming in 5 minutes")
            print(f"{repo_id}: {remaining} files still missing; resuming in {RATE_WAIT // 60} min", flush=True)
            time.sleep(RATE_WAIT)


def link_app_readers(dataset_root):
    for name in APP_READERS:
        target, link = dataset_root / name, ROOT / "src/dataset_collection" / name
        if link.is_symlink() and link.resolve() == target.resolve():
            continue
        if link.exists() or link.is_symlink():
            print(f"Kept existing {link.relative_to(ROOT)}; not linking the Hub copy.", flush=True)
            continue
        link.symlink_to(target, target_is_directory=True)
        print(f"Linked {link.relative_to(ROOT)} -> {target.relative_to(ROOT)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--models", nargs="*", choices=list(MODELS), default=["v4"], help="Which Rattil models (default: v4)")
    parser.add_argument("--datasets", nargs="*", choices=list(DATASETS), default=[d for d in DATASETS if d != "qaloon-all-reciters"],
                        help="Which dataset repos to pull (default: all but qaloon-all-reciters). Pass none to skip datasets.")
    parser.add_argument("--readers-only", action="store_true",
                        help="Only the web app's reference readers (Huthaify, Husary, Dokali; ~330 MB)")
    args = parser.parse_args()
    if args.readers_only:
        args.datasets = ["qaloon-reciter-dataset"]
    try:
        for name in [] if args.skip_model else args.models:
            repo, revision, out = MODELS[name]
            pull(repo, out, revision)
            print(f"Model {repo}@{revision[:7]} -> {out.relative_to(ROOT)}", flush=True)
        for name in args.datasets:
            out = ROOT / "data/hf" / name
            readers = [f"{reader}/*" for reader in APP_READERS] if args.readers_only else SKIP.get(name)
            pull(f"Mathani-Ayat/{name}", out, DATASETS[name], repo_type="dataset", patterns=readers)
            print(f"Dataset {name}@{DATASETS[name][:7]} -> {out.relative_to(ROOT)}", flush=True)
            if name == "qaloon-reciter-dataset":
                link_app_readers(out)
    except BaseException as error:
        status("failed", message=f"{type(error).__name__}: {error}"[:500])
        raise
    status("done")


if __name__ == "__main__":
    main()
