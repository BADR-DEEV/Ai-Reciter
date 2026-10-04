"""Restore the pinned private full-model release; never overwrite other work."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]


def restore(destination, cache_only=False):
    receipt = json.loads((Path(__file__).with_name("huggingface_release.json")).read_text(encoding="utf-8"))
    artifacts = {name: digest for name, digest in receipt["artifact_sha256"].items() if name.startswith("models/full-finetuning/")}
    if "models/full-finetuning/model.safetensors" not in artifacts:
        raise ValueError("Release receipt has no full-model weights")
    destination.mkdir(parents=True, exist_ok=True)
    # Fail before downloads if an existing file differs. Never overwrite an
    # unknown training run, checkpoint, reviewer file or partially different model.
    for remote, expected in artifacts.items():
        target = destination / Path(remote).name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Existing file differs from pinned release; choose another directory: {target}")
    for remote, expected in artifacts.items():
        target = destination / Path(remote).name
        if target.exists():
            continue
        downloaded = Path(hf_hub_download(receipt["repo_id"], remote, revision=receipt["commit"], local_files_only=cache_only))
        if hashlib.sha256(downloaded.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Published artifact checksum mismatch: {remote}")
        # Materialize completely before publishing a filename the model loader
        # might read. Exclusive hard-link creation never replaces another file.
        with tempfile.NamedTemporaryFile(dir=destination, prefix=".restore-", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            try:
                with downloaded.open("rb") as source:
                    shutil.copyfileobj(source, output)
            except BaseException:
                output.close()
                temporary.unlink(missing_ok=True)
                raise
        try:
            os.link(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Verified {target.name}", flush=True)
    print(f"Restored {len(artifacts)} pinned full-model artifacts to {destination}. No upload/retraining.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/gpu_base_full")
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()
    restore(args.output, args.cache_only)
