"""Copy an immutable candidate/audit package into dataset_collection.

Original runs stay intact. An existing legacy dataset may be archived ONLY with
an explicit destination, never deleted. Relocation verifies hashes and preserves
all approval flags; relocating an audit is not a new listening approval.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def organize(source, destination, audit_path=None, archive_existing=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination must be separate non-nested datasets")
    rows = [json.loads(line) for line in (source / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        for key, hash_key in (("relative_audio_path", "audio_sha256"), ("raw_audio_path", "raw_audio_sha256")):
            path = (source / row[key]).resolve()
            if source not in path.parents or digest(path) != row[hash_key]:
                raise ValueError(f"Invalid candidate path/hash: {path}")
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8")) if audit_path else None
    if audit and (Path(audit["dataset_root"]).resolve() != source or audit["metadata_sha256"] != digest(source / "metadata.jsonl")):
        raise ValueError("Audit does not identify this immutable source dataset")
    if destination.exists():
        if not archive_existing:
            raise ValueError("Destination exists; explicit archive path required, never overwrite")
        archive = Path(archive_existing).resolve()
        if archive.exists() or archive == destination or destination in archive.parents:
            raise ValueError("Archive must be new and outside the dataset being moved")
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(destination), str(archive))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    if digest(source / "metadata.jsonl") != digest(destination / "metadata.jsonl"):
        raise ValueError("Relocated metadata differs")
    for row in rows:
        for key, hash_key in (("relative_audio_path", "audio_sha256"), ("raw_audio_path", "raw_audio_sha256")):
            if digest(destination / row[key]) != row[hash_key]:
                raise ValueError("Relocated WAV differs")
    if audit:
        shutil.copy2(audit_path, destination / "validation.original.json")
        relocated = {**audit, "dataset_root": str(destination), "relocation": {
            "original_dataset_root": str(source), "original_report": str(Path(audit_path).resolve()),
            "original_report_sha256": digest(audit_path), "all_candidate_wav_hashes_verified": True,
            "new_listening_approval": False}}
        (destination / "validation.json").write_text(json.dumps(relocated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (destination / "collection_manifest.json").write_text(json.dumps({
        "original_dataset": str(source), "metadata_sha256": digest(destination / "metadata.jsonl"),
        "rows": len(rows), "all_paired_wav_hashes_verified": True, "archived_legacy": str(archive_existing) if archive_existing else None,
        "approval_flags_preserved": True, "human_approved": False}, indent=2) + "\n", encoding="utf-8")
    return len(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--destination", required=True, type=Path)
    p.add_argument("--audit", type=Path)
    p.add_argument("--archive-existing", type=Path)
    args = p.parse_args()
    print(f"Organized {organize(args.source, args.destination, args.audit, args.archive_existing)} unchanged candidate pairs at {args.destination}")
