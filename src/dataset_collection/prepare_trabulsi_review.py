"""Create a separate corrected dataset from exact-checked, direct-source cuts.

Never overwrites the original Trabulsi dataset. Failed/unmatched references stay
in the segmentation bundle; they are not relabeled or fabricated to fill counts.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.dataset_collection.validate_ayah_audio import file_hash
from src.dataset_collection.segment_and_slice import previous_terminal_evidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--additional-candidates", type=Path, nargs="*", default=[], help="Separate scoped retry outputs; duplicate ayah IDs fail")
    p.add_argument("--source-inventory", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--bundle-only", action="store_true", help="Add ONLY a missing consolidated bundle to an identical previously prepared dataset")
    args = p.parse_args()
    if not args.bundle_only and args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error("Choose a NEW empty directory; never overwrite an existing dataset")
    source_root = args.candidates.resolve()
    inventory = json.loads(args.source_inventory.read_text(encoding="utf-8"))
    if inventory.get("source_page") != "https://www.mp3quran.net/eng/trabulsi-qalon" or inventory.get("public_material_reuse_clause_found") is not True:
        p.error("Need direct Trabulsi Qaloon publisher/reuse evidence")
    sources = {s["source_sha256"]: s for s in inventory["sources"] if s.get("byte_identical") is True
               and s["source_sha256"] == s["publisher_sha256"] and s.get("riwayah") == "Qaloon"}
    roots = [source_root, *(p.resolve() for p in args.additional_candidates)]
    rows = [(root, json.loads(line)) for root in roots for line in (root / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    reports_by_root = {root: {r["surah"]: r for r in json.loads((root / "review_bundle.json").read_text(encoding="utf-8"))["source_reports"]} for root in roots}
    checked, copy_paths, seen = [], [], set()
    for source_root, row in rows:
        previous = row.get("previous_boundary_ayah")
        report = reports_by_root[source_root][row["surah"]]
        if previous is not None and previous not in report["exported_ayahs"]:
            row = {**row, "previous_terminal_evidence": previous_terminal_evidence(report, previous, source_root)}
        key = row["surah"], row["ayah"]
        if key in seen:
            p.error(f"Duplicate ayah across retry outputs: {key}; never silently replace candidates")
        seen.add(key)
        verification = row.get("clip_verification", {})
        if not verification.get("exact_check", {}).get("passed") or not verification.get("passed"):
            p.error(f"{row['surah']}:{row['ayah']}: raw exact text gate not passed")
        if row.get("cleaning", {}).get("mode") != "none" and not verification.get("cleaned_check", {}).get("exact_check", {}).get("passed"):
            p.error(f"{row['surah']}:{row['ayah']}: cleaned exact text gate not passed")
        source = sources.get(row["source_sha256"])
        if not source or source["surah"] != row["surah"] or file_hash(row["source_file"]) != row["source_sha256"]:
            p.error("Candidate recording lacks matching immutable publisher provenance")
        for key, hash_key in (("relative_audio_path", "audio_sha256"), ("raw_audio_path", "raw_audio_sha256")):
            relative = Path(row[key])
            path = (source_root / relative).resolve()
            if relative.is_absolute() or source_root not in path.parents or file_hash(path) != row[hash_key]:
                p.error("Candidate WAV path/hash mismatch")
            copy_paths.append((path, relative))
        checked.append({**row, "reciter": "Ahmad Al-Trabulsi (Qaloon)", "reciter_key": "trabulsi",
            "source_provider": "mp3quran-direct", "source_url": source["source_url"],
            "source_inventory_sha256": file_hash(args.source_inventory), "authorization_status": "publisher-public-reuse-ML-review-needed",
            "alignment_status": "exact-auto-checked-needs-listening-review", "usable_for_training": False})
    if not checked:
        p.error("No exact-checked candidates to prepare")
    checked.sort(key=lambda row: (row["surah"], row["ayah"]))
    if args.bundle_only:
        if not (args.output_dir / "metadata.jsonl").is_file() or (args.output_dir / "review_bundle.json").exists():
            p.error("Bundle-only needs an existing prepared dataset and a MISSING review_bundle.json")
        existing = [json.loads(line) for line in (args.output_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if sorted(existing, key=lambda r: (r["surah"], r["ayah"])) != checked:
            p.error("Bundle-only candidate metadata differs; no existing files modified")
        for row in checked:
            if file_hash(args.output_dir / row["relative_audio_path"]) != row["audio_sha256"] or file_hash(args.output_dir / row["raw_audio_path"]) != row["raw_audio_sha256"]:
                p.error("Bundle-only dataset WAVs differ from checked candidates")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for source, relative in copy_paths:
            destination = args.output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        (args.output_dir / "metadata.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in checked), encoding="utf-8")
        (args.output_dir / "metadata.json").write_text(json.dumps(checked, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        shutil.copyfile(args.source_inventory, args.output_dir / "source_inventory.json")
    source_reports, prior_failures, coverage = {}, [], {}
    for root in roots:
        bundle = json.loads((root / "review_bundle.json").read_text(encoding="utf-8"))
        coverage_path = root / "coverage.jsonl"
        if coverage_path.is_file():
            for line in coverage_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row.get("review_audio"):
                    row["review_audio"] = {**row["review_audio"], "path": str((root / row["review_audio"]["path"]).resolve())}
                coverage[row["surah"], row["ayah"]] = row
        for report in bundle["source_reports"]:
            number = report["surah"]
            if number in source_reports and not source_reports[number].get("error"):
                p.error(f"Duplicate successful source reports for surah {number}")
            if report.get("error"):
                prior_failures.append({"bundle": str(root / "review_bundle.json"), "surah": number, "error": report["error"]})
            source_reports[number] = {**report, "original_bundle": str(root / "review_bundle.json"),
                "review_candidates": [{**row, "path": str((root / row["path"]).resolve())} for row in report.get("review_candidates", [])],
                "review_contexts": [{**row, "path": str((root / row["path"]).resolve())} for row in report.get("review_contexts", [])]}
    prepared_ids = {(r["surah"], r["ayah"]): r for r in checked}
    for key, row in list(coverage.items()):
        if key in prepared_ids:
            coverage[key] = {**row, **prepared_ids[key]}
    if coverage:
        if (args.output_dir / "coverage.jsonl").exists():
            p.error("Existing coverage inventory will not be overwritten")
        (args.output_dir / "coverage.jsonl").write_text("".join(json.dumps(coverage[key], ensure_ascii=False) + "\n" for key in sorted(coverage)), encoding="utf-8")
    (args.output_dir / "review_bundle.json").write_text(json.dumps({"metadata": checked,
        "coverage": {"expected_rows": len(coverage), "exported_rows": len(checked), "complete": bool(coverage) and len(coverage) == len(checked)},
        "source_reports": [source_reports[s] for s in sorted(source_reports)], "prior_failures": prior_failures,
        "source_inventory": inventory, "human_approved_clips": 0, "usable_for_training": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.bundle_only:
        print(f"Added consolidated review_bundle.json only; existing {len(checked)} candidates/metadata unchanged")
        return
    (args.output_dir / "repair_receipt.json").write_text(json.dumps({"segmentation_bundles": [{"path": str(root / "review_bundle.json"),
        "sha256": file_hash(root / "review_bundle.json")} for root in roots], "candidate_count": len(checked),
        "original_dataset_modified": False, "human_approved_clips": 0, "usable_for_training": False}, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(checked)} exact-checked, unapproved direct Trabulsi candidates in {args.output_dir}; originals unchanged")


if __name__ == "__main__":
    main()
