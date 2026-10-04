"""Assemble human-selected Trabulsi training rows; never approve from ASR alone."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.training.reviewed_audio import load_reviewed_trabulsi, sha256


def selected_reviews(decisions, passed):
    selected, seen = {}, set()
    for decision in decisions:
        key = (int(decision["surah"]), int(decision["ayah"]))
        if key in seen:
            raise ValueError(f"Duplicate review decision for {key}")
        seen.add(key)
        yes = lambda field: decision.get(field, "").strip().lower() == "true"
        if not yes("listening_approved") or not yes("boundary_approved"):
            continue
        if not decision.get("reviewer", "").strip() or not decision.get("reviewed_at", "").strip():
            raise ValueError(f"{key}: approvals require reviewer and reviewed_at")
        if (*key, decision["audio_sha256"]) not in passed:
            raise ValueError(f"{key}: selected clip failed the exact WAV/text audit or its hash changed")
        selected[key] = decision
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--validation-report", type=Path, required=True)
    p.add_argument("--decisions", type=Path, required=True, help="CSV actually completed by a listening reviewer")
    p.add_argument("--permission-file", type=Path, required=True)
    p.add_argument("--output-name", default="metadata.reviewed.jsonl")
    args = p.parse_args()
    output = args.dataset / args.output_name
    if Path(args.output_name).name != args.output_name or output.exists():
        p.error("Output must be a NEW file directly inside the dataset")
    audit = json.loads(args.validation_report.read_text(encoding="utf-8"))
    passed = {(r["surah"], r["ayah"], r["audio_sha256"]) for r in audit.get("clips", []) if r.get("automatic_pass") is True}
    with args.decisions.open(encoding="utf-8-sig", newline="") as handle:
        selected = selected_reviews(csv.DictReader(handle), passed)
    if not selected:
        p.error("No human listening/boundary approvals selected; blank or automatic decisions never admit clips")
    rows = [json.loads(l) for l in (args.dataset / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    reviewed = []
    for row in rows:
        decision = selected.get((row["surah"], row["ayah"]))
        if decision:
            if row["audio_sha256"] != decision["audio_sha256"]:
                p.error("Review decisions identify different audio")
            reviewed.append({**row, "reviewer": decision["reviewer"], "reviewed_at": decision["reviewed_at"],
                "review_notes": decision.get("notes", ""), "listening_approved": True, "boundary_approved": True,
                "alignment_status": "approved", "authorization_status": "authorized", "usable_for_training": True,
                "validation_report_sha256": sha256(args.validation_report)})
    if len(reviewed) != len(selected):
        p.error("A selected review decision does not exist in the candidate metadata")
    # Stage in the dataset (relative paths remain valid), validate rights/hashes
    # through the SAME admission loader as training, then atomically rename.
    staged = args.dataset / (args.output_name + ".pending")
    if staged.exists():
        p.error("Pending reviewer file exists; preserve/investigate it first")
    staged.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in reviewed), encoding="utf-8")
    try:
        splits = load_reviewed_trabulsi(staged, args.permission_file, validation_report=args.validation_report)
    except Exception:
        # Preserve the diagnostic pending file, but no accepted manifest is made.
        raise
    staged.rename(output)
    print(f"Accepted {len(reviewed)} independently reviewed/validated clips: {output}; partitions { {k: len(v) for k,v in splits.items()} }")


if __name__ == "__main__":
    main()
