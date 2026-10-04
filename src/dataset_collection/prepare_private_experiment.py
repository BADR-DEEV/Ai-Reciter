"""Record a requested private experiment, preserving the source-use uncertainty.

Does not create a training grant or human listening approvals. The public reuse
policy basis and the user's experimental quality acceptance stay distinct.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.training.reviewed_audio import sha256
from src.training.experiment_data import experimental_candidates


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--reciter", choices=["trabulsi", "taha"], default="trabulsi")
    p.add_argument("--source-inventory", type=Path)
    p.add_argument("--source-use-basis", choices=["publisher-public-material-reuse-policy", "user-directed-private-research"], default="publisher-public-material-reuse-policy")
    p.add_argument("--source-page")
    p.add_argument("--user-source-use-instruction")
    p.add_argument("--quality-acceptance", required=True, help="Actual user statement accepting an imperfect PRIVATE experiment, not teacher approval")
    args = p.parse_args()
    target = args.dataset / "experiment_decision.json"
    if target.exists():
        p.error("Experiment decision already exists; preserve it")
    rows = [json.loads(line) for line in (args.dataset / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    decision = {"reciter_key": args.reciter, "private_experiment_only": True,
        "user_quality_acceptance": args.quality_acceptance, "human_boundary_approved": False,
        "source_use_basis": args.source_use_basis, "explicit_ml_redistribution_grant": False,
        "production_release_allowed": False,
        "covered_source_sha256": sorted({row["source_sha256"] for row in rows}),
        "notes": "Private research instruction/policy evidence is not teacher approval or a verified ML grant. Original metadata and production admission are unchanged."}
    if args.source_use_basis == "publisher-public-material-reuse-policy":
        if not args.source_inventory or args.reciter != "trabulsi":
            p.error("Verified publisher policy requires the direct Trabulsi inventory")
        decision.update(source_inventory=str(args.source_inventory.resolve()), source_inventory_sha256=sha256(args.source_inventory))
    else:
        if not args.source_page or not args.user_source_use_instruction:
            p.error("Explicit private-research instruction and source page required")
        decision.update(source_page=args.source_page, user_source_use_instruction=args.user_source_use_instruction,
                        rights_status="unverified-user-asserted-open-access")
    target.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    accepted = experimental_candidates(args.dataset, target)
    (args.dataset / "metadata.experiment.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in accepted), encoding="utf-8")
    print(f"Private, user-accepted machine-QA candidates: {len(accepted)}. Original metadata and production review gate unchanged.")
