"""Experimental machine-QA admission and leakage-safe adaptation partitions.

User acceptance of imperfect candidate cuts is NOT teacher approval. Public
download availability is NOT a training grant. These rows can never bypass the
production reviewed_audio gate; the experiment is private/non-release.
"""
import hashlib
import json
from pathlib import Path
from .reviewed_audio import sha256
from ..dataset_collection.qaloon_audio2text import load_quran, normalize_quran_for_asr, normalize_quran_for_asr_v1


def experimental_candidates(dataset, decision_file):
    root, decision_file = Path(dataset).resolve(), Path(decision_file).resolve()
    decision = json.loads(decision_file.read_text(encoding="utf-8"))
    if (decision.get("private_experiment_only") is not True or not decision.get("user_quality_acceptance")
            or decision.get("reciter_key") not in {"trabulsi", "taha"}):
        raise ValueError("Explicit private-experiment user quality acceptance required")
    reader = decision["reciter_key"]
    authorized = set(decision.get("covered_source_sha256", decision.get("authorized_source_sha256", [])))
    if not authorized:
        raise ValueError("Source-use basis must identify recording hashes")
    if decision.get("source_use_basis") == "publisher-public-material-reuse-policy":
        if reader != "trabulsi":
            raise ValueError("This verified public reuse basis covers direct Trabulsi only, not Assabile/Taha")
        inventory_path = Path(decision["source_inventory"])
        if sha256(inventory_path) != decision.get("source_inventory_sha256"):
            raise ValueError("Publisher inventory hash differs")
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        if (inventory.get("source_page") != "https://www.mp3quran.net/eng/trabulsi-qalon"
                or inventory.get("policy_url") != "https://www.mp3quran.net/eng/privacy"
                or inventory.get("public_material_reuse_clause_found") is not True):
            raise ValueError("Missing verified direct publisher reuse evidence")
        sources = {r["source_sha256"]: r for r in inventory["sources"] if r.get("byte_identical") is True
            and r["source_sha256"] == r.get("publisher_sha256") and r.get("source_provider") == "mp3quran-direct"
            and r.get("riwayah") == "Qaloon" and r.get("source_url", "").startswith("https://cdn.mp3quran.net/audio/ahmad-tarabulsi/r1/")}
        if not authorized <= sources.keys():
            raise ValueError("Source-use hashes not covered by the publisher inventory")
    elif decision.get("source_use_basis") == "explicit-training-grant":
        grant = Path(decision.get("grant_file", ""))
        if (not grant.is_file() or sha256(grant) != decision.get("grant_sha256")
                or decision.get("training_allowed") is not True or not decision.get("rights_holder")):
            raise ValueError("Explicit source-specific training grant evidence required")
        sources = None
    elif decision.get("source_use_basis") == "user-directed-private-research":
        # An explicit research instruction is NOT represented as a verified grant.
        # Production reviewed_audio admission is unchanged and still fail-closed.
        if (not decision.get("user_source_use_instruction") or not decision.get("source_page")
                or decision.get("rights_status") != "unverified-user-asserted-open-access"
                or decision.get("production_release_allowed") is not False
                or decision.get("explicit_ml_redistribution_grant") is not False):
            raise ValueError("User-directed research must retain unresolved rights and prohibit release")
        sources = None
    else:
        raise ValueError("Free downloads alone do not establish an experimental training-use basis")
    audit_path = root / "validation.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (Path(audit["dataset_root"]).resolve() != root or audit.get("complete_audit") is not True
            or audit.get("strict_content_comparison") is not True or audit.get("technical_only") is not False
            or audit["metadata_sha256"] != sha256(root / "metadata.jsonl")):
        raise ValueError("Full strict audit with unchanged dataset metadata required")
    passing = {(r["surah"], r["ayah"], r["audio_sha256"]) for r in audit["clips"]
        if r.get("automatic_pass") is True and not r.get("issues") and r.get("technical", {}).get("passed") is True
        and r.get("content", {}).get("passed") is True}
    # Existing strict audits bind ORIGINAL v1 labels. Do not reinterpret their
    # historical pass flags as a fresh v2 normalization/content audit.
    canonical = {k: normalize_quran_for_asr_v1(v["raw"]) for k, v in load_quran().items()}
    accepted, seen, source_hashes = [], set(), {}
    for line in (root / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if (row["surah"], row["ayah"], row.get("audio_sha256")) not in passing:
            continue
        audio = (root / row["relative_audio_path"]).resolve()
        if (root not in audio.parents or sha256(audio) != row["audio_sha256"] or row["audio_sha256"] in seen
                or row.get("reciter_key") != reader or row.get("source_sha256") not in authorized
                or row["text_asr_normalized"] != canonical.get((row["surah"], row["ayah"]))):
            raise ValueError("Candidate identity, source hash, canonical label or audio hash mismatch")
        source = Path(row["source_file"])
        if str(source) not in source_hashes:
            source_hashes[str(source)] = sha256(source)
        if source_hashes[str(source)] != row["source_sha256"]:
            raise ValueError("Immutable full recording hash mismatch")
        if sources and row.get("source_url") != sources[row["source_sha256"]]["source_url"]:
            raise ValueError("Direct-source URL differs")
        import wave
        with wave.open(str(audio), "rb") as wav:
            frames = wav.getnframes()
            if wav.getframerate() != 16000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2 or not 0 < frames <= 480000:
                raise ValueError("Expected intact <=30s mono PCM16 audio")
        if row.get("sample_rate") != 16000 or row["end_frame"] - row["start_frame"] != frames:
            raise ValueError("Exact sample grid mismatch")
        seen.add(row["audio_sha256"])
        accepted.append({**row, "path": str(audio), "duration_seconds": frames / 16000,
            "experimental_admission": "user-accepted-machine-QA-not-certified", "experiment_decision_sha256": sha256(decision_file),
            "source_use_basis": decision["source_use_basis"], "rights_status": decision.get("rights_status", "recorded-policy-or-grant-basis"),
            "original_content_audit_normalizer": "qaloon-asr-v1",
            "validation_report_sha256": sha256(audit_path)})
    if not accepted:
        raise ValueError("No exact passing experimental candidates")
    return accepted


def recording_safe_splits(rows, seed=42):
    """All voices/ayas from a SURAH share a bucket: no full-recording leakage."""
    surahs = sorted({r["surah"] for r in rows}, key=lambda s: hashlib.sha256(f"{seed}:surah:{s}".encode()).hexdigest())
    if len(surahs) < 10:
        raise ValueError("Need >=10 surah groups for the recording-held-out protocol")
    count = max(2, round(len(surahs) * .15))
    assignment = {s: "validation" if i < count else "test" if i < count * 2 else "train" for i, s in enumerate(surahs)}
    result = {name: [r for r in rows if assignment[r["surah"]] == name] for name in ("train", "validation", "test")}
    validate_partition_integrity(result)
    return result


def validate_partition_integrity(splits):
    seen_hashes, seen_sources, seen_ayahs = {}, {}, {}
    for name, rows in splits.items():
        if not rows:
            raise ValueError(f"Empty partition: {name}")
        for row in rows:
            for value, seen in ((sha256(row["path"]), seen_hashes), (row.get("source_sha256"), seen_sources),
                                ((row["surah"], row["ayah"]), seen_ayahs)):
                if value is None:
                    continue
                previous = seen.setdefault(value, name)
                if previous != name:
                    raise ValueError(f"Cross-partition audio/source/canonical-ID leakage: {value}")
    return True
