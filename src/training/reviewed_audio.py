"""Fail-closed admission of separately reviewed/authorized Trabulsi clips.

Acoustic approval cannot grant recording rights. A source-level permission
receipt and per-clip listening approval are both required; no auto-approval.
"""
import hashlib
import json
from pathlib import Path
import wave


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reviewed_trabulsi(manifest, permission_file, seed=42, canonical=None, validation_report=None):
    manifest, permission_file = Path(manifest), Path(permission_file)
    receipt = json.loads(permission_file.read_text(encoding="utf-8"))
    direct = receipt.get("source_provider") == "mp3quran-direct"
    if (receipt.get("reciter_key") != "trabulsi" or receipt.get("training_allowed") is not True
            or not receipt.get("grant_reference") or not receipt.get("rights_holder")
            or not receipt.get("reviewed_by") or not receipt.get("reviewed_at")
            or (not direct and receipt.get("supersedes_unauthorized_source_restriction") is not True)):
        raise ValueError("Trabulsi requires an explicit rights-holder training grant superseding the known unauthorized-source restriction; segmentation approval is insufficient")
    authorized_hashes = set(receipt.get("authorized_source_sha256", []))
    if not authorized_hashes or any(not isinstance(h, str) or len(h) != 64 for h in authorized_hashes):
        raise ValueError("Permission receipt must identify authorized recording SHA-256 hashes")
    if direct:
        evidence_path = Path(receipt.get("source_inventory", ""))
        evidence_path = evidence_path if evidence_path.is_absolute() else permission_file.parent / evidence_path
        if not evidence_path.is_file() or sha256(evidence_path) != receipt.get("source_inventory_sha256"):
            raise ValueError("Direct MP3Quran source inventory must exist and match its recorded hash")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if (evidence.get("source_page") != "https://www.mp3quran.net/eng/trabulsi-qalon"
                or evidence.get("policy_url") != "https://www.mp3quran.net/eng/privacy"
                or evidence.get("public_material_reuse_clause_found") is not True):
            raise ValueError("Missing direct Qaloon publisher/reuse evidence")
        verified = {s["source_sha256"]: s for s in evidence.get("sources", []) if s.get("byte_identical") is True
                    and s.get("source_sha256") == s.get("publisher_sha256") and s.get("source_provider") == "mp3quran-direct"
                    and s.get("riwayah") == "Qaloon"
                    and s.get("source_url", "").startswith("https://cdn.mp3quran.net/audio/ahmad-tarabulsi/r1/")}
        if not authorized_hashes.issubset(verified):
            raise ValueError("Receipt includes recordings not verified against the direct Qaloon publisher")
    if validation_report is None:
        raise ValueError("A complete strict WAV/text validation report is required")
    audit = json.loads(Path(validation_report).read_text(encoding="utf-8"))
    if (audit.get("complete_audit") is not True or audit.get("strict_content_comparison") is not True
            or audit.get("technical_only") is not False):
        raise ValueError("Validation must audit every candidate with strict blind ASR and WAV checks")
    if Path(audit.get("dataset_root", "")).resolve() != manifest.parent.resolve():
        raise ValueError("Validation report belongs to a different dataset")
    original_metadata = manifest.parent / "metadata.jsonl"
    if not original_metadata.is_file() or sha256(original_metadata) != audit.get("metadata_sha256"):
        raise ValueError("Candidate metadata changed since validation")
    passed = {(r["surah"], r["ayah"], r["audio_sha256"]): r for r in audit.get("clips", [])
              if r.get("automatic_pass") is True and r.get("technical", {}).get("passed") is True
              and r.get("content", {}).get("passed") is True and not r.get("issues")}
    if canonical is None:
        from src.dataset_collection.qaloon_audio2text import load_quran, normalize_quran_for_asr
        canonical = {key: normalize_quran_for_asr(row["raw"]) for key, row in load_quran().items()}
    partitions = {name: [] for name in ("train", "validation", "test")}
    seen_audio = set()
    source_hashes = {}
    receipt_hash, manifest_hash = sha256(permission_file), sha256(manifest)
    audit_hash = sha256(validation_report)
    root = manifest.parent.resolve()
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            prefix = f"{manifest}:{line_number}"
            if (row.get("reciter_key") != "trabulsi" or row.get("usable_for_training") is not True
                    or row.get("alignment_status") != "approved" or not row.get("reviewer")
                    or not row.get("reviewed_at") or row.get("listening_approved") is not True
                    or row.get("boundary_approved") is not True or row.get("authorization_status") != "authorized"):
                raise ValueError(f"{prefix}: requires source authorization and human listening/boundary approval")
            if row.get("source_sha256") not in authorized_hashes:
                raise ValueError(f"{prefix}: recording is not covered by the permission receipt")
            source = Path(row.get("source_file", ""))
            source = source if source.is_absolute() else root / source
            source = source.resolve()
            if source.is_file() and source not in source_hashes:
                source_hashes[source] = sha256(source)
            if source_hashes.get(source) != row["source_sha256"]:
                raise ValueError(f"{prefix}: immutable source hash mismatch/missing recording")
            relative = Path(row.get("relative_audio_path", ""))
            audio = (root / relative).resolve()
            if relative.is_absolute() or root not in audio.parents or not audio.is_file():
                raise ValueError(f"{prefix}: audio path must stay inside the reviewed dataset")
            audio_hash = sha256(audio)
            if row.get("audio_sha256") != audio_hash or audio_hash in seen_audio:
                raise ValueError(f"{prefix}: candidate audio hash mismatch or duplicate")
            seen_audio.add(audio_hash)
            key = (row.get("surah"), row.get("ayah"))
            if (*key, audio_hash) not in passed:
                raise ValueError(f"{prefix}: clip did not pass exact WAV/text validation")
            if direct and (row.get("source_provider") != "mp3quran-direct"
                           or row.get("source_url") != verified[row["source_sha256"]]["source_url"]):
                raise ValueError(f"{prefix}: direct-source provenance mismatch")
            if key not in canonical or row.get("text_asr_normalized") != canonical[key]:
                raise ValueError(f"{prefix}: label must match the canonical Qaloon ayah (never infer labels from filename)")
            if key[0] not in {1, *range(78, 115)}:
                raise ValueError(f"{prefix}: reviewed training remains scoped to Fatiha/Juz Amma")
            with wave.open(str(audio), "rb") as wav:
                if wav.getframerate() != 16000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                    raise ValueError(f"{prefix}: expected 16 kHz mono PCM16 WAV")
                frames = wav.getnframes()
                seconds = frames / 16000
            if not 0 < seconds <= 30:
                raise ValueError(f"{prefix}: audio must fit 30 seconds without transcript truncation")
            if (row.get("sample_rate") != 16000 or not isinstance(row.get("start_frame"), int)
                    or not isinstance(row.get("end_frame"), int) or row["start_frame"] < 0
                    or row["end_frame"] - row["start_frame"] != frames):
                raise ValueError(f"{prefix}: invalid/mismatched reviewed frame boundaries")
            # Every cut/repetition from a recording belongs to ONE partition.
            fraction = int.from_bytes(hashlib.sha256(f"{seed}:recording:{row['source_sha256']}".encode()).digest()[:8], "big") / 2**64
            partition = "test" if fraction < .1 else "validation" if fraction < .2 else "train"
            partitions[partition].append({**row, "path": str(audio), "duration_seconds": seconds,
                "permission_receipt_sha256": receipt_hash, "review_manifest_sha256": manifest_hash,
                "validation_report_sha256": audit_hash})
    if not seen_audio:
        raise ValueError("Reviewed Trabulsi manifest contains no approved clips")
    return partitions
