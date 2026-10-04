"""Audit WAV engineering AND blind, exact normalized Quran-text agreement.

ASR agreement is an automatic gate, NOT acoustic/riwayah certification. This
script never modifies data, rewrites missing words, or grants human approval.
"""
import argparse
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import wave

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.dataset_collection.qaloon_audio2text import load_quran, normalize_quran_for_asr
from src.dataset_collection.segment_and_slice import frame_energy, transcribe_words, exact_text_check, VERSION


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def inspect_wav(path, row=None):
    issues, warnings = [], []
    with wave.open(str(path), "rb") as wav:
        sr, channels, width, frames = wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()
        compression = wav.getcomptype()
    if (sr, channels, width, compression) != (16000, 1, 2, "NONE"):
        issues.append("require-16000Hz-mono-PCM16-WAV")
    audio, read_sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != read_sr or len(audio) != frames or not len(audio) or not np.isfinite(audio).all():
        issues.append("empty-nonfinite-or-inconsistent-audio")
    mono = audio.mean(axis=1)
    duration = frames / sr
    if not .25 <= duration <= 30:
        issues.append("duration-outside-0.25-to-30-seconds")
    peak = float(np.max(np.abs(audio))) if len(audio) else 0
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) if len(audio) else 0
    clipped = float(np.mean(np.abs(audio) >= .999)) if len(audio) else 0
    dc = float(np.mean(mono)) if len(audio) else 0
    if rms < 1e-5:
        issues.append("silent-or-near-silent")
    if clipped > .001:
        issues.append("excessive-full-scale-clipping")
    if abs(dc) > .01:
        issues.append("excessive-DC-offset")
    _, energies, _ = frame_energy(mono, sr)
    voiced = float(np.quantile(energies, .9)) if len(energies) else 0
    if voiced > 1e-5:
        if energies[0] > voiced * .35:
            issues.append("voiced-start-edge-onset-needs-review")
        if energies[-1] > voiced * .35:
            issues.append("voiced-end-edge-tail-needs-review")
        if np.mean(energies < voiced * .02) > .75:
            warnings.append("mostly-silence-listen-before-admission")
    if row:
        if row.get("sample_rate") is not None and row["sample_rate"] != sr:
            issues.append("metadata-sample-rate-mismatch")
        if row.get("start_frame") is not None and row.get("end_frame") is not None:
            if row["end_frame"] - row["start_frame"] != frames:
                issues.append("metadata-frame-count-mismatch")
        if row.get("start_time") is not None and row.get("end_time") is not None:
            if abs((row["end_time"] - row["start_time"]) / 1000 - duration) > .005:
                issues.append("metadata-duration-mismatch")
        if row.get("audio_sha256") and row["audio_sha256"] != file_hash(path):
            issues.append("metadata-audio-hash-mismatch")
    return mono, sr, {"passed": not issues, "issues": issues, "warnings": warnings,
        "sample_rate": sr, "channels": channels, "sample_width_bytes": width, "frames": frames,
        "duration": duration, "peak": peak, "rms": rms, "dc_offset": dc, "clipped_fraction": clipped,
        "first_frame_rms": float(energies[0]) if len(energies) else None,
        "last_frame_rms": float(energies[-1]) if len(energies) else None}


def classify_content(check, surah, ayah, canonical):
    if check["passed"]:
        return "exact-text-match-needs-listening-review"
    reference, observed = check["expected"].split(), check["observed"].split()
    if not observed:
        return "empty-recognizer-transcript"
    if len(observed) < len(reference) and any(reference[i:i + len(observed)] == observed for i in range(len(reference) - len(observed) + 1)):
        return "partial-ayah-missing-words"
    alternatives = [a for (s, a), text in canonical.items() if s == surah and a != ayah and text == check["observed"]]
    if alternatives:
        return "matches-another-ayah:" + ",".join(map(str, alternatives))
    for first in range(max(1, ayah - 2), ayah + 2):
        for count in (2, 3):
            texts = [canonical.get((surah, a)) for a in range(first, first + count)]
            if all(texts) and " ".join(texts) == check["observed"]:
                return "merged-ayahs-or-leakage"
    if len(reference) == len(observed):
        return "same-word-count-recognition-or-reading-disagreement"
    return "missing-extra-or-substituted-words"


def coverage_check(canonical, rows, scope="fatiha-juz-amma"):
    if scope == "fatiha-juz-amma":
        surahs = {1, *range(78, 115)}
    elif scope == "represented-surahs":
        surahs = {r["surah"] for r in rows}
    elif scope == "whole-quran":
        surahs = set(range(1, 115))
    else:
        raise ValueError("Unknown expected coverage scope")
    expected = {key for key in canonical if key[0] in surahs}
    ids = Counter((r["surah"], r["ayah"]) for r in rows)
    missing = sorted(expected - ids.keys())
    unexpected = sorted(ids.keys() - expected)
    duplicates = sorted(key for key, count in ids.items() if count != 1)
    return {"expected_scope": scope, "expected_row_count": len(expected),
            "missing_reference_ayahs": [list(k) for k in missing], "unexpected_reference_ayahs": [list(k) for k in unexpected],
            "duplicate_reference_ayahs": [list(k) for k in duplicates],
            "reference_coverage_complete": bool(expected) and not missing and not unexpected and not duplicates and len(rows) == len(expected)}


def metadata_schema_issues(row):
    issues = []
    for name in ("surah", "ayah"):
        if type(row.get(name)) is not int or row[name] < 1:
            issues.append(f"invalid-{name}-integer-ID")
    for name in ("audio_filename", "relative_audio_path", "text", "text_asr_normalized", "text_raw_uthmani", "reciter", "normalized_with_harakat"):
        if not isinstance(row.get(name), str) or not row[name].strip():
            issues.append(f"missing-or-invalid-schema-field:{name}")
    if not isinstance(row.get("source_ayahs"), list) or not row["source_ayahs"] or any(type(a) is not int or a < 1 for a in row["source_ayahs"]):
        issues.append("invalid-source-ayah-list")
    if type(row.get("surah")) is int and type(row.get("ayah")) is int:
        filename = f"{row['surah']:03d}{row['ayah']:03d}.wav"
        if row.get("audio_filename") != filename or row.get("relative_audio_path") != f"audio/{filename}":
            issues.append("canonical-ID-audio-filename-mismatch")
    return issues


def valid_previous_terminal_evidence(row, canonical):
    evidence = row.get("previous_terminal_evidence") or {}
    previous = row.get("previous_boundary_ayah")
    check, event = evidence.get("exact_check", {}), evidence.get("terminal_boundary", {})
    expected = canonical.get((row["surah"], previous))
    if not (evidence.get("guard_version") == VERSION and evidence.get("ayah") == previous
            and evidence.get("source_sha256") == row.get("source_sha256") and expected
            and check.get("passed") is True and check.get("expected") == expected and check.get("observed") == expected
            and evidence.get("cleaned_exact_passed") is True and event.get("resolved") is True
            and evidence.get("sample_rate") == 16000 and evidence.get("end_frame") == row.get("start_frame")
            and abs(event.get("left_end", -1) * 16000 - evidence["end_frame"]) <= 1):
        return False
    try:
        path = Path(evidence["raw_audio_path"])
        if file_hash(path) != evidence.get("raw_audio_sha256"):
            return False
        with wave.open(str(path), "rb") as wav:
            return (wav.getframerate() == 16000 and wav.getnchannels() == 1 and wav.getsampwidth() == 2
                    and wav.getnframes() == evidence["end_frame"] - evidence["start_frame"])
    except (OSError, ValueError, KeyError, wave.Error):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--compute-type", default="int8_float16")
    parser.add_argument("--surahs", help="Comma-separated subset; omit for every metadata row")
    parser.add_argument("--limit", type=int, default=0, help="0 audits all rows; partial audits never certify a dataset")
    parser.add_argument("--expected-reciter", help="Expected reciter_key; speaker identity still requires provenance/listening")
    parser.add_argument("--technical-only", action="store_true", help="No ASR; cannot pass the content/training gate")
    parser.add_argument("--expected-scope", choices=["fatiha-juz-amma", "represented-surahs", "whole-quran"], default="fatiha-juz-amma",
                        help="Default expects ALL 569 product-scope IDs, including entirely absent surahs")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit cannot be negative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Choose a NEW empty output directory; existing audit/reviewer files are not overwritten")
    root = args.dataset.resolve()
    manifest = root / "metadata.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    intervals, interval_issues = {}, {}
    for row in rows:
        key = row["surah"], row["ayah"]
        if row.get("start_frame") is not None and row.get("end_frame") is not None:
            sr = row.get("sample_rate", 16000)
            start, end = row["start_frame"] / sr, row["end_frame"] / sr
        elif row.get("start_time") is not None and row.get("end_time") is not None:
            start, end = row["start_time"] / 1000, row["end_time"] / 1000
        else:
            continue  # No inferred timestamps for metadata that has none.
        if start < 0 or end <= start:
            interval_issues.setdefault(key, []).append("invalid-source-interval")
        group = row["surah"], row.get("source_sha256", "metadata-recording-unspecified")
        intervals.setdefault(group, []).append((start, end, key))
    for group in intervals.values():
        previous = None
        for current in sorted(group):
            if previous and current[0] < previous[1] - 1e-8:
                for item in (previous, current):
                    interval_issues.setdefault(item[2], []).append("overlapping-source-intervals")
            if previous is None or current[1] > previous[1]:
                previous = current
    canonical = {key: normalize_quran_for_asr(row["raw"]) for key, row in load_quran().items()}
    selection = {int(s) for s in args.surahs.split(",")} if args.surahs else None
    selected = [r for r in rows if selection is None or r["surah"] in selection]
    if args.limit:
        selected = selected[:args.limit]
    model = None
    if not args.technical_only:
        import torch
        from faster_whisper import WhisperModel
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = WhisperModel(args.model, device=device, compute_type=args.compute_type if device == "cuda" else "int8")
    results, hashes, ids = [], {}, Counter((r["surah"], r["ayah"]) for r in rows)
    by_id = {(r["surah"], r["ayah"]): r for r in rows}
    for index, row in enumerate(selected, 1):
        key = (row["surah"], row["ayah"])
        result = {"surah": key[0], "ayah": key[1], "relative_audio_path": row.get("relative_audio_path"),
                  "automatic_pass": False, "usable_for_training": False,
                  "issues": list(interval_issues.get(key, [])) + metadata_schema_issues(row)}
        try:
            relative = Path(row["relative_audio_path"])
            path = (root / relative).resolve()
            if relative.is_absolute() or root not in path.parents:
                raise ValueError("Audio path escapes dataset")
            digest = file_hash(path)
            result["audio_sha256"] = digest
            if ids[key] > 1:
                result["issues"].append("duplicate-ayah-ID")
            previous = row.get("previous_boundary_ayah")
            if previous is not None:
                predecessor = by_id.get((key[0], previous))
                if (not predecessor or not predecessor.get("clip_verification", {}).get("passed")) and not valid_previous_terminal_evidence(row, canonical):
                    result["issues"].append("shared-start-depends-on-missing-or-rejected-previous-ayah")
                elif predecessor and predecessor.get("end_frame") != row.get("start_frame"):
                    result["issues"].append("shared-boundary-frame-mismatch")
            if digest in hashes:
                result["issues"].append("duplicate-WAV-content")
                results[hashes[digest]]["issues"].append("duplicate-WAV-content")
                results[hashes[digest]]["automatic_pass"] = False
            else:
                hashes[digest] = len(results)
            expected = canonical.get(key)
            result["canonical_text"] = expected
            if expected is None or row.get("text_asr_normalized") != expected:
                result["issues"].append("label-does-not-match-canonical-Qaloon-JSON")
            if args.expected_reciter:
                declared = (row.get("reciter_key") or row.get("reciter", "")).lower()
                aliases = {"trabulsi": ("trabulsi", "tarabulsi", "طرابلس")}.get(args.expected_reciter, (args.expected_reciter,))
                if not any(alias in declared for alias in aliases):
                    result["issues"].append("declared-reciter-mismatch")
            audio, sr, technical = inspect_wav(path, row)
            result["technical"] = technical
            result["issues"].extend(technical["issues"])
            if model is not None and len(audio) and np.isfinite(audio).all():
                if sr != 16000:
                    import torch
                    from torchaudio.functional import resample
                    audio = resample(torch.from_numpy(audio), sr, 16000).numpy()
                words = transcribe_words(model, audio, vad_filter=False)
                check = exact_text_check(expected or "", " ".join(w["clean_word"] for w in words))
                result.update({"content": check, "words": words, "classification": classify_content(check, *key, canonical)})
                if not check["passed"]:
                    result["issues"].append("non-exact-blind-ASR-text")
            else:
                result["issues"].append("content-not-verified")
            result["automatic_pass"] = not result["issues"]
        except Exception as error:
            result["issues"].append(f"audit-error:{type(error).__name__}:{error}")
        results.append(result)
        print(f"{index}/{len(selected)} {key[0]:03d}:{key[1]:03d} {'PASS-CANDIDATE' if result['automatic_pass'] else 'REVIEW'} {','.join(result['issues'])}", flush=True)
    referenced = {str(Path(r["relative_audio_path"]).as_posix()) for r in rows if r.get("relative_audio_path")}
    orphans = sorted(str(f.relative_to(root).as_posix()) for f in (root / "audio").glob("*.wav") if str(f.relative_to(root).as_posix()) not in referenced)
    report = {"schema_version": 2, "dataset_root": str(root), "metadata_sha256": file_hash(manifest),
        "model": args.model if model else None, "strict_content_comparison": True, "technical_only": args.technical_only,
        "metadata_rows": len(rows), "audited_rows": len(results), "complete_audit": len(results) == len(rows),
        "automatic_pass_count": sum(r["automatic_pass"] for r in results), "orphan_WAVs": orphans,
        "dataset_automatic_pass": bool(results) and len(results) == len(rows) and not orphans and all(r["automatic_pass"] for r in results),
        "human_approved_clips": 0, "usable_for_training": False, "clips": results}
    report.update(coverage_check(canonical, rows, args.expected_scope))
    report["dataset_automatic_pass"] = report["dataset_automatic_pass"] and report["reference_coverage_complete"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "validation.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["surah", "ayah", "automatic_pass", "canonical_text", "observed", "issues"])
        writer.writeheader()
        for result in results:
            writer.writerow({k: result.get(k) for k in ("surah", "ayah", "automatic_pass", "canonical_text")}
                            | {"observed": result.get("content", {}).get("observed"), "issues": " | ".join(result["issues"])})
    with (args.output_dir / "listening_decisions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["surah", "ayah", "audio_sha256", "automatic_pass", "listening_approved", "boundary_approved", "reviewer", "reviewed_at", "notes"])
        writer.writeheader()
        for result in results:
            writer.writerow({k: result.get(k) for k in ("surah", "ayah", "audio_sha256", "automatic_pass")})
    print(f"{report['automatic_pass_count']}/{len(results)} automatic candidates; zero human approvals. {args.output_dir / 'validation.json'}", flush=True)
    if not report["dataset_automatic_pass"]:
        raise SystemExit(2)  # Expected QA failure, not a reason to weaken gates.


if __name__ == "__main__":
    main()
