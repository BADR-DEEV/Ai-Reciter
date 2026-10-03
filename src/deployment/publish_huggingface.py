"""Publish two verified model variants and safely archive the existing Tiny adapter.

Default is a dry run. Pass --publish to create ONE atomic Hub commit. The
repository's visibility and LICENSE are never changed. Cached HF auth is used;
tokens, training checkpoints, optimizer state and recordings are not uploaded.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationCopy, CommitOperationDelete, hf_hub_download
import jiwer

ROOT = Path(__file__).resolve().parents[2]
REPO_ID = "BadrSh/qalon-reciter"
LEGACY = "legacy/whisper-tiny-lora"
VARIANTS = {
    "full-finetuning": {"run": "gpu_base_full", "weight": "model.safetensors", "kind": "full fine-tuning"},
    "base-lora-v2": {"run": "gpu_base_v2", "weight": "adapter_model.safetensors", "kind": "LoRA adapter"},
}
COMMON = {"added_tokens.json", "merges.txt", "metrics.json", "normalizer.json", "preprocessor_config.json",
          "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"}


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def verified_metrics():
    verified = {}
    cases = {}
    for variant, config in VARIANTS.items():
        folder = ROOT / "runs" / config["run"]
        report = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        verified[variant] = {}
        for split in ("validation", "test"):
            rows = [json.loads(line) for line in (folder / f"{split}_predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            refs = [row["reference"] for row in rows]
            hyps = [row["prediction"] for row in rows]
            scores = {"samples": len(rows), "wer": jiwer.wer(refs, hyps), "cer": jiwer.cer(refs, hyps)}
            for key in scores:
                if abs(scores[key] - report[split]["overall"][key]) > 1e-12:
                    raise ValueError(f"Saved/computed metric mismatch: {variant}/{split}/{key}")
            verified[variant][split] = scores
            keys = [(row["surah"], row["ayah"], row["reciter"], row["reference"]) for row in rows]
            if split in cases and keys != cases[split]:
                raise ValueError(f"The selected models have different {split} examples")
            cases[split] = keys
    # Confirm runner-up selection against the other comparable saved Base run.
    other = json.loads((ROOT / "runs" / "gpu_base" / "metrics.json").read_text(encoding="utf-8"))
    if not (verified["full-finetuning"]["test"]["wer"] < verified["base-lora-v2"]["test"]["wer"] < other["test"]["overall"]["wer"]):
        raise ValueError("Saved WER ranking changed; reselect models before publishing")
    return verified, {split: hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode("utf-8")).hexdigest()
                      for split, rows in cases.items()}


def percent(value):
    return f"{value * 100:.2f}%"


def model_card(repo, scores, parent, private):
    full, lora = scores["full-finetuning"], scores["base-lora-v2"]
    return f"""---
language:
- ar
license: other
license_name: qaloon-adapter-provenance-terms
license_link: LICENSE
library_name: transformers
base_model: openai/whisper-base
pipeline_tag: automatic-speech-recognition
tags:
- quran
- qaloon
- whisper
- lora
- full-fine-tuning
- research
---

# Qaloon Quran Reciter — two Whisper Base variants

**Recommended: `models/full-finetuning`.** It is a complete Whisper Base model.
**Runner-up: `models/base-lora-v2`.** It is a LoRA adapter and needs
`openai/whisper-base`; it is not a standalone full model.

Repository visibility: **{'private' if private else 'public'}** (unchanged).
This is a Qālūn ʿan Nāfiʿ research model for **al-Fātiḥah and surahs 78–114**,
not a trained full-Quran or tajweed assessment model.

## Current saved metrics

Recomputed from each run's saved reference/prediction files and verified against
its own `metrics.json`. These are **not new evaluations** and not model probabilities.
WER/CER below are percentages; JSON files store fractions (0–1). Lower is better.

| Variant | Test WER | Test CER | Validation WER | Validation CER |
| --- | ---: | ---: | ---: | ---: |
| Full fine-tuning (`gpu_base_full`) | **{percent(full['test']['wer'])}** | **{percent(full['test']['cer'])}** | {percent(full['validation']['wer'])} | {percent(full['validation']['cer'])} |
| Base LoRA v2 (`gpu_base_v2`, rank 32) | {percent(lora['test']['wer'])} | {percent(lora['test']['cer'])} | {percent(lora['validation']['wer'])} | {percent(lora['validation']['cer'])} |

Both models share the exact same saved **232 test** and **188 validation**
examples: Dokali, Husary, Huthaify, and Waleed, 58 test / 47 validation clips per
reciter. The dataset split used seed 42, with 1,851 training clips; four Husary
clips over 30 seconds were excluded. Fātiḥah is grouped across reciters into one
partition. Scores use the project's unvowelled Qālūn normalization and greedy
Arabic transcription; extra hamza-folded diagnostics are not substituted for
these primary scores. See each folder's `metrics.json` for per-reciter results.

The second-best comparable earlier Base adapter (`gpu_base`) had test
WER **33.23%**, CER **9.56%**; `gpu_base_v2` is the selected runner-up by saved WER.
This selection used existing test results for this release, not an untouched
future deployment benchmark.

## Quick start — one command per model

Install Python dependencies (use a CUDA-compatible Torch/Torchaudio pair for GPU):

```bash
pip install torch torchaudio "transformers==4.57.6" "peft==0.21.1" "huggingface_hub>=0.36,<1" soundfile numpy
```

If this private repository requires login:

```bash
python -c "from huggingface_hub import login; login()"
```

Download the small inference script:

```bash
python -c "from huggingface_hub import hf_hub_download; import shutil; shutil.copyfile(hf_hub_download('{repo}', 'examples/transcribe.py'), 'transcribe.py')"
```

Run either model; needed files are downloaded and cached automatically:

```bash
python transcribe.py clip.wav --variant full-finetuning
python transcribe.py clip.wav --variant base-lora-v2
```

Use a WAV clip of **at most 30 seconds**. Stereo and other sample rates are
converted to mono 16 kHz automatically. GPU is used when available; add
`--device cpu` to force CPU. Longer recordings must be segmented first, not
silently truncated. No expected ayah is injected into the decoder.

For reproducibility, add `--revision <COMMIT_HASH>` (the commit for this release
is visible in the Hub commit history); `main` may change later.

## Download only the model you want

```python
from pathlib import Path
from huggingface_hub import snapshot_download

repo = "{repo}"
variant = "full-finetuning"  # or "base-lora-v2"
snapshot = snapshot_download(repo, allow_patterns=[f"models/{{variant}}/*"])
model_folder = Path(snapshot) / "models" / variant
print(model_folder)
```

### Load the full model in your own application

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained(str(model_folder))
model = WhisperForConditionalGeneration.from_pretrained(str(model_folder)).eval()
```

### Load the second-best Base adapter

Use `variant = "base-lora-v2"` in the download block above, then:

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

processor = WhisperProcessor.from_pretrained(str(model_folder))
base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
model = PeftModel.from_pretrained(base, str(model_folder)).eval()
```

The ready-to-run script handles audio loading, feature extraction and generation
for either format. Tokenizer/processor assets are included in each model folder.

## Existing model preserved

The old root **Whisper Tiny LoRA adapter** was preserved, not deleted, in
`legacy/whisper-tiny-lora/`, including its original README, metrics and provenance
LICENSE. The pre-reorganization revision is `{parent}` and still loads using
the old root layout if pinned to that commit.

Its saved test WER/CER is **64.37% / 22.73%**, validation **73.08% / 25.32%**,
on a different **three-reciter, 180-test / 141-validation** split. Do **not**
compare those figures directly to the four-reciter Base results or attach the
full model's metrics to this adapter. Load the archived adapter with:

```bash
python transcribe.py clip.wav --variant legacy-tiny-lora
```

## Layout

```text
models/full-finetuning/       # Complete weights + processor + original metrics
models/base-lora-v2/          # Base adapter + processor + original metrics
legacy/whisper-tiny-lora/     # Previously published model, preserved
examples/transcribe.py       # One quick inference command for all variants
metrics.json                 # Verified overall metrics for the new variants
release.json                 # File SHA256s, split hashes and source revision
LICENSE                      # Existing provenance terms, unchanged
```

## Limits and provenance

- These metrics measure transcription on held-out ayah clips from known reciter
  voices, not generalization to unseen speakers or live student omission accuracy.
- False omission alerts, repetitions and substitutions are possible. This is a
  recognition aid, not a pronunciation/tajweed judgment or teacher replacement.
- Whisper's upstream base weights have their published Apache-2.0 terms. Existing
  repository `LICENSE` provenance restrictions remain unchanged; source-recording
  permissions and new derived-weight redistribution need their own review. No
  blanket public redistribution permission is inferred from this private release.
- No recordings, dataset rows, prediction transcripts, optimizer state, training
  checkpoints, tokens, signed URLs or credentials are included in this upload.

Project: https://github.com/BADR-DEEV/Ai-Reciter
"""


def publish(repo_id=REPO_ID, do_publish=False):
    api = HfApi()
    info = api.model_info(repo_id, files_metadata=True)
    original = {file.rfilename: file for file in info.siblings}
    scores, case_hashes = verified_metrics()
    operations = []
    archives = []
    if "adapter_config.json" in original:
        config = json.loads(Path(hf_hub_download(repo_id, "adapter_config.json", revision=info.sha)).read_text(encoding="utf-8"))
        if config.get("base_model_name_or_path") != "openai/whisper-tiny":
            raise ValueError("Unexpected root model identity; refusing to move it as the Tiny adapter")
        for path in sorted(original):
            if "/" in path or path == ".gitattributes":
                continue
            destination = f"{LEGACY}/{path}"
            if destination in original:
                raise ValueError(f"Archive target already exists: {destination}; refusing to overwrite")
            operations.append(CommitOperationCopy(path, destination, src_revision=info.sha))
            archives.append((path, destination))
            if path not in {"README.md", "LICENSE", "metrics.json"}:
                operations.append(CommitOperationDelete(path))
    elif "models/full-finetuning/model.safetensors" not in original:
        raise ValueError("Unexpected repository layout; inspect it before publishing")

    hashes = {}
    sizes = {}
    for variant, config in VARIANTS.items():
        folder = ROOT / "runs" / config["run"]
        filenames = COMMON | {config["weight"]}
        filenames |= {"config.json", "generation_config.json"} if variant == "full-finetuning" else {"adapter_config.json"}
        for name in sorted(filenames):
            path = folder / name
            if not path.is_file():
                raise FileNotFoundError(path)
            remote = f"models/{variant}/{name}"
            operations.append(CommitOperationAdd(remote, path))
            hashes[remote], sizes[remote] = digest(path), path.stat().st_size
    root_readme = model_card(repo_id, scores, info.sha, info.private)
    metadata = {"verified_at": datetime.now(timezone.utc).isoformat(), "metric_units": "fractions (0-1)",
                "verified_from_saved_predictions": True, "models": scores}
    release = {"repository": repo_id, "previous_revision": info.sha, "private": info.private,
               "variants": VARIANTS, "metrics": metadata, "evaluation_case_sha256": case_hashes,
               "artifact_sha256": hashes, "artifact_bytes": sizes,
               "archived_files": [{"source": source, "destination": dest} for source, dest in archives]}
    documents = {
        "README.md": root_readme,
        "metrics.json": json.dumps(metadata, indent=2) + "\n",
        "release.json": json.dumps(release, indent=2) + "\n",
        "legacy/README.md": f"# Archived original model\n\nThe original Whisper Tiny LoRA model is preserved under `whisper-tiny-lora/`.\nIts original README describes the former root layout; use the new root inference\nscript with `--variant legacy-tiny-lora`, or pin the original revision\n`{info.sha}` to use the old layout. Its metrics use a different three-reciter\nsplit and are not directly comparable to the new Base models.\n",
    }
    for variant, config in VARIANTS.items():
        metric = scores[variant]
        documents[f"models/{variant}/README.md"] = (
            f"# Qaloon Whisper Base — {config['kind']}\n\n"
            f"Source run: `{config['run']}`.\n\n"
            f"Test: **WER {percent(metric['test']['wer'])}, CER {percent(metric['test']['cer'])}** ({metric['test']['samples']} clips).\n"
            f"Validation: WER {percent(metric['validation']['wer'])}, CER {percent(metric['validation']['cer'])} ({metric['validation']['samples']} clips).\n\n"
            f"`python transcribe.py clip.wav --variant {variant}`\n\n"
            + ("This folder contains a complete model; no PEFT adapter is needed.\n" if variant == "full-finetuning"
               else "This is a LoRA adapter (rank 32), not full model weights. Load it with PEFT on `openai/whisper-base`.\n")
            + "\nSee the repository root README for installation, downloads, licensing/provenance and limitations.\n"
        )
    for path, text in documents.items():
        operations.append(CommitOperationAdd(path, text.encode("utf-8")))
    operations.append(CommitOperationAdd("examples/transcribe.py", Path(__file__).with_name("transcribe_hub.py")))
    operations.append(CommitOperationAdd("examples/requirements.txt", b'torch\ntorchaudio\ntransformers==4.57.6\npeft==0.21.1\nhuggingface_hub>=0.36,<1\nsoundfile\nnumpy\n'))
    print(json.dumps({"repository": repo_id, "private": info.private, "parent_revision": info.sha,
                      "new_model_bytes": sum(sizes.values()), "archive_files": archives,
                      "verified_metrics": scores, "publish": do_publish}, indent=2), flush=True)
    if not do_publish:
        print("Dry run only. Pass --publish to upload and commit.", flush=True)
        return
    commit = api.create_commit(repo_id, operations, parent_commit=info.sha,
                               commit_message="Publish full Whisper Base and runner-up LoRA v2; archive original Tiny model",
                               commit_description="Verified saved WER/CER and matching four-reciter evaluation cases. Preserve existing model and LICENSE; keep repository visibility unchanged.")
    final = api.model_info(repo_id, revision=commit.oid, files_metadata=True)
    files = {file.rfilename: file for file in final.siblings}
    if final.private != info.private:
        raise RuntimeError("Repository visibility unexpectedly changed")
    for variant, config in VARIANTS.items():
        path = f"models/{variant}/{config['weight']}"
        file = files[path]
        if file.size != sizes[path] or file.lfs is None or file.lfs.sha256 != hashes[path]:
            raise RuntimeError(f"Uploaded weight verification failed: {path}")
    for source, destination in archives:
        archived, old = files[destination], original[source]
        same = (archived.lfs is not None and archived.lfs.sha256 == old.lfs.sha256 and archived.size == old.size) if old.lfs else archived.blob_id == old.blob_id
        if not same:
            raise RuntimeError(f"Archived bytes differ from the original: {source}")
    print(f"Published and verified: {commit.commit_url}", flush=True)
    receipt = ROOT / "src" / "deployment" / "huggingface_release.json"
    receipt.write_text(json.dumps({"repo_id": repo_id, "commit": commit.oid, "url": commit.commit_url,
                                   "metrics": metadata, "artifact_sha256": hashes}, indent=2) + "\n", encoding="utf-8")
    print(f"Local receipt: {receipt.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    publish(args.repo_id, args.publish)
