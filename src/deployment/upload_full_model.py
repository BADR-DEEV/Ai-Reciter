"""Upload only the full fine-tuned model; leave visibility and other models unchanged."""
import argparse
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="BadrSh/qalon-reciter")
    parser.add_argument("--model-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "runs/gpu_base_full")
    args = parser.parse_args()
    folder = args.model_dir
    destination = "models/full-finetuning"
    filenames = [
        "model.safetensors", "config.json", "generation_config.json",
        "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "added_tokens.json", "vocab.json",
        "merges.txt", "normalizer.json", "metrics.json",
    ]
    for name in filenames:
        if not (folder / name).is_file():
            raise FileNotFoundError(folder / name)
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    test = metrics["test"]["overall"]
    readme = f"""# Full fine-tuned Qaloon Whisper Base

Complete model weights, not a LoRA adapter. Source: `runs/gpu_base_full`.
Saved test WER: **{test['wer'] * 100:.2f}%**; CER: **{test['cer'] * 100:.2f}%**
on {test['samples']} held-out clips. See `metrics.json` for evaluation details.
Training coverage: Fatiha and surahs 78–114; not a tajweed assessment model.
Existing repository licensing/provenance terms apply.

Install `torch transformers huggingface_hub`, then authenticate if private.

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration

repo = "{args.repo_id}"
folder = "{destination}"
processor = WhisperProcessor.from_pretrained(repo, subfolder=folder, token=True)
model = WhisperForConditionalGeneration.from_pretrained(repo, subfolder=folder, token=True)
```

Use the processor on mono 16 kHz audio and generate Arabic transcription.
Segment recordings longer than 30 seconds. No expected ayah prompt is required.
"""
    api = HfApi()
    info = api.model_info(args.repo_id)
    operations = [CommitOperationAdd(f"{destination}/{name}", folder / name)
                  for name in filenames]
    operations.append(CommitOperationAdd(f"{destination}/README.md", readme.encode("utf-8")))
    commit = api.create_commit(
        args.repo_id, operations=operations, parent_commit=info.sha,
        commit_message="Upload full fine-tuned Whisper Base with processor and metrics",
    )
    print(f"Uploaded: {commit.commit_url}")
    print("Repository visibility, root files and other model folders were not changed.")


if __name__ == "__main__":
    main()
