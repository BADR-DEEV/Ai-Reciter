"""Convert pinned Tarteel Quran-base WEIGHTS for blind timestamped segmentation.

The legacy Tarteel tokenizer omits timestamp IDs. Restore OpenAI's matching
token metadata only after checking every shared token ID; never swap weights.
Outputs are separate from trained/private deployments and never overwritten.
"""
import argparse
import json
from pathlib import Path

MODEL = "tarteel-ai/whisper-base-ar-quran"
REVISION = "5c3c53fdf9272c4f6ee0bee09a1e5a4a615ee25c"
TOKEN_MODEL = "openai/whisper-base"
TOKEN_REVISION = "e37978b90ca9030d5170a5c07aadb050351a65bb"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("models/segmentation/tarteel-whisper-base-ct2"))
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a new output directory; existing models are never overwritten")
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from ctranslate2.converters import TransformersConverter
    legacy = WhisperProcessor.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
    processor = WhisperProcessor.from_pretrained(TOKEN_MODEL, revision=TOKEN_REVISION, trust_remote_code=False)
    vocab = processor.tokenizer.get_vocab()
    # The historical Hebrew ISO alias iw was renamed he; both are ID 50279.
    # This metadata-only rename is explicit; every other token must match.
    aliases = {"<|iw|>": "<|he|>"}
    if any(vocab.get(aliases.get(token, token)) != index for token, index in legacy.tokenizer.get_vocab().items()):
        raise ValueError("Tokenizer IDs differ; refusing unsafe token-metadata restoration")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
    if model.config.vocab_size != len(vocab):
        raise ValueError("Restored timestamp vocabulary does not match Tarteel output layers")
    # Preserve original feature extractor, restore only the omitted token IDs.
    processor.feature_extractor = legacy.feature_extractor
    staging = args.output_dir.with_name(args.output_dir.name + "-hf")
    if staging.exists():
        parser.error("Conversion staging exists; choose a new output directory")
    staging.mkdir(parents=True)
    model.save_pretrained(staging)
    processor.save_pretrained(staging)
    del model
    TransformersConverter(str(staging), copy_files=["tokenizer.json", "preprocessor_config.json"]).convert(
        str(args.output_dir), quantization="float16")
    provenance = {"model": MODEL, "revision": REVISION, "weights": "original-tarteel-no-finetuning",
        "token_metadata_model": TOKEN_MODEL, "token_metadata_revision": TOKEN_REVISION,
        "shared_token_ids_verified": True, "token_aliases": aliases, "purpose": "blind-ASR-timestamps-unapproved-segmentation"}
    (args.output_dir / "segmentation_model.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Converted pinned Tarteel Quran-base: {args.output_dir}")


if __name__ == "__main__":
    main()
