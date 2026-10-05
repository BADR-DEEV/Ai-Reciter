"""Fine-tune Rattil's own ayah embedder on QurSim and check it on held-out verses.

Starts from the best Hugging Face model in benchmark_text_embeddings.py and learns
which verses the tafsir links (strongly related pairs). 20% of verses never appear
in training; the before/after scores on them go into rattil_training.json, and
build_text_embeddings.py picks the model up from runs/rattil_ayah_embed/.
"""
import argparse
import json
import random
from pathlib import Path

try:
    from .qursim import evaluate, held_out, load, related
except ImportError:  # run as a script
    from qursim import evaluate, held_out, load, related

ROOT = Path(__file__).resolve().parents[2]
BASE = "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2"
OUTPUT = ROOT / "runs/rattil_ayah_embed"
DIMS = [768, 512, 256, 128, 64]


def split(pairs):
    """Train only on pairs that touch no held-out verse; test on held-out verses' links."""
    train = [(a, b) for a, b, label in pairs if label == 2 and a != b and not held_out(a) and not held_out(b)]
    test_pairs = [p for p in pairs if held_out(p[0]) or held_out(p[1])]
    queries = sorted(q for q in related(test_pairs) if held_out(q))
    return train, test_pairs, queries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    from datasets import Dataset
    from huggingface_hub import model_info
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
    from sentence_transformers.training_args import BatchSamplers

    random.seed(args.seed)
    verses, pairs = load()
    train, test_pairs, queries = split(pairs)
    model = SentenceTransformer(args.base)
    model.max_seq_length = 256
    encode = lambda texts: model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    before = evaluate(encode, verses, test_pairs, queries, dims=(None, 256))
    print("held-out before:", json.dumps(before))

    rows = train + [(b, a) for a, b in train]  # relatedness is symmetric
    random.shuffle(rows)
    dataset = Dataset.from_dict({"anchor": [verses[a] for a, _ in rows], "positive": [verses[b] for _, b in rows]})
    loss = losses.MatryoshkaLoss(model, losses.MultipleNegativesRankingLoss(model), DIMS)
    checkpoints = ROOT / "runs/.rattil_ayah_embed_checkpoints"
    trainer = SentenceTransformerTrainer(model=model, train_dataset=dataset, loss=loss, args=SentenceTransformerTrainingArguments(
        output_dir=str(checkpoints), num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr, warmup_ratio=0.1, seed=args.seed, save_strategy="no", logging_steps=25, report_to=[],
        # Many pairs share a verse; a duplicate in the batch would be a false negative.
        batch_sampler=BatchSamplers.NO_DUPLICATES))
    trainer.train()

    after = evaluate(encode, verses, test_pairs, queries, dims=(None, 256))
    print("held-out after:", json.dumps(after))
    args.output.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    try:
        revision = model_info(args.base).sha
    except Exception:
        revision = "unknown"
    card = {
        "base_model": args.base, "base_revision": revision,
        "data": "QurSim strongly related pairs (label 2), deduplicated copy by Alsaleh et al. 2021",
        "split": "verse-level sha256 hash, 20% held out; training pairs never touch a held-out verse",
        "train_pairs": len(train), "held_out_queries": len(queries),
        "loss": f"MatryoshkaLoss({DIMS}) over MultipleNegativesRankingLoss",
        "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.lr, "seed": args.seed,
        "benchmark": {"held_out_before": before, "held_out_after": after},
    }
    (args.output / "rattil_training.json").write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
