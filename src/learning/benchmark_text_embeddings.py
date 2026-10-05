"""Compare Hugging Face embedding models on QurSim to pick the distractor model.

Writes src/learning/text_embedding_benchmark.json. "all" ranks the strongly related
verses of every QurSim verse; "held_out" uses only the 20% of verses that
train_ayah_embedder.py never trains on, so a fine-tuned model is comparable there.
"""
import argparse
import json
import time
from pathlib import Path

try:
    from .qursim import evaluate, load
    from .train_ayah_embedder import split
except ImportError:  # run as a script
    from qursim import evaluate, load
    from train_ayah_embedder import split

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).with_name("text_embedding_benchmark.json")
MODELS = [
    "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2",
    "Omartificial-Intelligence-Space/GATE-AraBert-v1",
    "silma-ai/silma-embedding-matryoshka-v0.1",
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-large-instruct",
    "Adanmohh/wahi-quran-bge-m3-v3",
    "Amer-Surur1/quran-finetuned-mpnet",
]
PREFIX = {"intfloat/multilingual-e5-base": "query: ", "intfloat/multilingual-e5-large-instruct": "query: "}
# Saved by a newer sentence-transformers than 5.1; rebuilt from transformer + pooling.
POOLING = {"Adanmohh/wahi-quran-bge-m3-v3": "cls", "Amer-Surur1/quran-finetuned-mpnet": "mean"}


def load_model(name):
    from sentence_transformers import SentenceTransformer, models
    if name in POOLING:
        body = models.Transformer(name, max_seq_length=256)
        return SentenceTransformer(modules=[body, models.Pooling(body.get_word_embedding_dimension(), POOLING[name])])
    model = SentenceTransformer(name)
    model.max_seq_length = min(model.max_seq_length or 256, 256)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", default=MODELS + [str(ROOT / "runs/rattil_ayah_embed")])
    args = parser.parse_args()

    verses, pairs = load()
    _, test_pairs, queries = split(pairs)
    results = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}
    for name in args.models:
        local = Path(name).exists()
        if name.startswith(str(ROOT)) and not local:
            continue  # not trained on this machine
        started = time.time()
        model = load_model(name)
        encode = lambda texts: model.encode([PREFIX.get(name, "") + t for t in texts], batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
        label = Path(name).name if local else name
        # Training pairs inflate "all" for the fine-tuned model; compare it on "held_out".
        results[label] = {"all": evaluate(encode, verses, pairs), "held_out": evaluate(encode, verses, test_pairs, queries, dims=(None, 256)),
                          "seconds": round(time.time() - started, 1)}
        print(label, json.dumps(results[label]), flush=True)
        OUTPUT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
