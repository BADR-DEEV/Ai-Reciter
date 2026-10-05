"""Embed every Qaloon ayah and word once for difficulty-aware challenge distractors.

The web app ranks wrong answers by these vectors plus spelling, rhyme and length
(web/lib/challenges.ts). Semantic closeness makes a choice plausible; it says
nothing about tafsir, correctness or tajweed. No Quran text is generated.
"""
import argparse
import base64
import json
import re
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SURAHS = ROOT / "web/public/quran/surahs"
OUTPUT = ROOT / "web/public/quran/text-embeddings.json"
# Best of eight Hugging Face models on QurSim; see benchmark_text_embeddings.py.
BASE_MODEL = "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2"
LOCAL_MODEL = ROOT / "runs/rattil_ayah_embed"
DIM = 256


def word_key(word):
    """Group spellings that read the same letters: strip vowels, Qaloon marks and tatweel."""
    letters = "".join(c for c in unicodedata.normalize("NFD", word) if unicodedata.category(c) != "Mn")
    return letters.replace("ـ", "")


def corpus(root=SURAHS):
    """Ayahs exactly as web/lib/challenges.ts corpusVerses() displays them."""
    ayahs = []
    for number in range(1, 115):
        for ayah in json.loads((root / f"{number:03d}.json").read_text(encoding="utf-8"))["ayahs"]:
            display = ayah.get("displayText") or re.sub(r"[\d٠-٩]+", "", ayah["text"]).strip()
            ayahs.append({"id": f"{number}:{ayah['ayah']}", "display": display, "normalized": ayah["normalized"]})
    return ayahs


def vocabulary(ayahs):
    """Distinct letter sequences in reading order; the web app maps a word to its key with wordKey()."""
    keys = {}
    for ayah in ayahs:
        for word in ayah["display"].split():
            keys.setdefault(word_key(word), None)
    return [key for key in keys if key]


def quantize(vectors):
    """Per-row int8 with a float scale: 4x smaller than float32, ranking unchanged in practice."""
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
    scales = np.maximum(np.abs(vectors).max(axis=1), 1e-9) / 127
    values = np.clip(np.rint(vectors / scales[:, None]), -127, 127).astype(np.int8)
    return {"scales": [round(float(s), 8) for s in scales], "vectors": base64.b64encode(values.tobytes()).decode("ascii")}


def embed(encoder, texts, dim):
    vectors = np.asarray(encoder(texts), dtype=np.float32)
    if vectors.shape[1] < dim:
        raise ValueError(f"Model returns {vectors.shape[1]} dimensions; need at least {dim}")
    # Matryoshka models keep most quality in the leading dimensions.
    return vectors[:, :dim]


def build(encoder, model_name, revision, output=OUTPUT, dim=DIM, root=SURAHS, benchmark=None, base_model=None):
    ayahs = corpus(root)
    keys = vocabulary(ayahs)
    ayah_vectors = embed(encoder, [a["normalized"] for a in ayahs], dim)
    word_vectors = embed(encoder, keys, dim)
    short = model_name.split("/")[-1]
    label = f"{short}, fine-tuned from {base_model.split('/')[-1]}" if base_model else short
    result = {
        "version": 1, "model": model_name, "base_model": base_model, "revision": revision, "dim": dim, "label": label,
        "method": f"{label}: sentence embeddings ({dim}-d Matryoshka, int8), cosine",
        "warning": "Semantic closeness only. It makes wrong answers plausible; it is not tafsir, correctness or tajweed.",
        "benchmark": benchmark,
        "ayahs": {"ids": [a["id"] for a in ayahs], **quantize(ayah_vectors)},
        "words": {"keys": keys, **quantize(word_vectors)},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    temp.replace(output)
    print(f"Embedded {len(ayahs)} ayahs and {len(keys)} words with {model_name}: {output}")


def load(name):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(name, device="cpu")
    model.max_seq_length = min(model.max_seq_length or 256, 256)
    revision = None
    if Path(name).exists():
        card = Path(name) / "rattil_training.json"
        revision = json.loads(card.read_text())["base_revision"] if card.exists() else "local"
    else:
        from huggingface_hub import model_info
        try:
            revision = model_info(name).sha
        except Exception:
            revision = "unknown"
    encoder = lambda texts: model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    return encoder, revision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help=f"Hugging Face id or local folder (default: {LOCAL_MODEL.relative_to(ROOT)} if trained, else {BASE_MODEL})")
    parser.add_argument("--dim", type=int, default=DIM)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    name = args.model or (str(LOCAL_MODEL) if (LOCAL_MODEL / "config.json").exists() else BASE_MODEL)
    card = Path(name) / "rattil_training.json"
    training = json.loads(card.read_text()) if card.exists() else {}
    encoder, revision = load(name)
    build(encoder, "rattil-ayah-embed" if training else name, revision, args.output, args.dim,
          benchmark=training.get("benchmark"), base_model=training.get("base_model"))
