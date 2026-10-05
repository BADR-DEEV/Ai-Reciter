"""QurSim: Quran verse pairs related through Ibn Kathir's tafsir (Sharaf & Atwell, LREC 2012).

Used to choose and fine-tune the ayah embedding model behind challenge distractors.
Labels: 2 strongly related, 1 related, 0 not obvious. The deduplicated copy from
Alsaleh et al. (WANLP 2021) is downloaded on first use into the Git-ignored data/.
"""
import csv
import hashlib
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
URL = "https://raw.githubusercontent.com/AlsalehAbdullah/QurSim/main/QurSim_filtered.csv"
CACHE = ROOT / "data/qursim/QurSim_filtered.csv"


def load(path=CACHE):
    """Return ({verse_id: text}, [(id_a, id_b, label)])."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    verses, pairs = {}, []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            a, b = f"{row['SS']}:{row['SV']}", f"{row['TS']}:{row['TV']}"
            verses[a], verses[b] = row["Verse1"].strip(), row["Verse2"].strip()
            pairs.append((a, b, int(row["Label"])))
    return verses, pairs


def held_out(verse_id, percent=20):
    """Stable verse-level split: the same verse is always train or always test."""
    return int(hashlib.sha256(verse_id.encode()).hexdigest(), 16) % 100 < percent


def related(pairs):
    out = {}
    for a, b, label in pairs:
        if label == 2 and a != b:
            out.setdefault(a, set()).add(b)
            out.setdefault(b, set()).add(a)
    return out


def evaluate(encode, verses, pairs, queries=None, dims=(None,)):
    """Recall@10 and MRR of strongly related verses among all QurSim verses, per embedding width."""
    from scipy.stats import spearmanr
    ids = sorted(verses)
    index = {v: i for i, v in enumerate(ids)}
    full = np.asarray(encode([verses[i] for i in ids]), dtype=np.float32)
    links = related(pairs)
    queries = [q for q in (queries or links) if q in links]
    results = {}
    for dim in dims:
        emb = full[:, :dim] if dim else full
        emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
        sims = emb @ emb.T
        np.fill_diagonal(sims, -2)
        recall, reciprocal = [], []
        for q in queries:
            order = np.argsort(-sims[index[q]])[:200]
            hits = [k for k, j in enumerate(order) if ids[j] in links[q]]
            recall.append(sum(h < 10 for h in hits) / min(len(links[q]), 10))
            reciprocal.append(1 / (hits[0] + 1) if hits else 0.0)
        cos = [float(emb[index[a]] @ emb[index[b]]) for a, b, _ in pairs]
        results[str(dim or full.shape[1])] = {
            "recall@10": round(float(np.mean(recall)), 4), "mrr": round(float(np.mean(reciprocal)), 4),
            "spearman": round(float(spearmanr(cos, [label for *_, label in pairs]).correlation), 4), "queries": len(queries),
        }
    return results
