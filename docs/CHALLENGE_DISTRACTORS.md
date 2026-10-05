# Challenge distractors: how wrong answers are chosen

The `/games` challenges show one right answer and two wrong ones. Wrong answers
that look nothing like the right one teach little: a learner can rule them out
without remembering the ayah. Rattil ranks every candidate by how close it is to
the right answer, then picks from a window of that ranking that depends on the
difficulty.

## The pieces

| File | What it does |
| --- | --- |
| `src/learning/qursim.py` | Loads QurSim and scores an embedding model on it. |
| `src/learning/benchmark_text_embeddings.py` | Compares Hugging Face models on QurSim → `src/learning/text_embedding_benchmark.json`. |
| `src/learning/train_ayah_embedder.py` | Fine-tunes the best model into Rattil's own ayah embedder → `runs/rattil_ayah_embed/` (Git-ignored). |
| `src/learning/build_text_embeddings.py` | Embeds all 6,210 Qālūn ayahs and 14,714 distinct words once → `web/public/quran/text-embeddings.json` (Git-ignored, ~8 MB). |
| `web/lib/challenges.ts` | Ranks candidates at request time. No model runs in the web server. |

Rebuild after changing the model or the Quran cache:

```bash
# Mac: prefix with PYTHONPATH=src/deployment/mac_shim (Anaconda's torchvision is broken)
python src/learning/benchmark_text_embeddings.py        # optional, ~15 min, downloads ~7 GB of models
python src/learning/train_ayah_embedder.py              # optional, a few minutes on Apple silicon
python src/learning/build_text_embeddings.py            # uses runs/rattil_ayah_embed if present
```

The first run of the web server builds the index by itself if it is missing
(`web/lib/first-run.ts`). Without the fine-tuned model it uses the base model from
Hugging Face. Without the index, challenges fall back to spelling, rhyme and length.

## Choosing the model

[QurSim](https://aclanthology.org/L12-1051/) holds 6,915 pairs of verses that Ibn
Kathir's tafsir links, labelled strongly related, related or not obvious. Each model
embeds all 4,114 verses, and we check whether a verse's strongly related partners
come up among its 10 nearest neighbours (recall@10) and how high the first one ranks
(MRR). Cohere's `embed-multilingual-v3` was not tested: it needs a paid API key and
would send the text out on every rebuild.

All numbers are in `src/learning/text_embedding_benchmark.json`. "All verses" uses every
QurSim verse as a query; "held-out, 256-d" uses only the 20% of verses the fine-tune
never sees, at the width the index stores.

| Model | Size | Recall@10, all verses | MRR, all verses | Recall@10, held-out, 256-d |
| --- | --- | --- | --- | --- |
| **Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2** | 135M | **0.283** | **0.237** | **0.286** |
| Omartificial-Intelligence-Space/GATE-AraBert-v1 | 135M | 0.257 | 0.223 | 0.266 |
| silma-ai/silma-embedding-matryoshka-v0.1 | 135M | 0.241 | 0.212 | 0.253 |
| intfloat/multilingual-e5-large-instruct | 560M | 0.240 | 0.213 | 0.225 |
| Adanmohh/wahi-quran-bge-m3-v3 (Quran-tuned) | 568M | 0.234 | 0.202 | 0.237 |
| BAAI/bge-m3 | 568M | 0.223 | 0.196 | 0.211 |
| Amer-Surur1/quran-finetuned-mpnet (Quran-tuned) | 278M | 0.185 | 0.170 | 0.213 |
| intfloat/multilingual-e5-base | 278M | 0.177 | 0.167 | 0.173 |

The smallest Arabic-only model wins, ahead of models four times its size and of the
two Quran-tuned models. It is also a Matryoshka model, so its first 256 dimensions
keep most of the quality; the index stores those, as int8.

## Rattil's own ayah embedder

`train_ayah_embedder.py` fine-tunes the winner on QurSim's strongly related pairs.
A hash of the verse number holds out 20% of verses: no training pair touches them,
and the scores below rank their partners among all 4,114 verses.

| Held-out verses (459 queries) | Recall@10, 768-d | MRR, 768-d | Recall@10, 256-d | MRR, 256-d |
| --- | --- | --- | --- | --- |
| Arabic-Triplet-Matryoshka-V2 | 0.308 | 0.252 | 0.286 | 0.236 |
| **rattil-ayah-embed** (3 epochs, 1,645 pairs) | **0.374** | **0.319** | **0.371** | **0.301** |

At the 256 dimensions the index stores, recall@10 rises by 30% and MRR by 28%
on verses the model never trained on. Training takes about 6 minutes on an M-series
Mac. The model and its card (`rattil_training.json`) stay in `runs/`, which is
Git-ignored, so a fresh clone builds the index from the base model until the
fine-tune is run or published.

## Ranking a candidate

For two ayahs (or two words), `closeness()` in `web/lib/challenges.ts` adds:

| Signal | Weight | Why |
| --- | --- | --- |
| Embedding cosine | 0.55 | A choice about the same subject is plausible. |
| Shared letter pairs | 0.20 | Mutashābihāt: ayahs that differ by a word or two. |
| Same last two letters | 0.15 | The fāṣila (rhyme) of a surah makes its ayahs sound alike. |
| Similar length | 0.10 | A much longer or shorter option stands out without being read. |

Per challenge:

- **What comes next?** Ranks every ayah against the right one. Outside "Same surah",
  ayahs from the target's surah get +0.1, since mixing up the order inside a surah is
  the most common memorisation slip. Never offered:
  - the ayah on screen, or a near copy of it (letter-pair overlap ≥ 0.85), such as
    `كلا سوف تعلمون` under `ثم كلا سوف تعلمون`;
  - what follows another copy of the ayah on screen. After a repeated ayah such as
    `ولا أنتم عابدون ما أعبد` (Al-Kāfirūn 3 and 5), both successors are right answers.
- **Find the surah.** A surah scores as its closest ayah to the prompt. An ayah whose
  text also occurs in another surah of the choice pool is never asked.
- **Listen & match.** Blends the MFCC recording index (when built) 50/50 with the text score.
- **Restore the word.** Ranks words by the same signals on letters only. A word that
  differs from the answer only by vowels or Qālūn marks is never offered, because the
  wasl vowel of `اَ۬لنَّاسِ` / `اِ۬لنَّاسِ` changes with the word before it. A word
  already visible elsewhere in the ayah is never offered for the gap.
- **Ayah sequence.** Hard picks three consecutive ayahs that resemble each other and
  swaps two neighbours; easy picks dissimilar ayahs and scrambles the whole order.

Difficulty picks from rank windows: hard from ranks 1–6, medium 7–20, easy 21–60
(surahs use half these windows, words one and a half times). Easy choices are
related but distinguishable, not random. Candidates are deduplicated by text before
ranking, so a refrain repeated 31 times in Ar-Raḥmān cannot fill a window by itself.

### Choices from

The **Choices from** setting (`from=` on `/api/challenges`) picks the pool that wrong
answers are drawn from. The reading scope still decides which questions are asked.

| Setting | Pool | Applies to |
| --- | --- | --- |
| Same surah | Ayahs (or words, or recordings) of the right answer's surah | Next, word, listening |
| Reading scope (default) | Fātiḥah + Juz ʿAmma, or the whole Quran if that scope is chosen | All but the sequence game |
| Whole Quran | All 6,210 ayahs, whatever the reading scope | Next, surah, word; listening only has clips for Juz ʿAmma |

A surah too short for two same-surah choices, such as Al-Kawthar, is topped up from
the reading scope, and the method note under the question says so. Adaptive
difficulty keeps a separate history per source, because "hard" means something
different in each.

## Limits

- Closeness is not meaning. Two ayahs can be close because they share words, not
  because tafsir links them. Nothing here judges recitation or tajweed.
- QurSim uses Hafs spelling and numbering; the index embeds the Qālūn ASR-normalised
  text, so spelling differences such as `فذالك` / `فذلك` may shift scores slightly.
- QurSim was built from one tafsir; "strongly related" reflects its links only.
