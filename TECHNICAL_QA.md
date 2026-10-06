# Rattil — Technical Questions and Presentation Answers

## The answer to memorize

> We fine-tuned an Arabic sentence-embedding model on related Quran verse pairs from QurSim. We used contrastive learning with Multiple Negatives Ranking Loss, wrapped in Matryoshka Loss so the embeddings also work at smaller dimensions. We then embedded the Qālūn ayahs and words, stored 256-dimensional int8 vectors, and ranked quiz distractors using cosine similarity combined with spelling, rhyme, and length. We evaluated the embeddings using held-out verses and before-and-after retrieval metrics.

## 1. Which embedding model did you use?

**Answer:** We started with **`Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2`**, a **135M-parameter Arabic text-embedding model**. Our fine-tuned version is **`Mathani-Ayat/rattil-ayah-embed`**.

We used the **Sentence Transformers** framework. This is a text encoder, separate from the two Whisper audio models.

## 2. Why did you choose that model?

**Answer:** We compared eight embedding models on QurSim. This Arabic model performed best in the recorded comparison, including against larger multilingual models.

On the all-verses benchmark, its **Recall@10 was 0.283** and **MRR was 0.237**. Its Matryoshka design also lets us keep useful embeddings at a smaller size.

Models benchmarked included Arabic GATE-AraBert, SILMA, multilingual E5, BGE-M3, and Quran-tuned embedding models. Benchmarking them does not mean we deployed or fine-tuned all of them.

## 3. What data did you fine-tune it on?

**Answer:** **QurSim**, a dataset of Quran verse pairs linked through Ibn Kathir's tafsir. The project uses a deduplicated copy.

- Dataset labels: `2` = strongly related, `1` = related, `0` = not obvious.
- Training uses only **strongly related pairs**, with different verse IDs.
- The documented run used **1,645 training pairs**.
- Each pair is also reversed because relatedness is symmetric: `(A, B)` and `(B, A)`.

We trained on verse relationships, not learner audio or manually labelled pronunciation mistakes.

## 4. What exactly did you embed for the application?

**Answer:** We embedded **6,210 Qālūn ayahs** and **14,714 distinct word keys**, according to the documented index build.

- Ayahs use the cached **ASR-normalized Qālūn text**.
- Word keys remove diacritics, Qālūn combining marks, and tatweel.
- Ayah IDs such as `78:1` connect each vector to its original text.

The model was fine-tuned on verse pairs. Encoding individual words is a downstream use of that same encoder, not a separate word-model training run.

## 5. What is an embedding?

**Answer:** An embedding is a list of numbers representing a piece of text. Similar texts should have vectors pointing in similar directions, allowing us to compare them numerically.

It is not generated Quran text, a tafsir explanation, or a correctness score.

## 6. Which training techniques did you use?

**Answer:** Three main techniques:

1. **Transfer learning:** adapt a pretrained Arabic encoder instead of training from scratch.
2. **Contrastive learning:** make related verses score higher than other verses in the batch.
3. **Matryoshka representation learning:** train the leading dimensions to remain useful when the vector is shortened.

The exact loss in the code is:

```python
MatryoshkaLoss(
    model,
    MultipleNegativesRankingLoss(model),
    [768, 512, 256, 128, 64],
)
```

Despite the base model's name containing “Triplet,” our fine-tuning script uses **Multiple Negatives Ranking Loss**, not an explicit triplet-loss implementation.

## 7. How does Multiple Negatives Ranking Loss work?

**Answer:** Each training example has an anchor verse and a related positive verse. The model should rank that positive above other positives in the same batch, which act as **in-batch negatives**.

For example, in a batch with `(A, B)` and `(C, D)`, the model learns that B is the intended match for A, while D acts as another candidate.

We use a **NO_DUPLICATES batch sampler** to avoid repeated texts in a batch becoming obvious false negatives. It does not eliminate every possible unlabelled semantic relationship between verses.

## 8. What is Matryoshka Loss, and why use it?

**Answer:** It trains embeddings at multiple nested sizes: **768, 512, 256, 128, and 64 dimensions**. This encourages the first dimensions to preserve useful information.

We store only **256 dimensions** in the application, reducing storage and similarity-computation cost. This is prefix truncation supported by training, not PCA.

## 9. What were the embedding training settings?

**Answer:** The training script defaults are:

| Setting | Value |
| --- | --- |
| Epochs | 3 |
| Batch size | 32 |
| Learning rate | `2e-5` |
| Warmup ratio | 0.1 |
| Maximum sequence length | 256 tokens |
| Random seed | 13 |
| Loss | Matryoshka Loss over Multiple Negatives Ranking Loss |

The script records the actual run configuration and before/after results in `rattil_training.json`.

## 10. How did you avoid training/test leakage?

**Answer:** We held out **20% of verse IDs using a stable SHA-256 hash split**. A training pair is excluded if either verse belongs to the held-out set.

This is stronger than randomly splitting pairs, because the same verse could otherwise appear in both training and test through different pairs. Evaluation ranks held-out queries against all QurSim verses.

This is a verse-level holdout; it does not guarantee that every theme or similar phrase is unseen.

## 11. How did you compare two embeddings?

**Answer:** Using **cosine similarity**:

```text
cosine(A, B) = dot(A, B) / (norm(A) × norm(B))
```

After L2 normalization, this becomes a dot product. We truncate to 256 dimensions and normalize before quantization. The web implementation computes an approximate dot product using int8 values and their stored scales.

A higher similarity is a ranking signal, not a probability that two ayahs have identical meanings.

## 12. How did you rank the quiz answers?

**Answer:** We compute a hybrid closeness score for each eligible candidate, then sort from highest to lowest:

```text
score = 0.55 × embedding similarity
      + 0.20 × spelling similarity
      + 0.15 × rhyme similarity
      + 0.10 × length similarity
```

- **Spelling:** Dice overlap of unique adjacent character pairs.
- **Rhyme:** 1 for matching last two characters, 0.5 for matching only the last character, otherwise 0.
- **Length:** shorter character length divided by longer character length.

These weights are **hand-set application heuristics**, not a learned reranker. Some next-ayah configurations also add a **0.1 same-surah bonus**.

## 13. Why combine embeddings with spelling and rhyme?

**Answer:** A useful wrong answer should be plausible. It can be confusing because it shares a topic, looks similar, ends similarly, or has a similar length. Embeddings alone do not capture every memorization difficulty.

The embedding benchmark evaluates the encoder. It does not separately prove that these hybrid weights are optimal for learners.

## 14. How does difficulty affect the ranking?

**Answer:** The standard ayah rank windows are:

| Difficulty | Candidate ranks |
| --- | --- |
| Hard | 1–6: closest candidates |
| Medium | 7–20 |
| Easy | 21–60: related but more distinguishable |

We randomly choose **two wrong answers** from the selected band, then shuffle the final options with the correct answer. Word and surah modes scale the bands; small pools have fallback logic.

The correct answer comes from the challenge's Quran data, not from predicting the top-ranked embedding result.

## 15. How do you handle repeated or ambiguous ayahs?

**Answer:** Candidates are deduplicated by text before ranking. Challenge-specific filters exclude options that would make the question ambiguous, such as valid alternative successors of repeated ayahs.

Next-ayah challenges also filter certain near-copy candidates; missing-word challenges avoid options differing only in marks removed by word normalization.

## 16. How did you show that ranking improved?

**Answer:** We compared the base encoder and fine-tuned encoder on the **same held-out queries**, using retrieval metrics:

| Held-out evaluation: 459 queries | Base model | Fine-tuned model |
| --- | --- | --- |
| Recall@10, 768 dimensions | 0.308 | **0.374** |
| MRR, 768 dimensions | 0.252 | **0.319** |
| Recall@10, 256 dimensions | 0.286 | **0.371** |
| MRR, 256 dimensions | 0.236 | **0.301** |

At 256 dimensions, that is approximately **30% relative improvement in Recall@10** and **28% in MRR**. These are embedding retrieval results, not measured improvements in student learning.

## 17. What do Recall@10 and MRR mean?

**Answer:**

- **Recall@10:** measures how many known related verses appear among the top ten results. The project's implementation divides hits by `min(number of known related verses, 10)` and averages over queries.
- **MRR:** rewards putting the first known related verse near the top. Rank 1 gives 1, rank 2 gives 0.5, rank 5 gives 0.2.

The implementation searches the top **200 candidates** for MRR; no related result there contributes zero. It also computes **Spearman correlation** between pair similarity and dataset labels.

## 18. How can you demonstrate the ranking during a presentation?

**Answer:** Show the actual pipeline:

```text
Target ayah
    → look up its stored embedding
    → score eligible candidates
    → sort by hybrid score
    → select a difficulty band
    → choose two distractors
    → shuffle the displayed answers
```

For a technical slide, show a table of candidate ID, cosine score, spelling score, final score, and rank. Then show the before/after metrics above as evidence of encoder improvement.

The current quiz displays shuffled options and a description of its similarity method, not a full numerical ranked-candidate table. Such a table is a suggested presentation demonstration, not a feature claimed to exist already.

## 19. Did you use a vector database or a cross-encoder reranker?

**Answer:** Not in this challenge-ranking implementation. We precompute embeddings into a JSON index, load them as arrays, score candidates, and sort them directly in TypeScript.

There is no separate learned cross-encoder reranker. The hybrid scoring function is the application ranking step.

## 20. Why store int8 vectors?

**Answer:** To reduce vector storage. Each row has an int8 vector and a floating-point scale:

```text
scale = max(abs(normalized_vector)) / 127
stored_vector = round(normalized_vector / scale)
```

Raw int8 values use one quarter of the storage of float32 values. JSON, base64, IDs, and scales add overhead; the documented complete index is approximately **8 MB**.

Quantization introduces approximation. The reported 256-dimensional retrieval results should not be described as a separate exhaustive validation of the serialized int8 index.

## 21. Does a model run every time the user opens a quiz?

**Answer:** No embedding inference runs during normal challenge requests. The Python encoder builds the vectors ahead of time, and the application uses those cached vectors for ranking.

If the fine-tuned encoder is missing, index generation can use the base encoder. If the index is missing, ranking falls back to spelling, rhyme, and length.

## 22. Did you also use audio embeddings?

**Answer:** Listening challenges can use a separate **MFCC-based recording-similarity index**, blended with text closeness. MFCCs are handcrafted audio features; this is not a fourth fine-tuned neural embedding model.

The standard documented blend is 50% recording similarity and 50% text closeness where a recording-neighbor score is available. This measures recording similarity, not pronunciation correctness.

## 23. How are the three trained models different?

| Model | Input | Output | Training approach |
| --- | --- | --- | --- |
| `rattil-qaloon-v4` | Audio | Recognized words | Full-model Whisper-base fine-tuning on paired Qālūn audio and text. |
| `rattil-qaloon-tajweed-v2` | Audio | Words with tajweed tokens | Fine-tuning from v4 with rule-tagged targets and plain-speech negatives. The app uses v4 for the words. |
| `rattil-ayah-embed` | Text | Numerical embedding | Contrastive fine-tuning with Matryoshka Loss on QurSim related verse pairs. |

Whisper's log-mel features are audio representations. They are not the stored text vectors used for challenge ranking.

## 24. What limitations should you explain if asked?

**Answer:** QurSim relationships reflect one tafsir's links, and its text uses Hafs spelling and numbering, whereas the application index uses normalized Qālūn text. Embedding closeness can reflect shared wording rather than a complete interpretation of meaning.

The reported benchmark supports better retrieval of known related verses. It does not establish tafsir correctness, tajweed accuracy, or improved learning outcomes for the hybrid ranking system.

## Source files

- `src/learning/train_ayah_embedder.py`: model, loss, split, and training settings.
- `src/learning/qursim.py`: dataset loading and retrieval metrics.
- `src/learning/build_text_embeddings.py`: ayah/word encoding and int8 storage.
- `src/learning/benchmark_text_embeddings.py`: model comparison.
- `web/lib/challenges.ts`: cosine scoring, hybrid ranking, and difficulty bands.
- [Challenge distractor documentation](docs/CHALLENGE_DISTRACTORS.md): recorded metrics and model selection.
- [Training and audio overview](TECH_OVERVIEW.md): the two audio models and their processing pipeline.
