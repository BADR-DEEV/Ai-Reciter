import base64
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .build_text_embeddings import build, quantize, vocabulary, word_key
from .qursim import evaluate, held_out
from .train_ayah_embedder import split


class TextEmbeddingTests(unittest.TestCase):
    def test_word_keys_match_the_web_app(self):
        # Same expectations as web/tests/challenges.spec.ts (wordKey).
        for word, key in [("اَ۬لنَّاسِ", "الناس"), ("يَوْمَئِذٖ", "يوميذ"), ("أَعْمَٰلَهُمْ", "اعملهم"), ("اُ۬لْقُرْءَانَ", "القرءان")]:
            self.assertEqual(word_key(word), key)

    def test_vocabulary_groups_vowel_variants(self):
        self.assertEqual(vocabulary([{"display": "اَ۬لنَّاسِ اِ۬لنَّاسُ نَاسٌ ـ"}]), ["الناس", "ناس"])

    def test_int8_rows_keep_cosine(self):
        rng = np.random.default_rng(0)
        vectors = rng.normal(size=(20, 64)).astype(np.float32)
        packed = quantize(vectors)
        values = np.frombuffer(base64.b64decode(packed["vectors"]), dtype=np.int8).reshape(20, 64).astype(np.float32)
        restored = values * np.array(packed["scales"])[:, None]
        unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        np.testing.assert_allclose(restored @ restored.T, unit @ unit.T, atol=0.02)

    def test_build_writes_web_index(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            surahs = root / "surahs"
            surahs.mkdir()
            for number in range(1, 115):
                ayahs = [{"ayah": 1, "text": f"كَلِمَةٌ {number} ١", "normalized": f"كلمة {number}"}]
                (surahs / f"{number:03d}.json").write_text(json.dumps({"ayahs": ayahs}), encoding="utf-8")
            encoder = lambda texts: np.eye(len(texts), 300, dtype=np.float32)
            output = root / "index.json"
            build(encoder, "test/model", "abc", output, dim=256, root=surahs)
            data = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(data["ayahs"]["ids"][:2], ["1:1", "2:1"])
        self.assertEqual(data["ayahs"]["scales"].__len__(), 114)
        self.assertEqual(len(base64.b64decode(data["ayahs"]["vectors"])), 114 * 256)
        self.assertEqual(data["words"]["keys"], ["كلمة"])
        self.assertEqual(data["label"], "model")

    def test_held_out_verses_never_reach_training(self):
        pairs = [(f"{s}:{v}", f"{s}:{v + 1}", 2) for s in range(1, 30) for v in range(1, 10)]
        train, test_pairs, queries = split(pairs)
        self.assertTrue(train and queries)
        self.assertFalse({v for pair in train for v in pair} & set(queries))
        self.assertTrue(all(held_out(q) for q in queries))

    def test_evaluate_rewards_related_neighbours(self):
        verses = {"1:1": "a", "1:2": "b", "2:1": "c", "2:2": "d"}
        pairs = [("1:1", "1:2", 2), ("2:1", "2:2", 2), ("1:1", "2:1", 0)]
        vectors = {"a": [1, 0], "b": [0.9, 0.1], "c": [0, 1], "d": [0.1, 0.9]}
        result = evaluate(lambda texts: np.array([vectors[t] for t in texts], dtype=np.float32), verses, pairs)
        self.assertEqual(result["2"]["mrr"], 1.0)


if __name__ == "__main__":
    unittest.main()
