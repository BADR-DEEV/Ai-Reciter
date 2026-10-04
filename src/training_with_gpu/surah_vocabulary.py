"""Word-level token trie for OPTIONAL, openly labelled surah-assisted decoding.

Not an ayah-order trie: arbitrary word orders, skips and repeats remain possible.
It restricts spelling but cannot prevent Quran-word hallucinations/completions.
Never use this mode as evidence that a learner pronounced a word or omitted none.
"""
from functools import lru_cache
from itertools import product
from src.dataset_collection.qaloon_audio2text import load_quran, normalize_quran_for_asr, normalize_with_harakat, strip_harakat


class SurahVocabularyConstraint:
    def __init__(self, tokenizer, vocabulary):
        self.prefix = tuple(tokenizer.prefix_tokens)
        self.eos = tokenizer.eos_token_id
        self.initial, self.following = {}, {}
        self.vocabulary = frozenset(vocabulary)
        if not self.vocabulary or any(not word or len(word.split()) != 1 for word in self.vocabulary):
            raise ValueError("Require nonempty whole-word vocabulary")
        for word in sorted(self.vocabulary):
            # A decoder may start with a leading space. Raw hamza forms are
            # legitimate even when the METRIC reference folds alif carriers.
            for text, root in ((word, self.initial), (" " + word, self.initial), (" " + word, self.following)):
                ids = tokenizer.encode(text, add_special_tokens=False)
                if not ids or self.eos in ids or tokenizer.decode(ids, skip_special_tokens=True) != text:
                    raise ValueError("Tokenizer cannot losslessly represent canonical whole word")
                node = root
                for token in ids:
                    node = node.setdefault(token, {})
                node[None] = True

    @classmethod
    def from_canonical(cls, tokenizer, surah):
        rows = [row for (s, _), row in load_quran().items() if s == surah]
        normalized = {word for row in rows for word in normalize_quran_for_asr(row["raw"]).split()}
        surface = set(normalized)
        # The scoring normalizer folds ا/أ/إ/آ. Masking those decoder forms
        # changes the acoustic search even when they score as the SAME word.
        # Enumerate this exact equivalence class, never insert an extra alif
        # (ملك cannot become مالك) or fold medial/final hamza/maqsura.
        for word in normalized:
            positions = [i for i, character in enumerate(word) if character == "ا"]
            if len(positions) > 4:
                continue  # Bound uncommon huge spelling classes; mode stays experimental.
            for carriers in product("اأإآ", repeat=len(positions)):
                characters = list(word)
                for position, carrier in zip(positions, carriers):
                    characters[position] = carrier
                form = "".join(characters)
                if normalize_quran_for_asr(form) == word:
                    surface.add(form)
        for row in rows:
            for word in normalize_with_harakat(row["raw"]).split():
                candidate = strip_harakat(word)
                for form in (candidate, candidate.replace("ٱ", "ا")):
                    # Only source-attested spelling alternatives, no arbitrary
                    # string similarity, no missing-word completion/extra grants.
                    if normalize_quran_for_asr(form) in normalized:
                        surface.add(form)
        constraint = cls(tokenizer, surface)
        constraint.canonical_normalized_vocabulary = frozenset(normalized)
        return constraint

    @lru_cache(maxsize=4096)
    def _allowed(self, sequence):
        if sequence[:len(self.prefix)] != self.prefix:
            raise ValueError("Unexpected Whisper decoder prefix; constraints fail closed")
        body = sequence[len(self.prefix):]
        terminated = self.eos in body
        if terminated:
            ending = body.index(self.eos)
            if any(token != self.eos for token in body[ending:]):
                raise ValueError("Unexpected non-padding tokens after EOS")
            body = body[:ending]
        active = [self.initial]
        for token in body:
            candidates = active + ([self.following] if any(None in n for n in active) else [])
            active = [node[token] for node in candidates if token in node]
            if not active:
                raise ValueError("Generated sequence left canonical vocabulary grammar")
        if terminated:
            if body and not any(None in node for node in active):
                raise ValueError("EOS emitted inside a canonical word")
            return (self.eos,)
        tokens = {key for node in active for key in node if key is not None}
        if not body or any(None in node for node in active):
            tokens.add(self.eos)
            if body:
                tokens.update(self.following)
        if not tokens:
            raise ValueError("Constraint has no legal continuation; no fallback hallucinated tokens")
        return tuple(sorted(tokens))

    def allowed(self, sequence):
        return list(self._allowed(tuple(sequence)))
