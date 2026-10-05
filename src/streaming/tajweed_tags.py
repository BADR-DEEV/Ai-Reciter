"""Tajweed tokens (`<n_ikhfa>`, `<qalqala>`, `<mad>`, ...) that the tajweed model appends to words.

A tag belongs to the word it sits in or follows; a tag before any word is
dropped. Tags are removed before normalization/matching, so word matching never
depends on them. The first model's `<tj:name>` tags are read too and renamed.
"""
from difflib import SequenceMatcher
import re

TAG = re.compile(r"<\s*(tj\s*:\s*)?([a-z_]+)\s*>")
# rattil-tajweed-v1 (<tj:...>) names → the current acoustic tokens (src/tajweed/rules.py TAGS)
LEGACY = {"ikhfa": "n_ikhfa", "iqlab": "m_ikhfa", "ikhfa_shafawi": "m_ikhfa", "ghunna": "n_ghunna", "idgham_shafawi": "m_ghunna",
          "madd_lazim": "mad", "madd_muttasil": "mad", "idgham_ghunna": "idgham_ghunna", "qalqala": "qalqala",
          "silah": "silah", "tasheel": "tasheel"}


def tag_name(match):
    return LEGACY.get(match.group(2)) if match.group(1) else match.group(2)


def tokens(text):
    """[(word, [tags])] in transcript order. A tag belongs to the word it sits in
    or follows; removing a tag never splits a word (the model sometimes emits a
    tag after a sub-word piece) and spaces around tags are tolerated."""
    text = text or ""
    pieces, marks, last = [], [], 0
    for match in TAG.finditer(text):
        pieces.append(text[last:match.start()])
        marks.append((sum(map(len, pieces)), tag_name(match)))
        last = match.end()
    pieces.append(text[last:])
    clean = "".join(pieces)
    words = [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", clean)]
    out = [(word, []) for word, _, _ in words]
    for position, tag in marks:
        owner = None
        for i, (_, start, end) in enumerate(words):
            if start < position <= end:
                owner = i
                break
            if end <= position:
                owner = i
        if tag and owner is not None and tag not in out[owner][1]:
            out[owner][1].append(tag)
    return out


def split_tags(text):
    """(tag-free text, [(word_index, [tags])]) for the words that carry tags."""
    pairs = tokens(text)
    return " ".join(w for w, _ in pairs), [(i, tags) for i, (_, tags) in enumerate(pairs) if tags]


def strip_tags(text):
    return " ".join(w for w, _ in tokens(text)) if "<" in (text or "") else " ".join((text or "").split())


def tagged_words(text, normalize=str.split):
    """Words of the tag-free text and each word's tags.

    `normalize` maps one raw word to zero or more matcher words; the tags go to
    the last of them (or the previous word when it normalizes away).
    """
    heard, tags = [], []
    for word, word_tags in tokens(text):
        parts = normalize(word)
        heard.extend(parts)
        tags.extend([] for _ in parts[1:])
        if parts:
            tags.append(list(word_tags))
        elif word_tags and tags:
            tags[-1] = tags[-1] + [t for t in word_tags if t not in tags[-1]]
    return heard, tags


def tag_list(tags):
    """Response form: [{word_index, tags}] for tagged words only."""
    return [{"word_index": i, "tags": t} for i, t in enumerate(tags) if t]


def carry_tags(previous, previous_tags, prefix, raw, raw_tags):
    """Tags for a stitched `prefix + raw` transcript.

    `raw` is the newest window (tagged by `raw_tags`, aligned to that window's
    words); `prefix` is a contiguous run of the previous stitched words, which
    keep the tags they were heard with.
    """
    n = len(prefix)
    start = 0 if not n else next((i for i in range(len(previous) - n + 1) if previous[i:i + n] == prefix), None)
    carried = previous_tags[start:start + n] if start is not None and len(previous_tags) == len(previous) else [[] for _ in prefix]
    return carried + raw_tags[len(raw_tags) - len(raw):]


def transfer_tags(plain, tagged, normalize=str.split):
    """The plain transcript's words, each carrying the tags the tajweed model
    heard on the same word. Words are aligned on their normalized spelling; a
    run of equally many differing words is aligned by position."""
    plain_words = (plain or "").split()
    heard = tokens(tagged)
    a = [" ".join(normalize(w)) for w in plain_words]
    b = [" ".join(normalize(w)) for w, _ in heard]
    tags = [[] for _ in plain_words]
    for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal" or (op == "replace" and i2 - i1 == j2 - j1):
            for k in range(i2 - i1):
                tags[i1 + k] = heard[j1 + k][1]
    return " ".join(word + "".join(f"<{t}>" for t in word_tags) for word, word_tags in zip(plain_words, tags))
