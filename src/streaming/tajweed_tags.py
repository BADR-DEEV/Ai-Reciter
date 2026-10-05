"""Tajweed tag tokens (`<tj:name>`) that the tajweed model appends to words.

A tag belongs to the PRECEDING word; a tag before any word is dropped. Tags are
removed before normalization/matching, so word matching never depends on them.
"""
import re

TAG = re.compile(r"<\s*tj\s*:\s*([a-z_]+)\s*>")


def tokens(text):
    """[(word, [tags])] in transcript order, tolerant of spaces around tags."""
    out = []
    for i, part in enumerate(TAG.split(text or "")):
        if not i % 2:
            out.extend((word, []) for word in part.split())
        elif out and part not in out[-1][1]:
            out[-1][1].append(part)
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
