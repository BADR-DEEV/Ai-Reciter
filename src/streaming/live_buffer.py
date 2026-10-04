"""Bounded chronological PCM and blind transcript overlap for windowed ASR.

No expected Quran text enters decoding/stitching. Ambiguous joins must recover
with retained audio, not fuzzy text completion or invented missing words.
"""
import numpy as np
from .matcher import words


class AudioQueue:
    def __init__(self, sample_rate=16000, capacity_seconds=45):
        self.sample_rate = sample_rate
        self.storage = np.empty(sample_rate * capacity_seconds, dtype=np.float32)
        self.start = self.total = 0

    def append(self, audio):
        if self.total - self.start + len(audio) > len(self.storage):
            raise ValueError("Recognition cannot keep up: audio queue is full. No unprocessed speech was silently discarded; resume from your last tracked ayah.")
        offset = self.total % len(self.storage)
        first = min(len(audio), len(self.storage) - offset)
        self.storage[offset:offset + first] = audio[:first]
        self.storage[:len(audio) - first] = audio[first:]
        self.total += len(audio)

    def read(self, start, end):
        if not self.start <= start <= end <= self.total:
            raise ValueError("Requested PCM is outside retained chronological audio")
        offset = start % len(self.storage)
        first = min(end - start, len(self.storage) - offset)
        return np.concatenate((self.storage[offset:offset + first], self.storage[:end - start - first]))

    def discard_before(self, start):
        if not self.start <= start <= self.total:
            raise ValueError("Invalid acknowledged audio frontier")
        self.start = start


class TranscriptOverlap:
    def __init__(self, limit=600):
        self.limit = limit
        self.start = None
        self.raw = []
        self.prefix = []

    def accept(self, start, transcript, silent_overlap=False):
        new = words(transcript)
        if not new:
            return None
        if self.start is None:
            prefix = []
        elif start == self.start:
            prefix = self.prefix
        elif start > self.start:
            matches = [(offset, size) for offset in range(len(self.raw))
                       for size in range(2, min(len(self.raw) - offset, len(new)) + 1)
                       if self.raw[offset:offset + size] == new[:size]
                       and len(self.raw) - offset - size <= (0 if size == 2 else 3)]
            origins = {offset for offset, _ in matches}
            if len(origins) > 1:
                return None  # Repeated phrases make the join ambiguous; don't deduplicate speech.
            if matches:
                # The old trailing hypothesis can revise (e.g. a cut word).
                # Require >=3 exact blind anchors to replace up to three old
                # trailing words; never fuzzy-complete them from expected text.
                offset = matches[0][0]
                prefix = self.prefix + self.raw[:offset]
            elif silent_overlap:
                prefix = self.prefix + self.raw
            else:
                return None  # Caller retries a larger retained context.
        else:
            raise ValueError("Transcript audio origins must be chronological")
        allowance = max(0, self.limit - len(new))
        self.prefix = prefix[-allowance:] if allowance else []
        self.raw, self.start = new[-self.limit:], start
        return " ".join(self.prefix + self.raw)

    def reset(self):
        self.start, self.raw, self.prefix = None, [], []

    def commit_boundary(self, start):
        """Carry blind words across acoustically quiet PCM, including internal waqf."""
        self.prefix = (self.prefix + self.raw)[-self.limit:]
        self.raw = []
        self.start = start
