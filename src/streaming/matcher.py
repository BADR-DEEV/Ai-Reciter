"""Ordered, one-to-one word matching for incremental Quran transcription.

This is transcript agreement, not acoustic correctness or a tajweed judgment.
The expected text is never supplied as a Whisper decoder prompt.
"""
from difflib import SequenceMatcher
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dataset_collection"))
from qaloon_audio2text import normalize_quran_for_asr


def words(text):
    return normalize_quran_for_asr(text).split()


def alignment(expected, heard):
    """LCS with fuzzy substitutions, preserving word order and multiplicity."""
    n, m = len(expected), len(heard)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            similar = SequenceMatcher(None, expected[i], heard[j]).ratio() >= 0.8
            dp[i][j] = max(dp[i + 1][j], dp[i][j + 1], dp[i + 1][j + 1] + int(similar))
    pairs = []
    i, j = 0, 0
    while i < n and j < m:
        similar = SequenceMatcher(None, expected[i], heard[j]).ratio() >= 0.8
        if similar and dp[i][j] == dp[i + 1][j + 1] + 1:
            pairs.append((i, j))
            i, j = i + 1, j + 1
        elif dp[i + 1][j] > dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


class RecitationTracker:
    def __init__(self, ayahs, threshold=0.65, surah=None):
        self.ayahs = ayahs
        self.threshold = threshold
        self.index = 0
        self.results = {}
        self.last_candidate = None
        self.stability = 0
        self.last_partial = ""
        self.surah = surah
        self.omission_observations = {}

    @property
    def done(self):
        return self.index >= len(self.ayahs)

    def snapshot(self, transcript="", changed=False):
        return {"type": "update", "current": None if self.done else self.ayahs[self.index]["ayah"],
                "results": self.results.copy(), "transcript": transcript,
                "complete": self.done, "advanced": changed, "threshold": self.threshold}

    def evaluate(self, index, heard):
        expected = words(self.ayahs[index]["normalized"])
        pairs = alignment(expected, heard)
        score = len(pairs) / max(1, len(expected))
        # Require an exact normalized terminal word plus earlier ordered evidence.
        end_pair = next((j for i, j in pairs if i == len(expected) - 1
                         and heard[j] == expected[-1]), None)
        minimum = min(2, len(expected))
        terminal = end_pair is not None and len(pairs) >= minimum
        missing = [word for i, word in enumerate(expected) if i not in {p[0] for p in pairs}]
        return score, terminal, end_pair, missing, pairs

    def word_results(self, index, heard, pairs, completed=False, boundary=False):
        expected = words(self.ayahs[index]["normalized"])
        ayah = self.ayahs[index]["ayah"]
        matched = dict(pairs)
        frontier = max(matched, default=-1)
        results = []
        for i, text in enumerate(expected):
            key = (ayah, i)
            status = "pending"
            if i in matched:
                status = "correct"
                self.omission_observations.pop(key, None)
            elif completed:
                status = "missed"
            elif i < frontier and len(pairs) >= 2:
                # A decoder can revise partial text. Confirm skipped positions
                # twice (or at a voiced boundary), never color future words red.
                count = self.omission_observations.get(key, 0) + 1
                self.omission_observations[key] = count
                if boundary or count >= 2:
                    status = "missed"
            else:
                self.omission_observations.pop(key, None)
            word = {"index": i, "text": text, "status": status}
            if i in matched:
                word["heard"] = heard[matched[i]]
            results.append(word)
        return results

    def feed(self, transcript, final=False):
        if self.done:
            return self.snapshot(transcript)
        heard = words(transcript)
        if self.surah == 1 and self.index == 0:
            intro = words("بسم الله الرحمن الرحيم")
            if heard[:len(intro)] == intro:
                heard = heard[len(intro):]
        if not heard:
            return self.snapshot(transcript)
        self.last_partial = transcript
        changed = False
        # One decoder window may contain several consecutive ayahs.
        while heard and not self.done:
            score, terminal, end, missing, pairs = self.evaluate(self.index, heard)
            target = self.index
            # A silent gap does not mean a skipped ayah. Only a strong later
            # verse match can advance past an omission; bounded lookahead.
            if score < self.threshold:
                for candidate in range(self.index + 1, min(self.index + 4, len(self.ayahs))):
                    other, other_terminal, other_end, other_missing, other_pairs = self.evaluate(candidate, heard)
                    if other >= self.threshold and other > score + 0.2:
                        target, score, terminal, end, missing = candidate, other, other_terminal, other_end, other_missing
                        pairs = other_pairs
            evidence = (target, terminal, end)
            if evidence == self.last_candidate:
                self.stability += 1
            else:
                self.last_candidate, self.stability = evidence, 1
            stable = final or self.stability >= 2
            if target != self.index and not stable:
                break
            if target != self.index:
                for skipped in range(self.index, target):
                    ayah = self.ayahs[skipped]
                    previous = self.results.get(ayah["ayah"], {})
                    word_results = [dict(word, status="correct" if word["status"] == "correct" else "missed")
                                    for word in previous.get("words", self.word_results(skipped, [], [], completed=True))]
                    coverage = sum(word["status"] == "correct" for word in word_results) / max(1, len(word_results))
                    self.results[ayah["ayah"]] = {"status": "correct" if coverage >= self.threshold else "missed",
                                                  "score": round(coverage, 3), "final": True,
                                                  "missing": [word["text"] for word in word_results if word["status"] == "missed"],
                                                  "words": word_results}
                self.index = target
                changed = True
            # Whisper can emit a lone word from the trailing breath/silence of
            # the previous ayah. This is not evidence the next ayah was reached.
            if score == 0 and not (final and len(heard) >= 3):
                break
            ayah = self.ayahs[self.index]["ayah"]
            self.results[ayah] = {"status": "correct" if score >= self.threshold else "listening",
                                  "score": round(score, 3), "missing": missing, "final": False,
                                  "words": self.word_results(self.index, heard, pairs,
                                                             completed=terminal and stable, boundary=final)}
            if terminal and stable:
                self.results[ayah].update(status="correct" if score >= self.threshold else "missed", final=True)
                self.index += 1
                changed = True
                heard = heard[end + 1:]
                self.last_candidate, self.stability = None, 0
            else:
                if final and score < self.threshold:
                    self.results[ayah]["status"] = "missed"
                break
        return self.snapshot(transcript, changed)
