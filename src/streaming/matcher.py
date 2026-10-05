"""Ordered, one-to-one word matching for incremental Quran transcription.

This is transcript agreement, not acoustic correctness or a tajweed judgment.
The expected text is never supplied as a Whisper decoder prompt.
"""
from functools import lru_cache
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dataset_collection"))
from qaloon_audio2text import normalize_quran_for_asr
from .tajweed_tags import strip_tags


@lru_cache(maxsize=512)
def _words(text):
    # Tajweed tags are not letters; the normalizer would leave their pieces as words.
    return tuple(normalize_quran_for_asr(strip_tags(text)).split())


def words(text):
    return list(_words(text))


def alignment(expected, heard):
    """Exact normalized LCS, preserving word order and multiplicity.

    Fuzzy spelling matches can mark Hafs مالك as Qaloon ملك or hide a real
    substitution. Navigation uses acoustic-ASR agreement, never forced completion.
    """
    n, m = len(expected), len(heard)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            similar = expected[i] == heard[j]
            dp[i][j] = max(dp[i + 1][j], dp[i][j + 1], dp[i + 1][j + 1] + int(similar))
    pairs = []
    i, j = 0, 0
    while i < n and j < m:
        similar = expected[i] == heard[j]
        if similar and dp[i][j] == dp[i + 1][j + 1] + 1:
            pairs.append((i, j))
            i, j = i + 1, j + 1
        elif dp[i + 1][j] > dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


class RecitationTracker:
    def __init__(self, ayahs, threshold=0.65, surah=None, context_limit=80):
        # Cached normalized fields may predate the v2 bug fix. Derive matching
        # text from intact source text when available, without altering artwork.
        self.ayahs = [{**row, "normalized": normalize_quran_for_asr(row["text"]) if row.get("text") else row["normalized"]}
                      for row in ayahs]
        self.threshold = threshold
        self.index = 0
        self.results = {}
        self.last_candidate = None
        self.stability = 0
        self.last_partial = ""
        self.surah = surah
        self.omission_observations = {}
        self.completed_context = []
        self.tentative_prefix = False
        self.has_advanced = False
        self.context_limit = context_limit
        self.uncertain_audio = False
        self.tagged_spans = {}

    @property
    def done(self):
        return self.index >= len(self.ayahs)

    def snapshot(self, transcript="", changed=False):
        return {"type": "update", "current": None if self.done else self.ayahs[self.index]["ayah"],
                "results": self.results.copy(), "transcript": transcript,
                "complete": self.done, "advanced": changed, "threshold": self.threshold,
                "tentative_prefix": self.tentative_prefix}

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

    def word_results(self, index, heard, pairs, completed=False, boundary=False, tags=None):
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
            elif completed and not self.uncertain_audio:
                status = "missed"
            elif i < frontier and len(pairs) >= 2:
                # A decoder can revise partial text. Confirm skipped positions
                # twice (or at a voiced boundary), never color future words red.
                count = self.omission_observations.get(key, 0) + 1
                self.omission_observations[key] = count
                if not self.uncertain_audio and (boundary or count >= 2):
                    status = "missed"
            else:
                self.omission_observations.pop(key, None)
            word = {"index": i, "text": text, "status": status}
            if i in matched:
                word["heard"] = heard[matched[i]]
                if tags and tags[matched[i]]:
                    word["tags"] = list(tags[matched[i]])
            results.append(word)
        return results

    def clear_context(self):
        self.completed_context = []

    def feed(self, transcript, final=False, continuous=False, stable_prefix=0, tags=None):
        """`tags` (optional) lists each word's tajweed tags, aligned with words(transcript)."""
        self.tentative_prefix = False
        if self.done:
            return self.snapshot(transcript)
        heard = words(transcript)
        tags = list(tags) if tags is not None and len(tags) == len(heard) else None
        if self.surah == 1 and self.index == 0:
            intro = words("بسم الله الرحمن الرحيم")
            if heard[:len(intro)] == intro:
                heard = heard[len(intro):]
                tags = tags and tags[len(intro):]
                stable_prefix = max(0, stable_prefix - len(intro))
        original = heard[:]
        consumed = 0
        if continuous and self.completed_context:
            context_pairs = alignment(self.completed_context, heard)
            # Retained audio can re-decode completed ayahs. Remove only a
            # strongly aligned context ending, never an arbitrary shared word.
            if (len(context_pairs) >= min(2, len(self.completed_context))
                    and len(context_pairs) / len(self.completed_context) >= 0.8
                    and context_pairs[-1][0] == len(self.completed_context) - 1):
                consumed = context_pairs[-1][1] + 1
                heard = heard[consumed:]
                tags = tags and tags[consumed:]
                stable_prefix = max(0, stable_prefix - consumed)
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
            # A high partial match with a mistranscribed final word used to
            # disable lookahead forever. Strong later-ayah evidence can recover
            # without inventing a terminal match or changing blind ASR output.
            if not terminal or score < self.threshold:
                for candidate in range(self.index + 1, min(self.index + 4, len(self.ayahs))):
                    other, other_terminal, other_end, other_missing, other_pairs = self.evaluate(candidate, heard)
                    ordered_later = (other_terminal and len(other_pairs) >= 2
                                     and other_pairs[0][1] > max((j for _, j in pairs), default=-1))
                    if other >= self.threshold and (other > score + 0.2 or ordered_later):
                        target, score, terminal, end, missing = candidate, other, other_terminal, other_end, other_missing
                        pairs = other_pairs
            evidence = (target, terminal, end)
            if evidence == self.last_candidate:
                self.stability += 1
            else:
                self.last_candidate, self.stability = evidence, 1
            stable = final or self.stability >= 2 or terminal and end is not None and end < stable_prefix
            if target != self.index and not stable:
                break
            if target != self.index:
                for skipped in range(self.index, target):
                    ayah = self.ayahs[skipped]
                    previous = self.results.get(ayah["ayah"], {})
                    word_results = [dict(word, status="correct" if word["status"] == "correct" else "pending" if self.uncertain_audio else "missed")
                                    for word in previous.get("words", self.word_results(skipped, [], [], completed=True))]
                    coverage = sum(word["status"] == "correct" for word in word_results) / max(1, len(word_results))
                    self.results[ayah["ayah"]] = {"status": "correct" if coverage >= self.threshold else "listening" if self.uncertain_audio else "missed",
                                                  "score": round(coverage, 3), "final": True,
                                                  "missing": [word["text"] for word in word_results if word["status"] == "missed"],
                                                  "words": word_results}
                self.index = target
                changed = True
                self.has_advanced = True
            # Whisper can hallucinate a single next-verse word at an utterance
            # tail. With retained completed context, require two target-word
            # anchors before first committing that new verse. Keep its PCM as
            # tentative; a genuine continued prefix can still be corroborated.
            previous = self.results.get(self.ayahs[self.index]["ayah"], {})
            if (continuous and self.has_advanced and len(pairs) == 1
                    and len(words(self.ayahs[self.index]["normalized"])) > 1
                    and not any(w["status"] == "correct" for w in previous.get("words", []))):
                self.tentative_prefix = True
                break
            # Whisper can emit a lone word from the trailing breath/silence of
            # the previous ayah. This is not evidence the next ayah was reached.
            if score == 0 and not (final and len(heard) >= 3):
                break
            ayah = self.ayahs[self.index]["ayah"]
            self.results[ayah] = {"status": "correct" if score >= self.threshold else "listening",
                                  "score": round(score, 3), "missing": missing, "final": False,
                                  "words": self.word_results(self.index, heard, pairs,
                                                             completed=terminal and stable, boundary=final, tags=tags)}
            if terminal and stable:
                self.results[ayah].update(status="correct" if score >= self.threshold else "listening" if self.uncertain_audio else "missed", final=True)
                self.index += 1
                changed = True
                self.has_advanced = True
                consumed += end + 1
                if tags is not None:
                    self.tagged_spans[ayah] = (heard[:end + 1], tags[:end + 1])
                if continuous:
                    # Bound matcher history to the rolling ASR context.
                    self.completed_context = original[:consumed][-self.context_limit:]
                heard = heard[end + 1:]
                tags = tags and tags[end + 1:]
                stable_prefix = max(0, stable_prefix - end - 1)
                self.last_candidate, self.stability = None, 0
            else:
                if final and score < self.threshold and not self.uncertain_audio:
                    self.results[ayah]["status"] = "missed"
                break
        return self.snapshot(transcript, changed)
