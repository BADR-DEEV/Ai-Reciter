export type TimedWord = { index: number; text: string; start: number; end: number };
export type PlaybackTiming = { displayText: string; duration: number; words: TimedWord[]; status: string };
export type PlaybackCursor = { ayah: number; word: number | null; tracking: "word" | "ayah" };

/** Gaps remain unhighlighted; never invent a letter/word clock by interpolation. */
export function wordAtTime(words: TimedWord[], time: number): number | null {
  return words.find(word => time >= word.start && time < word.end)?.index ?? null;
}

export function validTimings(value: PlaybackTiming, displayText: string): boolean {
  if (value.displayText !== displayText || !Number.isFinite(value.duration) || value.duration <= 0) return false;
  const tokens = displayText.split(/\s+/);
  if (!Array.isArray(value.words) || value.words.length !== tokens.length) return false;
  let end = 0;
  return value.words.every((word, i) => {
    const valid = word.index === i && word.text === tokens[i] && Number.isFinite(word.start) && Number.isFinite(word.end)
      && word.start >= end && word.end > word.start && word.end <= value.duration;
    end = word.end;
    return valid;
  });
}
