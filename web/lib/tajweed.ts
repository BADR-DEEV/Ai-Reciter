export type TajweedSpan = { start: number; end: number; rule: string; status: "needs-review"; context_note: string };
export type TajweedAyah = { ayah: number; text: string; spans: TajweedSpan[]; warnings: string[]; status: string };
export type TajweedMushaf = { status: string; approved_ayahs: number; rules: Record<string, { en: string; ar: string; color: string; harakat: number[] | null }>; surahs: { id: number; ayahs: TajweedAyah[] }[] };

export function tajweedSegments(text: string, spans: TajweedSpan[], start: number, end: number) {
  const relevant = spans.filter(s => s.start < end && s.end > start && s.start >= 0 && s.end <= text.length);
  const cuts = [...new Set([start, end, ...relevant.flatMap(s => [Math.max(start, s.start), Math.min(end, s.end)])])].sort((a, b) => a - b);
  return cuts.slice(0, -1).map((position, i) => ({ text: text.slice(position, cuts[i + 1]), rules: relevant.filter(s => s.start <= position && s.end >= cuts[i + 1]).map(s => s.rule) }));
}
