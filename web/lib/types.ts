export type Region = { source_ayah: number; display_ayah: number; polygon: string; x: string; y: string; page: string };
export type Ayah = { ayah: number; text: string; displayText?: string; normalized: string; regions: Region[] };
export type SurahInfo = { id: number; name: string; arabic: string; trained: boolean; ayahCount: number; preciseGeometry: boolean };
export type Surah = SurahInfo & { ayahs: Ayah[] };
export type Manifest = { surahs: SurahInfo[]; pages: Record<string, { path: string; viewBox: string }> };
export type WordResult = { index: number; text: string; status: "correct" | "missed" | "pending"; heard?: string };
export type Result = { status: "correct" | "missed" | "listening"; score: number; missing: string[]; final: boolean; words: WordResult[] };
export type Update = { type: "update" | "finished"; current: number | null; results: Record<number, Result>; transcript: string; complete: boolean; advanced: boolean; latency_ms?: number; audio_seconds?: number; decoded_seconds?: number; pending_seconds?: number; window_seconds?: number; tracking_uncertain?: boolean; asr_scorable?: boolean; decode_flags?: string[] };
