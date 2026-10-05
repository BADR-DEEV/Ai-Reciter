"use client";

import { useEffect, useState } from "react";
import { API } from "./learn/api";
import type { Result, Update, WordResult } from "./types";

// The inference service serves named engines: "plain" (word recognition) and,
// once trained, "tajweed" (same words plus <tj:...> tags). Asking for "tajweed"
// while it is unavailable falls back to plain; responses then carry fallback_reason.
export type ModelName = "plain" | "tajweed";
export type FallbackReason = "missing" | "load_failed" | "not_configured";
export type ModelInfo = { name: string; label?: string; path: string; kind: ModelName; available: boolean; loaded: boolean; device: string | null; error?: string };
export type Health = {
  status?: string; ready?: boolean; device: string; model?: string; default_model?: string; models?: ModelInfo[];
  dtype?: string | null; beams?: number | null; sessions: number; max_sessions?: number; busy?: boolean; last_latency_ms?: number | null;
};

/** Tags the tajweed model heard on a word, by index into the transcript's words. */
export type TajweedTag = { word_index: number; tags: string[] };
export type TajweedFeedback = Record<string, unknown>;
export type ModelUsage = { model_used?: string; fallback_reason?: FallbackReason | null; tajweed_tags?: TajweedTag[]; tajweed_feedback?: TajweedFeedback };
export type TaggedWordResult = WordResult & { tags?: string[] };
export type TaggedResult = Omit<Result, "words"> & { words: TaggedWordResult[]; tajweed_feedback?: TajweedFeedback };
export type TaggedUpdate = Omit<Update, "results"> & { results: Record<number, TaggedResult>; tajweed_tags?: TajweedTag[] };

export async function fetchHealth(signal?: AbortSignal): Promise<Health | null> {
  try {
    const response = await fetch(`${API}/health`, { signal, cache: "no-store" });
    return response.ok ? await response.json() : null;
  } catch { return null; }
}

/** Polls /health (paused while the tab is hidden). `checked` is false until the first answer. */
export function useServiceHealth(intervalMs = 20000) {
  const [health, setHealth] = useState<Health | null>(null);
  const [checked, setChecked] = useState(false);
  useEffect(() => {
    let disposed = false;
    const controller = new AbortController();
    const refresh = async () => {
      if (document.hidden) return;
      const data = await fetchHealth(AbortSignal.any([controller.signal, AbortSignal.timeout(5000)]));
      if (!disposed) { setHealth(data); setChecked(true); }
    };
    void refresh(); const timer = setInterval(() => void refresh(), intervalMs);
    return () => { disposed = true; controller.abort(); clearInterval(timer); };
  }, [intervalMs]);
  return { health, checked };
}

/** Which engines the service can use right now, e.g. to enable a "read with tajweed" toggle. */
export function useAvailableModels(intervalMs?: number) {
  const { health, checked } = useServiceHealth(intervalMs);
  // Older single-model services report no `models`: treat them as plain only.
  const models: ModelInfo[] = health?.models ?? (health ? [{ name: "plain", path: "", kind: "plain", available: true, loaded: true, device: health.device }] : []);
  const available = (name: ModelName) => models.some(m => m.name === name && m.available);
  return { online: checked ? health !== null : null, health, models, defaultModel: health?.default_model ?? "plain", plain: available("plain"), tajweed: available("tajweed") };
}
