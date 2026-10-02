"use client";

import { useCallback, useEffect, useState } from "react";

// Per-browser progress. Storage can be unavailable (private windows, blocked
// site data), so every access is guarded and the app works without it.
export type LessonRecord = { done: boolean; accuracy: number; xp: number; at: string };
export type Progress = { lessons: Record<string, LessonRecord>; xp: number; days: string[] };

const KEY = "rattil.learn.v1";
const empty: Progress = { lessons: {}, xp: 0, days: [] };

function read(): Progress {
  try { return { ...empty, ...JSON.parse(localStorage.getItem(KEY) || "{}") }; }
  catch { return empty; }
}

const today = () => new Date().toISOString().slice(0, 10);

export function streak(days: string[]) {
  const set = new Set(days);
  const date = new Date();
  if (!set.has(today())) date.setDate(date.getDate() - 1);
  let count = 0;
  while (set.has(date.toISOString().slice(0, 10))) { count++; date.setDate(date.getDate() - 1); }
  return count;
}

export function useProgress() {
  const [progress, setProgress] = useState<Progress>(empty);
  const [loaded, setLoaded] = useState(false);
  useEffect(() => { setProgress(read()); setLoaded(true); }, []);

  const complete = useCallback((id: string, accuracy: number, xp: number) => {
    const current = read();
    const previous = current.lessons[id];
    const next: Progress = {
      lessons: { ...current.lessons, [id]: { done: true, accuracy: Math.max(accuracy, previous?.accuracy || 0), xp: Math.max(xp, previous?.xp || 0), at: new Date().toISOString() } },
      xp: current.xp + xp,
      days: [...new Set([...current.days, today()])].slice(-120),
    };
    try { localStorage.setItem(KEY, JSON.stringify(next)); } catch { /* Progress is a convenience. */ }
    setProgress(next);
  }, []);

  const resetAll = useCallback(() => {
    try { localStorage.removeItem(KEY); } catch { /* Nothing stored. */ }
    setProgress(empty);
  }, []);

  return { progress, loaded, complete, resetAll };
}
