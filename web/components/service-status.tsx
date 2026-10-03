"use client";
import { useEffect, useState } from "react";
import { API } from "@/lib/learn/api";
type Health = { device: string; sessions: number; max_sessions?: number; busy?: boolean; last_latency_ms?: number | null };
export function ServiceStatus() {
  const [health, setHealth] = useState<Health | null>(null);
  useEffect(() => {
    let disposed = false;
    const controller = new AbortController();
    const refresh = async () => {
      if (document.hidden) return;
      try {
        const response = await fetch(`${API}/health`, { signal: AbortSignal.any([controller.signal, AbortSignal.timeout(5000)]), cache: "no-store" });
        if (!response.ok) throw new Error();
        const data = await response.json();
        if (!disposed) setHealth(data);
      } catch { if (!disposed) setHealth(null); }
    };
    void refresh(); const timer = setInterval(() => void refresh(), 20000);
    return () => { disposed = true; controller.abort(); clearInterval(timer); };
  }, []);
  return <section className="service-status"><h3>Local inference service</h3>{health ? <>
    <p><strong>{health.device === "cuda" ? "GPU" : "CPU"}</strong> · {health.busy ? "Decoding" : "Ready"} · {health.sessions} / {health.max_sessions || 2} sessions</p>
    <small>{health.last_latency_ms != null ? `Last window: ${health.last_latency_ms}ms, including queue wait.` : "Model loads once. Audio stays local."} No accuracy or tajweed guarantee.</small>
  </> : <p>Service offline or starting. Reading, text quizzes and tafsir still work.</p>}</section>;
}
