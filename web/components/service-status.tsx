"use client";
import { useEffect, useState } from "react";
import { API } from "@/lib/learn/api";
import { useLang } from "@/lib/i18n";
type Health = { device: string; sessions: number; max_sessions?: number; busy?: boolean; last_latency_ms?: number | null };
export function ServiceStatus() {
  const { c } = useLang();
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
  return <section className="service-status"><h3>{c("Local inference service", "خدمة الاستدلال المحلي")}</h3>{health ? <>
    <p><strong>{health.device === "cuda" ? "GPU" : "CPU"}</strong> · {health.busy ? c("Decoding", "جارٍ التعرف") : c("Ready", "جاهز")} · {health.sessions} / {health.max_sessions || 2} {c("sessions", "جلسات")}</p>
    <small>{health.last_latency_ms != null ? c(`Last window: ${health.last_latency_ms}ms, including queue wait.`, `آخر نافذة: ${health.last_latency_ms} مللي ثانية، تشمل الانتظار.`) : c("Model loads once. Audio stays local.", "يُحمّل النموذج مرة واحدة. معالجة الصوت محلية.")} {c("No accuracy or tajweed guarantee.", "لا ضمان لصحة النطق أو التجويد.")}</small>
  </> : <p>{c("Service offline or starting. Reading, text quizzes and tafsir still work.", "الخدمة متوقفة أو قيد التشغيل. القراءة والتحديات النصية والتفسير متاحة.")}</p>}</section>;
}
