"use client";
import { useAvailableModels } from "@/lib/models";
import { useLang } from "@/lib/i18n";
export function ServiceStatus() {
  const { c } = useLang();
  const { health, models } = useAvailableModels();
  const label = (name: string) => name === "plain" ? c("Recognition", "التعرّف") : name === "tajweed" ? c("Tajweed", "التجويد") : name;
  return <section className="service-status"><h3>{c("Local inference service", "خدمة الاستدلال المحلي")}</h3>{health ? <>
    <p><strong>{health.device === "cuda" ? "GPU" : "CPU"}</strong> · {health.busy ? c("Decoding", "جارٍ التعرف") : c("Ready", "جاهز")} · {health.sessions} / {health.max_sessions || 2} {c("sessions", "جلسات")}</p>
    {health.models && <p>{c("Models", "النماذج")}: {models.map(m => m.available ? label(m.name) : `${label(m.name)} ${c("(unavailable)", "(غير متاح)")}`).join(" · ")}</p>}
    <small>{health.last_latency_ms != null ? c(`Last window: ${health.last_latency_ms}ms, including queue wait.`, `آخر نافذة: ${health.last_latency_ms} مللي ثانية، تشمل الانتظار.`) : c("Model loads once. Audio stays local.", "يُحمّل النموذج مرة واحدة. معالجة الصوت محلية.")} {c("No accuracy or tajweed guarantee.", "لا ضمان لصحة النطق أو التجويد.")}</small>
  </> : <p>{c("Service offline or starting. Reading, text quizzes and tafsir still work.", "الخدمة متوقفة أو قيد التشغيل. القراءة والتحديات النصية والتفسير متاحة.")}</p>}</section>;
}
