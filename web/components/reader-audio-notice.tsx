"use client";
import { useEffect, useState } from "react";
import { Download, Loader2 } from "lucide-react";
import { useLang } from "@/lib/i18n";
import type { ReciterID } from "@/lib/reciters";
import type { ReaderAudioState } from "@/lib/reader-audio";

/** Shown when a clip fails: offers to fetch missing reader recordings, otherwise explains the clip is unavailable. */
export function ReaderAudioNotice({ reciter }: { reciter: ReciterID }) {
  const { c, lang } = useLang();
  const dir = lang === "ar" ? "rtl" : "ltr";
  const [audio, setAudio] = useState<ReaderAudioState | null>(null);
  const [error, setError] = useState(false);
  const [waited, setWaited] = useState(false);
  const busy = audio?.state === "downloading" || audio?.state === "waiting";
  useEffect(() => { if (busy) setWaited(true); }, [busy]);
  useEffect(() => {
    let disposed = false;
    const refresh = () => fetch("/api/reader-audio", { cache: "no-store" }).then(r => r.json()).then(data => { if (!disposed) setAudio(data); }).catch(() => {});
    void refresh();
    const timer = busy ? setInterval(refresh, 5000) : undefined;
    return () => { disposed = true; clearInterval(timer); };
  }, [busy]);
  const start = async () => {
    setError(false);
    try { setAudio(await (await fetch("/api/reader-audio", { method: "POST" })).json()); } catch { setError(true); }
  };
  if (!audio) return null;
  if (!audio.missing.includes(reciter)) {
    if (waited) {
      return <small role="status" dir={dir} className="reader-audio-note">{c("Reader audio is ready. Press Listen again.", "صوت القارئ جاهز. اضغط استمع مرة أخرى.")}</small>;
    }
    return <small role="alert" dir={dir}>{c("This reader’s aligned ayah clip is unavailable. Choose another reader; no audio was substituted.", "لا يتوفر مقطع مواءَم لهذه الآية بصوت القارئ. اختر قارئًا آخر؛ لم نستبدل الصوت.")}</small>;
  }
  if (busy) {
    const progress = audio.total ? ` · ${audio.done}/${audio.total}` : "";
    return <small role="status" dir={dir} className="reader-audio-note"><Loader2 size={13} className="spin" />
      {audio.state === "waiting"
        ? c(`Downloading reader recordings${progress}. Paused for Hugging Face's rate limit; it resumes by itself.`, `جارٍ تنزيل تسجيلات القرّاء${progress}. متوقف مؤقتًا بسبب حدّ Hugging Face وسيستأنف تلقائيًا.`)
        : c(`Downloading reader recordings in the background${progress}. You can keep using the app.`, `جارٍ تنزيل تسجيلات القرّاء في الخلفية${progress}. يمكنك متابعة استخدام التطبيق.`)}</small>;
  }
  return <div className="reader-audio-ask" role="alert" dir={dir}>
    <small>{c("The reader recordings aren’t on this computer yet. Download them now? (about 330 MB from the team’s Hugging Face)", "تسجيلات القرّاء غير موجودة على هذا الجهاز بعد. هل تريد تنزيلها الآن؟ (حوالي 330 ميغابايت من Hugging Face الخاص بالفريق)")}</small>
    {audio.state === "failed" && <small>{c(`Last attempt failed: ${audio.message || "unknown error"}. This needs \`hf auth login\` with access to Mathani-Ayat.`, "فشلت المحاولة السابقة. يتطلب ذلك تسجيل الدخول إلى Hugging Face مع صلاحية الوصول إلى Mathani-Ayat.")}</small>}
    {error && <small>{c("Could not reach the app server.", "تعذّر الاتصال بخادم التطبيق.")}</small>}
    <button className="btn-quiet" onClick={start}><Download size={14} /> {c("Download reader audio", "تنزيل صوت القرّاء")}</button>
  </div>;
}
