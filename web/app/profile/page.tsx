"use client";
import { useEffect, useState } from "react";
import { SiteHeader } from "@/components/learn/site-header";
import { activeProfile, deleteActive, PROFILE_EVENT, register, setActive, signIn, type LocalProfile } from "@/lib/local-profile";
import { streak, useProgress } from "@/lib/learn/progress";
import { useLang } from "@/lib/i18n";

export default function ProfilePage() {
  const { lang, c } = useLang();
  const [profile, setProfile] = useState<LocalProfile | null>(null);
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [mode, setMode] = useState("create");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const { progress } = useProgress();
  useEffect(() => { const refresh = () => setProfile(activeProfile()); refresh(); window.addEventListener(PROFILE_EVENT, refresh); return () => window.removeEventListener(PROFILE_EVENT, refresh); }, []);
  return <div className="learn-shell" dir={lang === "ar" ? "rtl" : "ltr"}><SiteHeader active="profile" /><main className="course">
    <div className="course-head"><div><p className="hero-kicker">{c("YOUR LOCAL JOURNEY", "رحلتك المحلية")}</p><h1>{profile ? c(`As-salāmu ʿalaykum, ${profile.name}`, `السلام عليكم، ${profile.name}`) : c("Your practice profile", "ملف التدريب")}</h1><p>{c("Learn Quran reading, not an Arabic-language course. Little steps, at your pace.", "تعلّم قراءة القرآن، لا دورة في اللغة العربية. خطوة بخطوة على مهلك.")}</p></div></div>
    <p className="safety-note">{c("Competition prototype: profiles live only in this browser. Passwords are salted and hashed, but this is not secure account authentication. Anyone with access to this device can edit storage. Do not reuse a real password. No cloud sync or password recovery.", "نموذج أولي: الملفات في هذا المتصفح فقط. كلمات المرور مُملّحة ومجزّأة، لكن هذا ليس تسجيل دخول آمنًا؛ يمكن لمن يصل للجهاز تعديل التخزين. لا تستخدم كلمة مرور حقيقية. لا مزامنة سحابية ولا استعادة لكلمة المرور.")}</p>
    {profile ? <section className="challenge-card"><h2>{c("Your progress", "تقدّمك")}</h2><div className="profile-stats"><strong>{progress.xp} {c("XP", "نقطة")}</strong><strong>{streak(progress.days)} {c("day streak", "أيام متتابعة")}</strong><strong>{Object.values(progress.lessons).filter(p => p.done).length} {c("activities", "أنشطة")}</strong></div>
      <a className="btn-primary" href="/learn">{c("Continue learning", "واصل التعلّم")}</a> <a className="btn-quiet" href="/games">{c("Quran challenges", "تحديات القرآن")}</a>
      <div className="profile-actions"><button className="btn-quiet" onClick={() => { try { setActive(null); } catch { setError("Browser storage is unavailable."); } }}>{c("Sign out", "تسجيل الخروج")}</button>
      <button className="btn-quiet" onClick={() => { if (window.confirm(c("Delete this local profile and its progress?", "هل تريد حذف هذا الملف المحلي وتقدّمه؟"))) { try { deleteActive(); } catch { setError("Could not update browser storage."); } } }}>{c("Delete local profile", "احذف الملف المحلي")}</button></div>
    </section> : <form className="challenge-card profile-form" onSubmit={async e => {
      e.preventDefault(); setError(""); setBusy(true);
      try { setProfile(await (mode === "create" ? register(name, password) : signIn(name, password))); setPassword(""); }
      catch (cause) { setError(cause instanceof Error ? cause.message : "Browser storage is unavailable."); }
      finally { setBusy(false); }
    }}><h2>{mode === "create" ? c("Create a local profile", "أنشئ ملفًا محليًا") : c("Sign in on this browser", "سجّل الدخول في هذا المتصفح")}</h2>
      <label>{c("Name", "الاسم")}<input required minLength={2} maxLength={32} autoComplete="username" value={name} onChange={e => setName(e.target.value)} /></label>
      <label>{c("Demo password", "كلمة مرور تجريبية")}<input type="password" required minLength={8} maxLength={128} autoComplete={mode === "create" ? "new-password" : "current-password"} value={password} onChange={e => setPassword(e.target.value)} /></label>
      <button className="btn-primary" disabled={busy}>{busy ? c("Saving…", "جارٍ الحفظ…") : mode === "create" ? c("Create profile", "أنشئ الملف") : c("Sign in", "تسجيل الدخول")}</button>
      <button type="button" className="btn-quiet" disabled={busy} onClick={() => { setMode(mode === "create" ? "login" : "create"); setError(""); }}>{mode === "create" ? c("I have a profile", "لديّ ملف") : c("Create a profile instead", "أنشئ ملفًا بدلًا من ذلك")}</button>
    </form>}{error && <p role="alert">{c(error, "تعذّر الدخول أو الحفظ. تحقّق من الاسم وكلمة المرور وإتاحة التخزين المحلي.")}</p>}
  </main></div>;
}
