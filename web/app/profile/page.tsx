"use client";
import { useEffect, useState } from "react";
import { SiteHeader } from "@/components/learn/site-header";
import { activeProfile, deleteActive, PROFILE_EVENT, register, setActive, signIn, type LocalProfile } from "@/lib/local-profile";
import { streak, useProgress } from "@/lib/learn/progress";

export default function ProfilePage() {
  const [profile, setProfile] = useState<LocalProfile | null>(null);
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [mode, setMode] = useState("create");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const { progress } = useProgress();
  useEffect(() => { const refresh = () => setProfile(activeProfile()); refresh(); window.addEventListener(PROFILE_EVENT, refresh); return () => window.removeEventListener(PROFILE_EVENT, refresh); }, []);
  return <div className="learn-shell"><SiteHeader active="profile" /><main className="course">
    <div className="course-head"><div><p className="hero-kicker">YOUR LOCAL JOURNEY</p><h1>{profile ? `As-salāmu ʿalaykum, ${profile.name}` : "Your practice profile"}</h1><p>Learn Quran reading, not an Arabic-language course. Little steps, at your pace.</p></div></div>
    <p className="safety-note">Competition prototype: profiles live only in this browser. Passwords are salted and hashed, but this is not secure account authentication. Anyone with access to this device can edit storage. Do not reuse a real password. No cloud sync or password recovery.</p>
    {profile ? <section className="challenge-card"><h2>Your progress</h2><div className="profile-stats"><strong>{progress.xp} XP</strong><strong>{streak(progress.days)} day streak</strong><strong>{Object.values(progress.lessons).filter(p => p.done).length} activities</strong></div>
      <a className="btn-primary" href="/learn">Continue learning</a> <a className="btn-quiet" href="/games">Quran challenges</a>
      <div className="profile-actions"><button className="btn-quiet" onClick={() => { try { setActive(null); } catch { setError("Browser storage is unavailable."); } }}>Sign out</button>
      <button className="btn-quiet" onClick={() => { if (window.confirm("Delete this local profile and its progress?")) { try { deleteActive(); } catch { setError("Could not update browser storage."); } } }}>Delete local profile</button></div>
    </section> : <form className="challenge-card profile-form" onSubmit={async e => {
      e.preventDefault(); setError(""); setBusy(true);
      try { setProfile(await (mode === "create" ? register(name, password) : signIn(name, password))); setPassword(""); }
      catch (cause) { setError(cause instanceof Error ? cause.message : "Browser storage is unavailable."); }
      finally { setBusy(false); }
    }}><h2>{mode === "create" ? "Create a local profile" : "Sign in on this browser"}</h2>
      <label>Name<input required minLength={2} maxLength={32} autoComplete="username" value={name} onChange={e => setName(e.target.value)} /></label>
      <label>Demo password<input type="password" required minLength={8} maxLength={128} autoComplete={mode === "create" ? "new-password" : "current-password"} value={password} onChange={e => setPassword(e.target.value)} /></label>
      <button className="btn-primary" disabled={busy}>{busy ? "Saving…" : mode === "create" ? "Create profile" : "Sign in"}</button>
      <button type="button" className="btn-quiet" disabled={busy} onClick={() => { setMode(mode === "create" ? "login" : "create"); setError(""); }}>{mode === "create" ? "I have a profile" : "Create a profile instead"}</button>
    </form>}{error && <p role="alert">{error}</p>}
  </main></div>;
}
