"use client";
import { useLang } from "@/lib/i18n";
import { RECITERS, type ReciterID } from "@/lib/reciters";
export function ReciterSelector({ value, onChange, disabled = false }: { value: ReciterID; onChange: (id: ReciterID) => void; disabled?: boolean }) {
  const { lang, c } = useLang();
  return <label className="reciter-select">{c("Reference reciter", "قارئ الاستماع")}<select aria-label={c("Reference reciter", "قارئ الاستماع")} value={value} disabled={disabled} onChange={e => onChange(e.target.value as ReciterID)}>{RECITERS.map(r => <option key={r.id} value={r.id}>{lang === "ar" ? r.arabic : r.name} · {c("Qālūn", "قالون")}</option>)}</select></label>;
}
