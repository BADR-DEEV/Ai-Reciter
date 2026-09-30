import type { Ayah, Result } from "@/lib/types";

export function AyahWords({ ayah, result }: { ayah: Ayah; result?: Result }) {
  const displayWords = (ayah.displayText || ayah.text).replace(/[\u0660-\u0669\d]+/g, "").trim().split(/\s+/);
  return <>{displayWords.map((text, index) => {
    const word = result?.words?.[index];
    const status = word?.status || "pending";
    const label = status === "correct" ? "Matched" : status === "missed" ? "Omitted / not matched" : "Not reached";
    return <span key={index}><span className={`quran-word ${status}`} data-word-index={index} data-status={status} title={`${label}${word?.heard ? ` · heard: ${word.heard}` : ""}`}>{text}</span>{" "}</span>;
  })}</>;
}
