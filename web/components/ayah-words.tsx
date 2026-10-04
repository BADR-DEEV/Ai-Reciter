import type { Ayah, Result } from "@/lib/types";
import { tajweedSegments, type TajweedAyah, type TajweedMushaf } from "@/lib/tajweed";

export function AyahWords({ ayah, result, tajweed, rules, lang = "en", activeWord = null }: { ayah: Ayah; result?: Result; tajweed?: TajweedAyah; rules?: TajweedMushaf["rules"]; lang?: "en" | "ar"; activeWord?: number | null }) {
  const source = (ayah.displayText || ayah.text).replace(/[\u0660-\u0669\d]+/g, "").trim();
  // Stale/generated annotations may never replace Quran text or color the wrong
  // code points. A different display spelling requires regenerated offsets.
  const annotation = tajweed?.text === source ? tajweed : undefined;
  const displayWords = [...source.matchAll(/\S+/g)];
  return <>{displayWords.map((match, index) => {
    const text = match[0];
    const word = result?.words?.[index];
    const status = word?.status || "pending";
    const label = lang === "ar" ? status === "correct" ? "مطابق للنص" : status === "missed" ? "لم يطابق النص" : "لم تصل إليه" : status === "correct" ? "Matched" : status === "missed" ? "Omitted / not matched" : "Not reached";
    return <span key={index}><span className={`quran-word ${status} ${annotation ? "tajweed-word" : ""} ${activeWord === index ? "playback-word" : ""}`} data-word-index={index} aria-current={activeWord === index ? "true" : undefined} data-status={status} title={`${label}${word?.heard ? ` · ${word.heard}` : ""}`}>
      {annotation && rules ? tajweedSegments(source, annotation.spans, match.index!, match.index! + text.length).map((piece, i) => <span key={i} className={piece.rules.length ? "tajweed-span" : ""} style={{ color: piece.rules.length ? rules[piece.rules[0]]?.color : undefined }} title={piece.rules.map(id => rules[id]?.[lang] || id).join(" · ")}>{piece.text}</span>) : text}
    </span>{" "}</span>;
  })}</>;
}
