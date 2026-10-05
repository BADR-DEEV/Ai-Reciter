import type { Ayah, Result } from "@/lib/types";
import { visibleRules, wordText, type Segment, type TajweedAyah, type TajweedRules, type TajweedSettings } from "@/lib/tajweed";

/** Words of an ayah. With generated tajweed data the text is the full Qālūn
 *  spelling (ṣilah, iqlāb mīm, tas-hīl dots); colours only change `color` on
 *  inline spans inside a word, so Arabic shaping and word widths are kept. */
export function AyahWords({ ayah, result, tajweed, rules, settings, missed, lang = "en", activeWord = null }: {
  ayah: Ayah; result?: Result; tajweed?: TajweedAyah | { w: Segment[][] }; rules?: TajweedRules | null; settings?: TajweedSettings;
  missed?: Map<number, Set<string>>; lang?: "en" | "ar"; activeWord?: number | null;
}) {
  const fallback = (ayah.displayText || ayah.text).replace(/[٠-٩\d]+/g, "").trim().split(/\s+/).map(w => [[w]] as Segment[]);
  const words = tajweed?.w.length === fallback.length ? tajweed.w : fallback;
  const colored = Boolean(settings?.show && rules && words !== fallback);
  return <>{words.map((segments, index) => {
    const word = result?.words?.[index];
    const status = word?.status || "pending";
    const label = lang === "ar" ? status === "correct" ? "مطابق للنص" : status === "missed" ? "لم يطابق النص" : "لم تصل إليه" : status === "correct" ? "Matched" : status === "missed" ? "Omitted / not matched" : "Not reached";
    const misses = missed?.get(index);
    return <span key={index}><span className={`quran-word ${status} ${colored ? "tajweed-word" : ""} ${activeWord === index ? "playback-word" : ""}`} data-word-index={index} aria-current={activeWord === index ? "true" : undefined} data-status={status} title={`${label}${word?.heard ? ` · ${word.heard}` : ""}`}>
      {colored ? segments.map((segment, i) => {
        const shown = visibleRules(segment[1], rules!, settings!);
        if (!shown.length) return segment[0];
        const miss = misses && segment[1]!.some(r => misses.has(rules!.rules[rules!.order[r]]?.tag || ""));
        return <span key={i} className={`tajweed-span tj-${rules!.rules[shown[0]].group}${miss ? " tj-miss" : ""}`} data-tj={segment[1]!.join(",")} data-w={index}>{segment[0]}</span>;
      }) : wordText(segments)}
    </span>{" "}</span>;
  })}</>;
}
