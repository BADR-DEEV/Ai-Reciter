import { memo } from "react";
import type { Ayah, Result } from "@/lib/types";
import { visibleRules, wordText, type Segment, type TajweedAyah, type TajweedRules, type TajweedSettings } from "@/lib/tajweed";
import { missedByWord, type TajweedFeedback } from "@/components/tajweed-feedback";

type Props = {
  ayah: Ayah; result?: Result; tajweed?: TajweedAyah | { w: Segment[][] }; rules?: TajweedRules | null; settings?: TajweedSettings;
  feedback?: TajweedFeedback; lang?: "en" | "ar"; activeWord?: number | null;
};

/** Words of an ayah. With generated tajweed data the text is the full Qālūn
 *  spelling (ṣilah, iqlāb mīm, tas-hīl dots); colours only change `color` on
 *  inline spans inside a word, so Arabic shaping and word widths are kept. */
function Words({ ayah, result, tajweed, rules, settings, feedback, lang = "en", activeWord = null }: Props) {
  const fallback = (ayah.displayText || ayah.text).replace(/[٠-٩\d]+/g, "").trim().split(/\s+/).map(w => [[w]] as Segment[]);
  const words = tajweed?.w.length === fallback.length ? tajweed.w : fallback;
  const colored = Boolean(settings?.show && rules && words !== fallback);
  const missed = missedByWord(feedback);
  return <>{words.map((segments, index) => {
    const word = result?.words?.[index];
    const status = word?.status || "pending";
    const label = lang === "ar" ? status === "correct" ? "مطابق للنص" : status === "missed" ? "لم يطابق النص" : "لم تصل إليه" : status === "correct" ? "Matched" : status === "missed" ? "Omitted / not matched" : "Not reached";
    const misses = missed.get(index);
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

// Live recitation sends fresh result objects for every ayah on each update;
// only re-render an ayah when what it shows actually changed.
const resultKey = (r?: Result) => r?.words?.map(w => `${w.status}:${w.heard || ""}`).join("|") || "";
const feedbackKey = (f?: TajweedFeedback) => f?.words.map(w => `${w.word}:${w.missed.join(",")}`).join("|") || "";
export const AyahWords = memo(Words, (a, b) => a.ayah === b.ayah && a.tajweed === b.tajweed && a.rules === b.rules
  && a.settings === b.settings && a.lang === b.lang && a.activeWord === b.activeWord
  && resultKey(a.result) === resultKey(b.result) && feedbackKey(a.feedback) === feedbackKey(b.feedback));
