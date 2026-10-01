import { Fragment } from "react";

const ARABIC_RUN = /([\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+(?:[\s\u200C\u200D]+[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+)*)/;

/** English text with Arabic runs isolated (correct bidi) and set in the Arabic font. */
export function Rich({ text }: { text: string }) {
  return <>{text.split(ARABIC_RUN).map((part, i) => i % 2
    ? <bdi key={i} lang="ar" className="ar-inline">{part}</bdi>
    : <Fragment key={i}>{part}</Fragment>)}</>;
}
