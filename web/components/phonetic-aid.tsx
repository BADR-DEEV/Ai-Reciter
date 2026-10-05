import { qaloonG2P } from "@/lib/qaloon-g2p";
import { useLang } from "@/lib/i18n";

export function PhoneticAid({ text, activeWord = null }: { text: string; activeWord?: number | null }) {
  const { c } = useLang();
  const result = qaloonG2P(text);
  return <div className="phonetic-aid" lang="en" dir="ltr">
    <p>{result.words.map((word, index) => <span key={index}><span className={activeWord === index ? "playback-word" : ""} data-word-index={index} aria-current={activeWord === index ? "true" : undefined}>{word || "_____"}</span>{" "}</span>)}</p>
    <small>{c("Qālūn source · draft reading aid · pause at the end", "نص قالون · نقل صوتي تجريبي · وقف في نهاية الآية")}</small>
    <details><summary>{c("Pronunciation key & review notes", "دليل النطق وملاحظات المراجعة")}</summary>
      <p>{c("ā / ī / ū: long vowels; doubled letters: hold the consonant; ʿ: ʿayn; ʾ: hamza; ḥ / ṣ / ḍ / ṭ / ẓ have no exact English equivalents.", "ā / ī / ū: أصوات طويلة؛ تكرار الحرف للتشديد؛ ʿ: عين؛ ʾ: همزة؛ ḥ / ṣ / ḍ / ṭ / ẓ ليس لها مقابل إنجليزي دقيق.")}</p>
      <p>{c("This spelling cannot teach tajweed, nasalization or permitted Qālūn variants. Listen to a verified Qālūn recitation and consult a teacher. Not scholar-approved.", "لا يعلّم هذا النقل أحكام التجويد والغنة وأوجه قالون المسموح بها. استمع لتلاوة موثوقة بقالون واستعن بمعلم. لم يعتمده متخصص.")}</p>
      {result.warnings.map(note => <p key={note}>{note}</p>)}
    </details>
  </div>;
}
