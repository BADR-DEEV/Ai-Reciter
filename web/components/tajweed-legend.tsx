"use client";
import { useLang } from "@/lib/i18n";
import type { TajweedMushaf } from "@/lib/tajweed";
export function TajweedLegend({ rules }: { rules: TajweedMushaf["rules"] }) {
  const { lang, c } = useLang();
  return <details className="tajweed-legend"><summary>{c("Draft tajweed color key · counts, not seconds", "مفتاح ألوان التجويد التجريبي · حركات وليست ثواني")}</summary>
    <p>{c("Not a certified mujawwad mushaf. Natural madd is 2 harakat; necessary madd is 6. Other lengths depend on rule, stopping and Qālūn route—not a universal 4. No teacher-approved annotations yet.", "ليس مصحفًا مجودًا معتمدًا. المد الطبيعي حركتان واللازم ست حركات. غيرهما يتوقف على الحكم والوقف وطريق قالون، وليس دائمًا أربعًا. لم يعتمد معلم هذه العلامات بعد.")}</p>
    <ul>{Object.entries(rules).map(([id, rule]) => <li key={id}><i style={{ background: rule.color }} />{rule[lang]}</li>)}</ul>
    <p>{c("Connected reading inside each ayah, pause at its end. Internal waqf, rā/Allah-lām heaviness, mīm al-jam and ṣilah need route-aware review. Speech-match backgrounds are separate from these letter colors.", "الوصل داخل الآية والوقف في نهايتها. الوقف الداخلي وتفخيم الراء ولام الجلالة وميم الجمع والصلة تحتاج مراجعة بحسب الطريق. خلفية مطابقة التلاوة منفصلة عن ألوان الحروف.")}</p>
  </details>;
}
