"use client";
import { useLang } from "@/lib/i18n";
import type { TajweedMushaf } from "@/lib/tajweed";
export function TajweedLegend({ rules, legend }: { rules: TajweedMushaf["rules"]; legend?: TajweedMushaf["legend"] }) {
  const { lang, c } = useLang();
  return <section className="tajweed-legend" aria-label={c("Tajweed color annotations", "شرح ألوان التجويد")}>
    <h3>{c("Tajweed color key · Qālūn reading", "مفتاح ألوان التجويد · قراءة قالون")}</h3>
    <ul className="tajweed-palette">{Object.entries(legend || rules).map(([id, rule]) => <li key={id} data-color-group={id}><i style={{ background: rule.color }} aria-hidden="true" /><span>{rule[lang]}</span></li>)}</ul>
    <details><summary>{c("Qālūn counts, context & review notes", "حركات قالون والسياق وملاحظات المراجعة")}</summary>
    <p>{c("Not a certified mujawwad mushaf. Natural madd is 2 harakat; necessary madd is 6. Other lengths depend on rule, stopping and Qālūn route—not a universal 4. No teacher-approved annotations yet.", "ليس مصحفًا مجودًا معتمدًا. المد الطبيعي حركتان واللازم ست حركات. غيرهما يتوقف على الحكم والوقف وطريق قالون، وليس دائمًا أربعًا. لم يعتمد معلم هذه العلامات بعد.")}</p>
    <p>{c("The sample image supplies the palette, not the recitation-specific lengths. Qālūn from Shāṭibiyyah: ordinary unchanged connected madd 4; separate madd 2 or 4, with consistent choices. Changed adjacent hamzas can alter the madd cause; those combinations are not given a blanket length.", "الصورة مثال للألوان وليست مصدرًا لأطوال المد الخاصة بالرواية. قالون من الشاطبية: المتصل العادي مع همزة غير متغيرة أربع حركات، والمنفصل حركتان أو أربع مع التزام الوجه. تغير الهمزتين المتجاورتين قد يغير سبب المد؛ لا نضع طولًا واحدًا عامًا لهذه الأوجه.")}</p>
    <ul className="tajweed-rule-notes">{Object.entries(rules).map(([id, rule]) => <li key={id}><i style={{ background: rule.color }} />{rule[lang]}</li>)}</ul>
    <p>{c("Connected reading inside each ayah, pause at its end. Internal waqf, rā/Allah-lām heaviness, mīm al-jam and ṣilah need route-aware review. Speech-match backgrounds are separate from these letter colors.", "الوصل داخل الآية والوقف في نهايتها. الوقف الداخلي وتفخيم الراء ولام الجلالة وميم الجمع والصلة تحتاج مراجعة بحسب الطريق. خلفية مطابقة التلاوة منفصلة عن ألوان الحروف.")}</p>
    <p>{c("Source access: the supplied Scribd document is blocked by a client challenge. Public secondary Qālūn reference: The Secure Way to Rewayat Qalun, pp. 8–14 and 18–22. Exact-source and teacher review remain required.", "المصادر: تعذّر قراءة وثيقة Scribd المرفقة بسبب حاجز التحقق. راجعنا مرجعًا عامًا ثانويًا: The Secure Way to Rewayat Qalun، الصفحات ٨–١٤ و١٨–٢٢. ما زالت مراجعة الوثيقة الأصلية والمعلم مطلوبة.")} <a href="https://archive.org/details/UsulRewayatQalun" target="_blank" rel="noopener noreferrer">{c("Reference", "المرجع")}</a></p>
    </details><small>{c("Unapproved annotations · counts are harakat, not seconds · hover a colored letter for its rule", "علامات غير معتمدة · المقادير بالحركات لا بالثواني · مرّر المؤشر على الحرف الملوّن لشرح الحكم")}</small>
  </section>;
}
