"use client";

import { useEffect, useMemo, useState } from "react";
import { BookOpen } from "lucide-react";
import { SiteHeader } from "@/components/learn/site-header";
import { useLang } from "@/lib/i18n";
import { groupStyle, harakatLabel, loadRules, type Segment, type TajweedRules } from "@/lib/tajweed";
import { AREAS, LETTERS, LETTER_NAMES, PLACES, QUALITIES, placeOf, qualitiesOf } from "@/lib/tajweed-letters";

function ColoredWord({ segments, rules }: { segments: Segment[]; rules: TajweedRules }) {
  return <span className="tj-example" lang="ar" dir="rtl">{segments.map((s, i) => {
    const id = s[1]?.length ? rules.order[s[1][0]] : null;
    return id ? <span key={i} className={`tajweed-span tj-${rules.rules[id].group}`}>{s[0]}</span> : s[0];
  })}</span>;
}

const studioLink = (surah: number, ayah: number) => `/studio?surah=${surah}&ayah=${ayah}`;

export default function TajweedPage() {
  const { lang, c } = useLang();
  const [rules, setRules] = useState<TajweedRules | null>(null);
  const [error, setError] = useState(false);
  const [letter, setLetter] = useState("ض");
  useEffect(() => { loadRules().then(setRules).catch(() => setError(true)); }, []);
  const place = placeOf(letter);
  const groups = useMemo(() => rules ? Object.entries(rules.groups).map(([id, group]) => ({ id, group, members: rules.order.filter(r => rules.rules[r].group === id) })) : [], [rules]);

  return <div className="app-shell studio-shell" dir={lang === "ar" ? "rtl" : "ltr"}>
    <div className="workspace">
      <SiteHeader active="tajweed" />
      <main className="tajweed-page" style={groupStyle(rules) as React.CSSProperties}>
        <section className="tj-hero">
          <div className="eyebrow">{c("QĀLŪN ʿAN NĀFIʿ · ṬARĪQ AL-SHĀṬIBIYYAH", "رواية قالون عن نافع · من طريق الشاطبية")}</div>
          <h1>{c("Tajweed, the Qālūn way", "التجويد على رواية قالون")}</h1>
          <p>{c("How each letter is made, every tajweed rule with its Qālūn count, and the topics that make Qālūn's reading distinct. Rules follow the Libyan Awqaf curriculum", "مخارج الحروف وصفاتها، وأحكام التجويد بمقاديرها عند قالون، ومواضيع الرواية التي يتميز بها. الأحكام وفق منهج الهيئة العامة للأوقاف الليبية")} <bdi lang="ar" dir="rtl">«المنهج العلمي في أحكام التجويد وأصول رواية الإمام قالون»</bdi> (2022).</p>
          <nav className="tj-jump" aria-label={c("Sections", "الأقسام")}><a href="#letters">{c("Letter sounds", "مخارج الحروف وصفاتها")}</a><a href="#rules">{c("Tajweed rules", "أحكام التجويد")}</a><a href="#riwaya">{c("Qālūn topics", "مواضيع الرواية")}</a><a href="/studio">{c("Practise in the studio →", "تدرّب في الاستوديو ←")}</a></nav>
        </section>

        <section id="letters" className="tj-section">
          <h2>{c("1 · Where each letter comes from", "١ · مخارج الحروف وصفاتها")}</h2>
          <p className="tj-lead">{c("A letter is defined by its place (makhraj) and its qualities (ṣifāt). Pick a letter.", "يتميز الحرف بمخرجه وصفاته. اختر حرفًا.")}</p>
          <div className="tj-letter-grid" role="listbox" aria-label={c("Arabic letters", "الحروف العربية")}>
            {LETTERS.map(l => <button key={l} role="option" aria-selected={l === letter} className={l === letter ? "chosen" : ""} onClick={() => setLetter(l)}>
              <span lang="ar">{l}</span><small>{LETTER_NAMES[l]}</small></button>)}
          </div>
          <div className="tj-letter-card" aria-live="polite">
            <div className="tj-letter-big" lang="ar">{letter}</div>
            <div>
              <h3>{LETTER_NAMES[letter]} · {AREAS[place.area][lang]}</h3>
              <p><strong>{lang === "ar" ? place.ar : place.en}</strong>{lang === "en" && <span lang="ar"> ({place.ar})</span>}</p>
              <p>{place.tip}</p>
              <ul className="tj-chips">{qualitiesOf(letter).map(q => <li key={q.id} title={q.explain}>{lang === "ar" ? q.ar : q.en}</li>)}</ul>
            </div>
          </div>
          <details className="tj-places"><summary>{c("All 17 articulation points", "المخارج السبعة عشر")}</summary>
            {Object.entries(AREAS).map(([area, label]) => <div key={area}><h4>{label[lang]}</h4><ul>{PLACES.filter(p => p.area === area).map(p =>
              <li key={p.id}><span lang="ar" className="tj-place-letters">{p.letters.split("").join(" ")}</span><span>{lang === "ar" ? p.ar : p.en}</span></li>)}</ul></div>)}
          </details>
          <details className="tj-places"><summary>{c("The qualities (ṣifāt)", "صفات الحروف")}</summary>
            <ul>{QUALITIES.map(q => <li key={q.id}><span lang="ar" className="tj-place-letters">{q.letters.split("").join(" ")}</span><span><strong>{lang === "ar" ? q.ar : q.en}</strong> · {q.explain}</span></li>)}</ul>
          </details>
        </section>

        <section id="rules" className="tj-section">
          <h2>{c("2 · Tajweed rules and their colors", "٢ · أحكام التجويد وألوانها")}</h2>
          {error && <p role="alert">{c("Tajweed data is not generated yet. Start the app server once, or run python -m src.tajweed.build.", "بيانات التجويد لم تُنشأ بعد. شغّل خادم التطبيق مرة، أو نفّذ python -m src.tajweed.build.")}</p>}
          {rules && groups.map(({ id, group, members }) => <div key={id} className="tj-group">
            <h3><i style={{ background: group.light }} aria-hidden="true" />{group[lang]}</h3>
            <ul>{members.map(rid => { const rule = rules.rules[rid]; return <li key={rid} id={`rule-${rid}`}>
              <div className="tj-rule-head"><strong>{rule[lang]}</strong>{lang === "en" && <span lang="ar">{rule.ar}</span>}
                {rule.harakat && <span className="tajweed-count">{harakatLabel(rule, lang)}</span>}
                {rule.page && <span className="tj-page">{c("book p.", "ص")} {rule.page}</span>}
                <span className="tj-page">{(rules.counts[rid] || 0).toLocaleString(lang)} {c("times", "موضعًا")}</span></div>
              <p>{lang === "ar" ? rule.explain_ar : rule.explain_en}</p>
              {rule.wajh && <p className="tj-wajh">{c("Qālūn's ways:", "الأوجه لقالون:")} {rule.wajh.join(" · ")}</p>}
              {rules.examples[rid]?.length > 0 && <div className="tj-examples">{rules.examples[rid].map(([s, a, , segments]) =>
                <a key={`${s}:${a}`} href={studioLink(s, a)}><ColoredWord segments={segments} rules={rules} /><small>{s}:{a}</small></a>)}</div>}
            </li>; })}</ul>
          </div>)}
        </section>

        <section id="riwaya" className="tj-section">
          <h2>{c("3 · Qālūn's riwāyah topics", "٣ · مواضيع رواية قالون")}</h2>
          <p className="tj-lead">{c("Chapter 3 of the curriculum: the principles (uṣūl) where Qālūn's reading has its own way. Each topic lists where it happens; open one in the studio to see it colored in purple.", "الفصل الثالث من المنهج: الأصول التي لقالون فيها مذهب خاص، مع مواضعها في المصحف؛ افتحها في الاستوديو لتراها ملوّنة بالبنفسجي.")}</p>
          {rules?.topics.map(topic => <details key={topic.id} className="tj-topic" id={`topic-${topic.id}`}>
            <summary><BookOpen size={16} /><span>{lang === "ar" ? topic.ar : topic.en}</span>{lang === "en" && <span lang="ar" className="tj-topic-ar">{topic.ar}</span>}
              {topic.page && <span className="tj-page">{c("p.", "ص")} {topic.page}</span>}{topic.total > 0 && <span className="mini-pill">{topic.total.toLocaleString(lang)}</span>}</summary>
            <p>{lang === "ar" ? topic.summary_ar : topic.summary_en}</p>
            {topic.points.length > 0 && <ul className="tj-points">{topic.points.map(([en, ar]) => <li key={en}>{lang === "ar" ? ar : en}</li>)}</ul>}
            {topic.found.length > 0 && <div className="tj-found"><h4>{c(`Where it happens (${topic.total === topic.found.length ? topic.total : `first ${topic.found.length} of ${topic.total}`})`, `مواضعه (${topic.total === topic.found.length ? topic.total : `أول ${topic.found.length} من ${topic.total}`})`)}</h4>
              <div>{topic.found.map(([s, a, w, text, rule], i) => <a key={i} href={studioLink(s, a)} title={rules.rules[rule]?.[lang]} data-word={w}><span lang="ar">{text}</span><small>{s}:{a}</small></a>)}</div></div>}
          </details>)}
        </section>
        <footer className="footer"><span>{c("Machine-applied from the written text. A qualified Qālūn teacher has the final word.", "طُبّقت الأحكام آليًا من النص المكتوب، والقول الفصل لمعلم متقن لرواية قالون.")}</span></footer>
      </main>
    </div>
  </div>;
}
