"use client";

import { useLang } from "@/lib/i18n";
import { harakatLabel, type TajweedRules, type TajweedSettings } from "@/lib/tajweed";

/** Colour key with per-group switches and the full rule list. */
export function TajweedLegend({ rules, settings, update }: {
  rules: TajweedRules; settings: TajweedSettings; update: (patch: Partial<TajweedSettings>) => void;
}) {
  const { lang, c } = useLang();
  const groups = Object.entries(rules.groups).filter(([id]) => settings.details || Object.values(rules.rules).some(r => r.group === id && r.level === "core"));
  const toggle = (id: string) => update({ hidden: settings.hidden.includes(id) ? settings.hidden.filter(g => g !== id) : [...settings.hidden, id] });
  return <section className="tajweed-legend" aria-label={c("Tajweed color key", "مفتاح ألوان التجويد")}>
    <h3>{c("Tajweed colors · Qālūn ʿan Nāfiʿ", "ألوان التجويد · قالون عن نافع")}</h3>
    <ul className="tajweed-palette">{groups.map(([id, group]) => <li key={id} data-color-group={id}>
      <button aria-pressed={!settings.hidden.includes(id)} onClick={() => toggle(id)} className={settings.hidden.includes(id) ? "off" : ""}>
        <i style={{ background: group.light }} aria-hidden="true" /><span>{group[lang]}</span></button></li>)}</ul>
    <label className="tajweed-details-switch"><input type="checkbox" checked={settings.details} onChange={e => update({ details: e.target.checked })} />
      {c("Show every detail (natural madd, heavy and light letters, clear letters)", "أظهر كل التفاصيل (المد الطبيعي، التفخيم والترقيق، الإظهار)")}</label>
    <details><summary>{c("What each color means", "معنى كل لون")}</summary>
      {Object.entries(rules.groups).map(([gid, group]) => {
        const members = rules.order.filter(id => rules.rules[id].group === gid && (settings.details || rules.rules[id].level === "core"));
        return members.length > 0 && <div key={gid} className="tajweed-rule-group"><h4><i style={{ background: group.light }} />{group[lang]}</h4>
          <ul className="tajweed-rule-notes">{members.map(id => { const rule = rules.rules[id]; return <li key={id}>
            <strong>{rule[lang]}</strong>{rule.harakat && <span className="tajweed-count">{harakatLabel(rule, lang)}</span>}
            <span>{lang === "ar" ? rule.explain_ar : rule.explain_en}</span></li>; })}</ul></div>;
      })}
    </details>
    <small>{c("Tap a colored letter for its rule. Counts are ḥarakāt (beats), not seconds. Rules from", "انقر على الحرف الملوّن لمعرفة حكمه. المقادير بالحركات لا بالثواني. الأحكام من")} <bdi lang="ar" dir="rtl">«{rules.source.title_ar}»</bdi> ({rules.source.edition}). {c("Applied by software: confirm with a qualified Qālūn teacher.", "طُبّقت آليًا؛ راجعها مع معلم متقن لرواية قالون.")} <a href="/tajweed">{c("Qālūn topics and letter sounds →", "مواضيع الرواية ومخارج الحروف ←")}</a></small>
  </section>;
}
