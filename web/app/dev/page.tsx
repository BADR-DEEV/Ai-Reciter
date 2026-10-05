import { notFound } from "next/navigation";
import { SiteHeader } from "@/components/learn/site-header";
import { DEV_MODE } from "@/lib/dev-mode";
import { devInventory, formatBytes } from "@/lib/dev-inventory";
import "../dev.css";

export const dynamic = "force-dynamic";
export const metadata = { title: "Rattil · Development" };

const pct = (v?: number) => typeof v === "number" ? `${(v * 100).toFixed(2)}%` : "—";

export default async function DevPage() {
  if (!DEV_MODE) notFound();
  const { root, service, models, data, env } = await devInventory();
  const active = typeof service.data?.model === "string" ? service.data.model : null;
  const groups = [...new Set(data.map(d => d.group))];
  const missing = data.filter(d => !d.present).length;
  return <div className="learn-shell"><SiteHeader active="dev" /><main className="course dev-page">
    <div className="course-head"><div>
      <p className="hero-kicker">DEVELOPMENT MODE</p>
      <h1>What is running</h1>
      <p>Live model service, local model files and every data source the app reads. Shown only when <code>NEXT_PUBLIC_DEV_MODE=1</code>. Reload to refresh.</p>
    </div></div>

    <section className="dev-card">
      <h2><span className={`dev-dot ${service.data ? "ok" : "bad"}`} /> Inference service</h2>
      <p className="dev-sub"><code>{service.url}/health</code></p>
      {service.data ? <dl className="dev-grid">{Object.entries(service.data).map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{v === null ? "—" : String(v)}</dd></div>)}</dl>
        : <p className="dev-warn">Offline ({service.error}). Start it from the repo root: <code>python -m src.streaming.serve --model gpu-full-base</code></p>}
    </section>

    <section className="dev-card">
      <h2>Models</h2>
      <div className="dev-models">{models.map(m => <article key={m.preset} className={`dev-model${active === m.preset ? " active" : ""}`}>
        <header><strong>{m.preset}</strong>{active === m.preset ? <span className="dev-tag ok">loaded</span> : <span className={`dev-tag ${m.present ? "" : "bad"}`}>{m.present ? "on disk" : "missing"}</span>}</header>
        <p className="dev-sub">{m.kind}</p>
        <dl className="dev-list">
          <div><dt>Path</dt><dd><code>{m.path}</code></dd></div>
          <div><dt>Weights</dt><dd>{m.weightsBytes === null ? "—" : formatBytes(m.weightsBytes)}</dd></div>
          {m.specs.map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}
          <div><dt>Source</dt><dd>{m.source}</dd></div>
          <div><dt>Start</dt><dd><code>{m.startWith}</code></dd></div>
        </dl>
        {m.scores.length > 0 && <table className="dev-table"><thead><tr><th>Split</th><th>Clips</th><th>WER</th><th>CER</th></tr></thead>
          <tbody>{m.scores.map(s => <tr key={s.split}><td>{s.split}</td><td>{s.samples ?? "—"}</td><td>{pct(s.wer)}</td><td>{pct(s.cer)}</td></tr>)}</tbody></table>}
      </article>)}</div>
    </section>

    <section className="dev-card">
      <h2>Data <span className="dev-count">{data.length - missing} present · {missing} missing</span></h2>
      <p className="dev-sub">Paths are relative to <code>{root}</code>.</p>
      {groups.map(g => <div key={g} className="dev-scroll"><h3>{g}</h3><table className="dev-table">
        <thead><tr><th /><th>Item</th><th>Path</th><th>Contents</th><th>Read by</th><th>Comes from</th></tr></thead>
        <tbody>{data.filter(d => d.group === g).map(d => <tr key={d.name} className={d.present ? "" : "missing"}>
          <td><span className={`dev-dot ${d.present ? "ok" : "bad"}`} /></td><td>{d.name}</td><td><code>{d.path}</code></td><td>{d.detail}</td><td>{d.readBy}</td><td>{d.source}</td>
        </tr>)}</tbody></table></div>)}
    </section>

    <section className="dev-card">
      <h2>Environment</h2>
      <dl className="dev-list">{env.map(([k, v]) => <div key={k}><dt><code>{k}</code></dt><dd>{v}</dd></div>)}</dl>
    </section>
  </main></div>;
}
