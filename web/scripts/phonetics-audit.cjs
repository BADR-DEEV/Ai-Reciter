/* Generate a teacher-editable audit from the exact cached Qaloon source.
   npm run phonetics:audit; no network, no mutation of source Quran text. */
const fs = require("node:fs");
const path = require("node:path");
const ts = require("typescript");
const Module = require("node:module");
const filename = path.resolve(__dirname, "../lib/qaloon-g2p.ts");
const moduleObject = new Module(filename, module);
moduleObject._compile(ts.transpileModule(fs.readFileSync(filename, "utf8"), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText, filename);
const { qaloonG2P, G2P_VERSION } = moduleObject.exports;
const root = path.resolve(__dirname, "../public/quran");
const hafs = JSON.parse(fs.readFileSync(path.join(root, "hafs-reference.json"), "utf8"));
const out = path.resolve(__dirname, "../../docs/generated");
fs.mkdirSync(out, { recursive: true });
const csv = value => `"${String(value ?? "").replaceAll('"', '""')}"`;
const header = ["qaloon_surah", "qaloon_ayah", "source_text", "draft_pause", "draft_connected", "hafs_reference_ids", "hafs_reference_text", "hafs_rule_output_NOT_authority", "review_flags", "version", "reviewer", "approved_pause", "approved_connected", "audio_reference", "status"];
const lines = [header.map(csv).join(",")];
let count = 0, flagged = 0;
for (const file of fs.readdirSync(path.join(root, "surahs")).filter(f => f.endsWith(".json")).sort()) {
  const surah = JSON.parse(fs.readFileSync(path.join(root, "surahs", file), "utf8"));
  for (const ayah of surah.ayahs) {
    const pause = qaloonG2P(ayah.text), connected = qaloonG2P(ayah.text, "connect");
    const ids = [...new Set(ayah.regions.map(r => r.display_ayah))];
    const reference = ids.map(id => hafs.find(s => s.id === surah.id)?.verses.find(v => v.id === id)?.text || "").join(" / ");
    lines.push([surah.id, ayah.ayah, ayah.text, pause.text, connected.text, ids.join(";"), reference,
      reference ? qaloonG2P(reference).text : "", pause.warnings.join(" | "), G2P_VERSION, "", "", "", "", "needs-qualified-review"].map(csv).join(","));
    count++; if (pause.warnings.length) flagged++;
  }
}
fs.writeFileSync(path.join(out, "qaloon-phonetics-audit.csv"), "\ufeff" + lines.join("\r\n") + "\r\n");
fs.writeFileSync(path.join(out, "phonetics-summary.json"), JSON.stringify({ version: G2P_VERSION, ayahs: count, flagged, approved: 0, source: "cached QaloonData_v10 via Quran cache", note: "Hafs columns are only mapped comparison, never replacement labels or scholarly approval." }, null, 2));
console.log(`Audit: ${count} ayahs (${flagged} with additional flags), 0 scholar-approved. ${out}`);
