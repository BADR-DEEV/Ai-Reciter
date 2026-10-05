import { readFile, readdir, stat } from "node:fs/promises";
import path from "node:path";
import { API } from "./learn/api";
import { RECITERS } from "./reciters";

/** Development-only snapshot of what the app loads and where it reads it from. Never throws. */
const ROOT = path.resolve(process.cwd(), "..");
const rel = (p: string) => path.relative(ROOT, p) || ".";
const json = async <T = Record<string, unknown>>(p: string): Promise<T | null> => { try { return JSON.parse(await readFile(p, "utf8")); } catch { return null; } };
const size = async (p: string) => { try { return (await stat(p)).size; } catch { return null; } };
async function folder(p: string, match = /./) {
  try {
    const names = (await readdir(p)).filter(n => match.test(n));
    const sizes = await Promise.all(names.map(n => size(path.join(p, n))));
    return { files: names.length, bytes: sizes.reduce<number>((a, b) => a + (b || 0), 0) };
  } catch { return null; }
}
async function wavs(p: string): Promise<{ files: number; bytes: number } | null> {
  try {
    const all = await readdir(p, { recursive: true });
    const found = all.filter(n => /\.wav$/i.test(n));
    const sizes = await Promise.all(found.map(n => size(path.join(p, n))));
    return { files: found.length, bytes: sizes.reduce<number>((a, b) => a + (b || 0), 0) };
  } catch { return null; }
}
async function lines(p: string) { try { return (await readFile(p, "utf8")).trim().split(/\r?\n/).filter(Boolean).length; } catch { return null; } }

export type Health = Record<string, string | number | boolean | null>;
export type Score = { split: string; samples?: number; wer?: number; cer?: number };
export type ModelEntry = { preset: string; path: string; present: boolean; kind: string; weightsBytes: number | null; specs: [string, string][]; scores: Score[]; source: string; startWith: string };
export type DataEntry = { group: string; name: string; path: string; present: boolean; detail: string; readBy: string; source: string };

async function health(): Promise<{ url: string; data: Health | null; error?: string }> {
  try {
    const response = await fetch(`${API}/health`, { cache: "no-store", signal: AbortSignal.timeout(3000) });
    if (!response.ok) return { url: API, data: null, error: `HTTP ${response.status}` };
    return { url: API, data: await response.json() };
  } catch (cause) { return { url: API, data: null, error: cause instanceof Error ? cause.message : "unreachable" }; }
}

function scores(metrics: Record<string, unknown> | null): Score[] {
  if (!metrics) return [];
  return ["validation", "test"].flatMap(split => {
    const overall = (metrics[split] as Record<string, Score> | undefined)?.overall;
    return overall && typeof overall.wer === "number" ? [{ split, samples: overall.samples, wer: overall.wer, cer: overall.cer }] : [];
  });
}

async function fullModel(preset: string, dir: string, kind: string, source: string, reported?: string): Promise<ModelEntry> {
  const config = await json(path.join(dir, "config.json"));
  const metrics = await json(path.join(dir, "metrics.json"));
  const split = metrics?.split_reciters as Record<string, string[]> | undefined;
  const counts = metrics?.split_counts as Record<string, number> | undefined;
  return {
    preset, path: rel(dir), present: config !== null, kind, weightsBytes: await size(path.join(dir, "model.safetensors")),
    specs: config ? [
      ["Initialised from", /^[A-Za-z]:[\\/]/.test(String(metrics?.model)) ? `${path.basename(String(metrics?.model))} (training-PC path)` : String(metrics?.model ?? "—")],
      ["Architecture", `${config.model_type} · ${config.d_model} hidden · ${config.encoder_layers}+${config.decoder_layers} layers · ${config.encoder_attention_heads} heads`],
      ["Vocabulary / max tokens", `${config.vocab_size} / ${config.max_target_positions}`],
      ...(counts ? [["Training clips", `${counts.train}${metrics?.epochs ? ` · ${metrics.epochs} epochs · lr ${metrics.learning_rate}` : ""}`] as [string, string]] : []),
      ...(split?.train?.length ? [["Training readers", split.train.join(", ")] as [string, string]] : []),
      ...(reported ? [["Reported results", reported] as [string, string]] : []),
    ] : [],
    scores: scores(metrics), source, startWith: `python -m src.streaming.serve --model ${preset}`,
  };
}

async function models(): Promise<ModelEntry[]> {
  const release = await json<{ repo_id: string; commit: string }>(path.join(ROOT, "src/deployment/huggingface_release.json"));
  const run = path.join(ROOT, "runs/deepdml_qaloon_lora_base_v1");
  const adapter = await json(path.join(run, "adapter/adapter_config.json"));
  return [
    await fullModel("rattil-v3", path.join(ROOT, "runs/rattil_qaloon_v3"), "Full fine-tune · recommended",
      "Hugging Face Mathani-Ayat/rattil-qaloon-v3@e9e59ac (src/deployment/pull_hf_assets.py)",
      "Waleed (held-out voice): 2.2% WER normal speed · 13.4% at 1.25× · 21.2% at 1.5× (model card)"),
    await fullModel("gpu-full-base", path.join(ROOT, "runs/gpu_base_full"), "Full fine-tune · original team model",
      release ? `Hugging Face ${release.repo_id}@${release.commit.slice(0, 7)} (src/deployment/restore_local_full.py)` : "Hugging Face release"),
    {
      preset: "deepdml", path: rel(path.join(run, "adapter")), present: adapter !== null, kind: "LoRA adapter on DeepDML Whisper-base",
      weightsBytes: await size(path.join(run, "adapter/adapter_model.safetensors")),
      specs: adapter ? [
        ["Base", String(adapter.base_model_name_or_path)],
        ["LoRA rank / alpha", `${adapter.r} / ${adapter.lora_alpha}`],
        ["Target modules", Array.isArray(adapter.target_modules) ? adapter.target_modules.join(", ") : String(adapter.target_modules)],
      ] : [],
      scores: scores(await json(path.join(run, "metrics.json"))),
      source: "Training PC only (not published). Superseded by rattil-v3.",
      startWith: "python -m src.streaming.serve --model deepdml",
    },
  ];
}

async function data(): Promise<DataEntry[]> {
  const pub = path.join(process.cwd(), "public/quran");
  const fileEntry = async (group: string, name: string, p: string, readBy: string, source: string, describe?: (bytes: number) => Promise<string>): Promise<DataEntry> => {
    const bytes = await size(p);
    return { group, name, path: rel(p), present: bytes !== null, detail: bytes === null ? "missing" : describe ? await describe(bytes) : mb(bytes), readBy, source };
  };
  const dirEntry = async (group: string, name: string, p: string, match: RegExp, readBy: string, source: string): Promise<DataEntry> => {
    const f = await folder(p, match);
    return { group, name, path: rel(p), present: !!f && f.files > 0, detail: f ? `${f.files} files · ${mb(f.bytes)}` : "missing", readBy, source };
  };
  const cache = "src/dataset_collection/cache_quran_pages.py";
  const entries: Promise<DataEntry>[] = [
    fileEntry("Quran text", "Qālūn source text", path.join(ROOT, "src/dataset_collection/QaloonData_v10(1).json"), "cache script (builds surahs/*.json)", "In git"),
    fileEntry("Quran text", "Surah manifest", path.join(pub, "manifest.json"), "Studio (browser fetch /quran/manifest.json)", cache, async b => {
      const m = await json<{ surahs?: unknown[] }>(path.join(pub, "manifest.json"));
      return `${m?.surahs?.length ?? "?"} surahs · ${mb(b)}`;
    }),
    dirEntry("Quran text", "Per-surah text + regions", path.join(pub, "surahs"), /\.json$/, "Studio, challenges, tafsir, timing routes, Python /ws/recite", cache),
    dirEntry("Quran text", "Mushaf page SVGs", path.join(pub, "pages"), /\.svg$/, "Studio Mushaf view (browser)", `${cache} ← mp3quran.net`),
    dirEntry("Quran text", "Page geometry", path.join(pub, "geometry"), /\.json$/, "cache script (merged into surahs/*.json)", `${cache} ← mp3quran.net`),
    fileEntry("Quran text", "Hafs numbering reference", path.join(pub, "hafs-reference.json"), "cache script (numbering alignment only)", "quran-json (jsdelivr mirror)"),
    fileEntry("Quran text", "Qālūn ↔ Hafs text mapping", path.join(pub, "text-mapping.json"), "cache script", cache),
    fileEntry("Learning", "Tajweed draft", path.join(pub, "qalon_majwad_mushaf.json"), "/api/tajweed", "src/learning/build_qalon_tajweed.py"),
    fileEntry("Learning", "Ayah & word embeddings", path.join(pub, "text-embeddings.json"), "/api/challenges (falls back to spelling similarity)", "src/learning/build_text_embeddings.py ← runs/rattil_ayah_embed or Hugging Face", async b => {
      const m = await json<{ model?: string; ayahs?: { ids?: unknown[] }; words?: { keys?: unknown[] } }>(path.join(pub, "text-embeddings.json"));
      return `${m?.model ?? "?"} · ${m?.ayahs?.ids?.length ?? "?"} ayahs · ${m?.words?.keys?.length ?? "?"} words · ${mb(b)}`;
    }),
    ...RECITERS.map(r => fileEntry("Learning", `Audio similarity · ${r.name}`, path.join(pub, `audio-similarity-${r.id}.json`), "/api/challenges (falls back to text similarity)", "src/learning/build_audio_similarity.py")),
    Promise.resolve({ group: "External", name: "Tafsir & translation", path: "http://api.quran-tafseer.com", present: true, detail: "remote API, fetched per request", readBy: "/api/tafsir (server-side proxy)", source: "Quran Tafseer API" }),
  ];
  const app = new Map<string, string>(RECITERS.map(r => [r.folder, r.name]));
  for (const repo of ["qaloon-reciter-dataset", "qaloon-reciter-experiments", "qaloon-new-reciters"]) {
    const base = path.join(ROOT, "data/hf", repo);
    let folders: string[] = [];
    try { folders = (await readdir(base, { withFileTypes: true })).filter(d => d.isDirectory() && !d.name.startsWith(".")).map(d => d.name).sort(); } catch { /* not pulled */ }
    if (!folders.length) entries.push(Promise.resolve({ group: "Reciter audio (Hugging Face)", name: repo, path: rel(base), present: false, detail: "not pulled", readBy: "—", source: `Mathani-Ayat/${repo} · python src/deployment/pull_hf_assets.py` }));
    for (const name of folders) entries.push((async () => {
      const dir = path.join(base, name);
      const [rows, audio] = await Promise.all([lines(path.join(dir, "metadata.jsonl")), wavs(dir)]);
      const reader = app.get(name);
      return { group: "Reciter audio (Hugging Face)", name: reader ? `${reader} (${name})` : name, path: rel(dir), present: !!audio && audio.files > 0,
        detail: `${rows ?? 0} metadata rows · ${audio ? `${audio.files} WAVs · ${mb(audio.bytes)}` : "no audio"}`,
        readBy: reader ? `/api/reference-audio, /api/challenges (via src/dataset_collection/${name} symlink)` : "training/evaluation only",
        source: `Mathani-Ayat/${repo}` };
    })());
  }
  for (const r of RECITERS) for (const variant of ["large-v3/", ""])
    entries.push(fileEntry("Word timings", `Word timings · ${r.name}${variant ? " (large-v3)" : ""}`, path.join(ROOT, "data/playback_timings", `${variant}${r.id}.json`), "/api/reference-timing", "src/dataset_collection/build_playback_timings.py"));
  return Promise.all(entries);
}

const mb = (bytes: number) => bytes >= 1e6 ? `${(bytes / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1e3))} KB`;
export const formatBytes = mb;

export async function devInventory() {
  const [service, modelList, dataList] = await Promise.all([health(), models(), data()]);
  return {
    root: ROOT, service, models: modelList, data: dataList,
    env: [
      ["NEXT_PUBLIC_DEV_MODE", process.env.NEXT_PUBLIC_DEV_MODE ?? "(unset)"],
      ["NEXT_PUBLIC_RECITER_WS", process.env.NEXT_PUBLIC_RECITER_WS ?? "(unset → ws://127.0.0.1:8000/ws/recite)"],
      ["NEXT_PUBLIC_RECITER_API", process.env.NEXT_PUBLIC_RECITER_API ?? `(unset → ${API})`],
      ["NODE_ENV", process.env.NODE_ENV ?? ""],
    ] as [string, string][],
  };
}
