import { spawn } from "node:child_process";
import { access } from "node:fs/promises";
import path from "node:path";
import { startReaderAudioDownload } from "./reader-audio";
import { RECITERS } from "./reciters";

/** Generates or fetches everything a fresh clone lacks (none of it is in Git). Each step only runs when its output is missing. */
const ROOT = path.resolve(process.cwd(), "..");
const QURAN = path.join(process.cwd(), "public/quran");
const exists = (p: string) => access(p).then(() => true, () => false);

function run(label: string, args: string[], env: Record<string, string> = {}) {
  return new Promise<boolean>(resolve => {
    console.log(`[rattil] ${label}…`);
    const child = spawn(process.env.RATTIL_PYTHON || "python3", args, { cwd: ROOT, stdio: ["ignore", "ignore", "pipe"], env: { ...process.env, ...env } });
    let errors = "";
    child.stderr.on("data", chunk => { errors = (errors + chunk).slice(-4000); });
    child.on("error", error => { console.warn(`[rattil] ${label}: could not start Python (${error.message}). Set RATTIL_PYTHON.`); resolve(false); });
    child.on("close", code => {
      if (code === 0) console.log(`[rattil] ${label}: done`);
      else console.warn(`[rattil] ${label} failed (exit ${code}). Run it by hand: python ${args.join(" ")}\n${errors.trim().split("\n").slice(-3).join("\n")}`);
      resolve(code === 0);
    });
  });
}

export async function firstRunSetup() {
  const manifest = path.join(QURAN, "manifest.json");
  if (!await exists(manifest)) await run("First run: caching Quran pages and text (~350 MB)", ["src/dataset_collection/cache_quran_pages.py", "--skip-metadata"]);
  if (await exists(manifest) && !await exists(path.join(QURAN, "qalon_majwad_mushaf.json")))
    await run("First run: generating the tajweed draft", ["src/learning/build_qalon_tajweed.py"]);
  // Optional, and independent of audio, so it runs alongside the reader download. Challenges fall back to
  // spelling similarity without it. Downloads the embedding model (~0.5 GB) unless runs/rattil_ayah_embed exists.
  // The Mac shim hides Anaconda's broken torchvision from transformers; elsewhere it changes nothing.
  if (await exists(manifest) && !await exists(path.join(QURAN, "text-embeddings.json")))
    void run("First run: embedding ayahs and words for challenge choices", ["src/learning/build_text_embeddings.py"],
      { PYTHONPATH: [path.join(ROOT, "src/deployment/mac_shim"), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter) });

  const audio = await startReaderAudioDownload();
  if (audio.state !== "ready") {
    console.log(`[rattil] Reader audio ${audio.state}: ${audio.missing.join(", ")} (progress at /api/reader-audio)`);
    return;
  }
  // Optional: challenges fall back to text similarity without these. Stop at the first failure (usually a torchaudio problem).
  for (const reader of RECITERS) {
    if (await exists(path.join(QURAN, `audio-similarity-${reader.id}.json`))) continue;
    if (!await run(`Building the listening-distractor index for ${reader.name}`, ["src/learning/build_audio_similarity.py", "--reciter", reader.id])) break;
  }
}
