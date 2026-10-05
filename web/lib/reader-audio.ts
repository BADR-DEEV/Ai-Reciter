import { spawn } from "node:child_process";
import { access, mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { RECITERS, type ReciterID } from "./reciters";

/** Reference-reader recordings are not in Git; fetch them from the team's Hugging Face dataset on demand. */
const ROOT = path.resolve(process.cwd(), "..");
const STATUS = path.join(ROOT, "data/hf/pull-status.json");
const COMMAND = ["src/deployment/pull_hf_assets.py", "--skip-model", "--readers-only"];
export const MANUAL_COMMAND = `python ${COMMAND.join(" ")}`;

export type ReaderAudioState = {
  state: "ready" | "missing" | "downloading" | "waiting" | "failed";
  installed: ReciterID[]; missing: ReciterID[]; done?: number; total?: number; message?: string;
};
type PullStatus = { state?: string; pid?: number; done?: number; total?: number; message?: string };

const alive = (pid?: number) => { if (!pid) return false; try { process.kill(pid, 0); return true; } catch { return false; } };

export async function readerAudioState(): Promise<ReaderAudioState> {
  const present = await Promise.all(RECITERS.map(r => access(path.join(ROOT, "src/dataset_collection", r.folder, "metadata.jsonl")).then(() => true, () => false)));
  const installed = RECITERS.filter((_, i) => present[i]).map(r => r.id);
  const missing = RECITERS.filter((_, i) => !present[i]).map(r => r.id);
  if (!missing.length) return { state: "ready", installed, missing };
  let pull: PullStatus | null = null;
  try { pull = JSON.parse(await readFile(STATUS, "utf8")); } catch { /* never started */ }
  if (pull?.state === "downloading" || pull?.state === "waiting") {
    return alive(pull.pid) ? { state: pull.state, installed, missing, done: pull.done, total: pull.total, message: pull.message }
      : { state: "failed", installed, missing, message: "The download stopped before finishing." };
  }
  if (pull?.state === "failed") return { state: "failed", installed, missing, message: pull.message };
  return { state: "missing", installed, missing };
}

let starting: Promise<ReaderAudioState> | null = null;
export function startReaderAudioDownload() {
  starting ??= (async () => {
    const current = await readerAudioState();
    if (current.state !== "missing" && current.state !== "failed") return current;
    await mkdir(path.dirname(STATUS), { recursive: true });
    const child = spawn(process.env.RATTIL_PYTHON || "python3", COMMAND, {
      cwd: ROOT, detached: true, stdio: "ignore", env: { ...process.env, HF_HUB_DISABLE_PROGRESS_BARS: "1" },
    });
    const failed = (message: string) => writeFile(STATUS, JSON.stringify({ state: "failed", message })).catch(() => {});
    child.on("error", error => void failed(`Could not start Python (${error.message}). Set RATTIL_PYTHON or run: ${MANUAL_COMMAND}`));
    if (child.pid) {
      // Claim the download before Python writes its first status so parallel requests don't start a second one.
      await writeFile(STATUS, JSON.stringify({ state: "downloading", pid: child.pid, done: 0, total: 0 }));
      child.unref();
    }
    return readerAudioState();
  })().finally(() => { starting = null; });
  return starting;
}
