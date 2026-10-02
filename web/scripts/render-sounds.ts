// Pre-render every letter/syllable/word sound used by the lessons into small
// MP3 files, so playback never depends on the browser's speech engine.
//
//   cd web && npx tsx scripts/render-sounds.ts        (macOS: uses `say` + ffmpeg)
//
// The voice is macOS's Arabic voice "Majed": a stopgap. To use a teacher's
// recordings instead, replace the MP3 files in public/audio/sounds/ keeping
// the same names (see lib/learn/sounds.json for the text of each file).
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { LESSONS } from "../lib/learn/curriculum";
import { LETTERS } from "../lib/learn/letters";
import { soundFile } from "../lib/learn/sound-name";

const VOICE = process.env.SOUND_VOICE || "Majed";
const out = join(__dirname, "..", "public", "audio", "sounds");
mkdirSync(out, { recursive: true });

const texts = new Set<string>(LETTERS.map(l => l.practice));
for (const lesson of LESSONS) for (const step of lesson.steps) {
  if ((step.kind === "choice" || step.kind === "say") && step.listen) texts.add(step.listen);
  if (step.kind === "vowel") step.examples.forEach(e => texts.add(e));
}

const scratch = mkdtempSync(join(tmpdir(), "sounds-"));
const manifest: Record<string, string> = {};
for (const text of [...texts].sort()) {
  const file = soundFile(text);
  manifest[text] = file;
  if (existsSync(join(out, file)) && !process.env.FORCE) continue;
  const aiff = join(scratch, "x.aiff");
  execFileSync("say", ["-v", VOICE, "-r", "120", "-o", aiff, text]);
  // Mono 64 kbps MP3, leading/trailing silence trimmed, a short tail of padding.
  execFileSync("ffmpeg", ["-y", "-loglevel", "error", "-i", aiff, "-af",
    "silenceremove=start_periods=1:start_threshold=-45dB,areverse,silenceremove=start_periods=1:start_threshold=-45dB,areverse,apad=pad_dur=0.08",
    "-ac", "1", "-ar", "22050", "-b:a", "64k", join(out, file)]);
}
rmSync(scratch, { recursive: true, force: true });
writeFileSync(join(__dirname, "..", "lib", "learn", "sounds.json"), JSON.stringify(manifest, null, 1) + "\n");
console.log(`${texts.size} sounds in ${out}`);
