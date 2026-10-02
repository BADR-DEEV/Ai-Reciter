/** Stable file name for a pre-rendered sound (FNV-1a hash of the Arabic text). */
export function soundFile(text: string) {
  let h = 2166136261;
  for (const c of text.normalize("NFC")) h = Math.imul(h ^ c.codePointAt(0)!, 16777619);
  return `${(h >>> 0).toString(16).padStart(8, "0")}.mp3`;
}
