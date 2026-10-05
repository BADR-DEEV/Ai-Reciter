// A fresh clone has no Quran cache, tajweed draft or reader recordings (none are in Git), so set them
// up in the background. Set RATTIL_AUTO_SETUP=0 to skip; the UI then offers the reader download.
import { firstRunSetup } from "./lib/first-run";

if (process.env.RATTIL_AUTO_SETUP !== "0") {
  firstRunSetup().catch(error => console.warn("[rattil] First-run setup failed:", error));
}
