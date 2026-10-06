import { test, expect, type Page } from "@playwright/test";

// A 1.5 s tone: enough sound to search. The model's answer is mocked below.
function wav(seconds: number) {
  const rate = 16000, samples = Math.round(rate * seconds), buffer = Buffer.alloc(44 + samples * 2);
  buffer.write("RIFF", 0); buffer.writeUInt32LE(36 + samples * 2, 4); buffer.write("WAVEfmt ", 8);
  buffer.writeUInt32LE(16, 16); buffer.writeUInt16LE(1, 20); buffer.writeUInt16LE(1, 22);
  buffer.writeUInt32LE(rate, 24); buffer.writeUInt32LE(rate * 2, 28); buffer.writeUInt16LE(2, 32); buffer.writeUInt16LE(16, 34);
  buffer.write("data", 36); buffer.writeUInt32LE(samples * 2, 40);
  for (let i = 0; i < samples; i++) buffer.writeInt16LE(Math.round(Math.sin(i / 8) * 9000), 44 + i * 2);
  return buffer;
}

const place = (rank: number, surah: number, ayah: number, confidence: string) => ({
  rank, surah, name: `Surah ${surah}`, arabic: "سورة", trained: surah > 77, start: { surah, ayah }, end: { surah, ayah },
  juz: [surah > 77 ? 30 : 1], score: 5 - rank, coverage: 1 / rank, matched_words: 4, confidence,
  ayahs: [{ surah, ayah, juz: 30, text: "قُلْ هُوَ اَ۬للَّهُ أَحَدٌ", displayText: "قُلْ هُوَ اَللَّهُ أَحَدٌ",
    words: ["قل", "هو", "الله", "احد"].map((text, index) => ({ index, text, status: "correct" })) }],
});

async function mockSearch(page: Page) {
  const requests: { juz: number[] }[] = [];
  await page.route("**/api/search", async route => {
    const body = route.request().postDataJSON();
    requests.push(body);
    await route.fulfill({ json: { verdict: "found", transcript: "بسم الله الرحمن الرحيم قل هو الله احد", heard: ["قل", "هو", "الله", "احد"],
      opening: ["basmala"], juz: body.juz, model_used: "plain",
      results: [place(1, 112, 1, "high"), place(2, 2, 79, "medium"), place(3, 3, 32, "low")] } });
  });
  return requests;
}

test("best match first, other places on demand, juzʾ scope sent with the clip", async ({ page }) => {
  const requests = await mockSearch(page);
  await page.goto("/search");
  await page.getByText("Only in juzʾ I choose").click();
  const juz = page.locator(".find-juz-grid button");
  await expect(juz).toHaveCount(30);
  await expect(juz.nth(29)).toHaveAttribute("aria-pressed", "true");  // Juz ʿAmma is the starting choice
  await juz.first().click();
  await page.locator('input[type="file"]').setInputFiles({ name: "clip.wav", mimeType: "audio/wav", buffer: wav(1.5) });
  await expect(page.locator(".find-match.primary h2")).toContainText("Surah 112");
  expect(requests[0].juz).toEqual([1, 30]);
  await expect(page.locator(".find-match.primary .find-confidence")).toHaveText("Strong match");
  await expect(page.getByText(/set aside the basmalah/)).toBeVisible();
  await expect(page.locator(".find-others")).toHaveCount(0);
  await page.getByRole("button", { name: "Show 2 more matches" }).click();
  await expect(page.locator(".find-others .find-match")).toHaveCount(2);
  await expect(page.locator(".find-others .find-confidence")).toHaveText(["Possible match", "Weak match"]);

  // Changing the scope offers to search the same clip again.
  await juz.first().click();
  await page.getByRole("button", { name: "Search this clip again" }).click();
  await expect.poll(() => requests.length).toBe(2);
  expect(requests[1].juz).toEqual([30]);
});

test("search page fits a phone with the juzʾ picker open", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/search");
  await page.getByText("Only in juzʾ I choose").click();
  await expect(page.locator(".find-juz-grid button")).toHaveCount(30);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});
