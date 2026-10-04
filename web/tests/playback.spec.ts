import { test, expect } from "@playwright/test";
import { validTimings, wordAtTime } from "../lib/playback";

function wav(seconds = 10) {
  const samples = seconds * 16000, data = Buffer.alloc(44 + samples * 2);
  data.write("RIFF", 0); data.writeUInt32LE(data.length - 8, 4); data.write("WAVEfmt ", 8);
  data.writeUInt32LE(16, 16); data.writeUInt16LE(1, 20); data.writeUInt16LE(1, 22);
  data.writeUInt32LE(16000, 24); data.writeUInt32LE(32000, 28); data.writeUInt16LE(2, 32); data.writeUInt16LE(16, 34);
  data.write("data", 36); data.writeUInt32LE(samples * 2, 40);
  for (let i = 0; i < samples; i++) data.writeInt16LE(Math.round(100 * Math.sin(i / 10)), 44 + 2 * i);
  return data;
}

test.beforeEach(async ({ page }) => {
  await page.route("**/api/tafsir?**", route => route.fulfill({ json: {
    kind: "Test commentary", book: { name: "Test book", author: "Test author" },
    entries: [{ providerAyah: 1, text: route.request().url().includes("lang=ar") ? "تفسير عربي للاختبار" : "Test English meaning" }], mappingNote: "Text aligned, not recitation text",
  } }));
  await page.route("**/api/reference-audio?**", route => route.fulfill({ contentType: "audio/wav", body: wav() }));
});

test("clock selection preserves silence and rejects reordered or guessed timing", () => {
  const data = { displayText: "قل هو", duration: 3, status: "draft", words: [
    { index: 0, text: "قل", start: .1, end: .5 }, { index: 1, text: "هو", start: .8, end: 1.4 },
  ] };
  expect(validTimings(data, "قل هو")).toBe(true);
  expect(wordAtTime(data.words, .4)).toBe(0);
  expect(wordAtTime(data.words, .6)).toBeNull();
  expect(wordAtTime(data.words, 2)).toBeNull();
  expect(validTimings({ ...data, words: [...data.words].reverse() }, "قل هو")).toBe(false);
  expect(validTimings(data, "قل هي")).toBe(false);
});

test("real reference timing stays reader-specific and canonical", async ({ request }) => {
  const result = await request.get("/api/reference-timing?surah=1&ayah=1&reciter=huthaify");
  expect(result.status()).toBe(200);
  const timing = await result.json();
  const source = await (await request.get("/quran/surahs/001.json")).json();
  const verse = source.ayahs[0];
  const display = (verse.displayText || verse.text).replace(/[\u0660-\u0669\d]+/g, "").trim();
  expect(validTimings(timing, display)).toBe(true);
  expect(timing.status).toBe("machine-timing-proposal");
  expect((await request.get("/api/reference-timing?surah=1&ayah=1&reciter=../../other")).status()).toBe(400);
  expect((await request.get("/api/reference-timing?surah=2&ayah=1&reciter=huthaify")).status()).toBe(404);
});

test("translation and Arabic tafsir stay physically left in both UI languages", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto("/studio");
  await expect(page.locator(".commentary-sidebar")).toContainText("Test English meaning");
  await expect(page.locator(".commentary-sidebar")).toContainText("تفسير عربي للاختبار");
  for (const language of ["en", "ar"]) {
    if (language === "ar") await page.locator(".lh-lang").click();
    const left = await page.locator(".commentary-sidebar").boundingBox(), reader = await page.locator(".mushaf-card").boundingBox();
    expect(left!.x + left!.width).toBeLessThanOrEqual(reader!.x);
  }
});

for (const mode of ["Ayah view", "Phonetics", "Meaning + phonetics"]) {
  test(`audio clock drives square word cursor in ${mode}, stop clears it`, async ({ page }) => {
    await page.goto("/studio");
    const source = await (await page.request.get("/quran/surahs/001.json")).json();
    const verse = source.ayahs[0];
    const displayText = (verse.displayText || verse.text).replace(/[\u0660-\u0669\d]+/g, "").trim();
    await page.route("**/api/reference-timing?**", route => route.fulfill({ json: {
      displayText, duration: 10, status: "machine-timing-proposal", words: displayText.split(/\s+/).map((text: string, index: number) => ({ text, index, start: .1 + index * 1.7, end: 1.4 + index * 1.7 })),
    } }));
    await page.getByRole("button", { name: mode, exact: true }).click();
    const ayah = page.locator('.text-ayah[data-ayah="1"]');
    await ayah.getByRole("button", { name: "Listen to ayah 1" }).click();
    const first = mode === "Meaning + phonetics" ? ayah.locator('.phonetic-aid .playback-word[data-word-index="0"]') : ayah.locator('.ayah-reading-line .playback-word[data-word-index="0"]');
    await expect(first).toBeVisible();
    const next = mode === "Meaning + phonetics" ? ayah.locator('.phonetic-aid .playback-word[data-word-index="1"]') : ayah.locator('.ayah-reading-line .playback-word[data-word-index="1"]');
    await expect(next).toBeVisible();
    await ayah.getByRole("button", { name: "Listen to ayah 1" }).click();
    await expect(page.locator(".playback-word")).toHaveCount(0);
    await expect(page.locator(".text-ayah.playback-ayah")).toHaveCount(0);
  });
}

test("missing timing keeps ayah-only highlight; Mushaf marks whole ayah", async ({ page }) => {
  await page.route("**/api/reference-timing?**", route => route.fulfill({ status: 404, json: { words: [] } }));
  await page.goto("/studio");
  const ayah = page.locator('.text-ayah[data-ayah="1"]');
  await ayah.getByRole("button", { name: "Listen to ayah 1" }).click();
  await expect(ayah).toHaveClass(/playback-ayah/);
  await expect(page.locator(".playback-word")).toHaveCount(0);
  await expect(ayah).toContainText("word timings unavailable");
  await ayah.getByRole("button", { name: "Listen to ayah 1" }).click();
  await page.getByRole("button", { name: "Mushaf", exact: true }).click();
  await expect(page.locator(".svg-page")).toBeVisible();
  await page.getByRole("button", { name: "Listen to ayah 1" }).click();
  await expect(page.locator('.ayah-region.playback-ayah[data-ayah="1"]').first()).toBeVisible();
  await expect(page.locator(".playback-word")).toHaveCount(0);
  await page.getByLabel("Reference reciter").selectOption("husary");
  await expect(page.locator(".ayah-region.playback-ayah")).toHaveCount(0);
});
