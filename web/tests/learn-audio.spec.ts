import { test, expect } from "@playwright/test";

// Instruments the Web Audio player: every AudioBufferSourceNode.start() is
// recorded with its offset and duration, so we can verify real decoded audio
// is scheduled without needing speakers.
test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    const w = window as unknown as { __starts: [number, number, number | undefined][] };
    w.__starts = [];
    const start = AudioBufferSourceNode.prototype.start;
    AudioBufferSourceNode.prototype.start = function (when?: number, offset?: number, duration?: number) {
      w.__starts.push([this.buffer?.duration ?? 0, offset ?? 0, duration]);
      return start.call(this, when, offset, duration);
    };
  });
});

const starts = (page: import("@playwright/test").Page) => page.evaluate(() => (window as unknown as { __starts: [number, number, number | undefined][] }).__starts);

test("letter sounds play from pre-rendered files", async ({ page }) => {
  await page.goto("/learn/letters-1");
  await page.getByRole("button", { name: /Listen: ʾā/ }).click();
  await expect.poll(() => starts(page)).toHaveLength(1);
  const [[duration]] = await starts(page);
  expect(duration).toBeGreaterThan(0.2);
});

test("landing alphabet tiles play their sound", async ({ page }) => {
  await page.goto("/");
  await page.getByTitle(/^Ḥā’/).click();
  await expect.poll(() => starts(page)).toHaveLength(1);
});

test("ayah plays only its own segment of Al-Husary's recording", async ({ page }) => {
  test.skip(!!process.env.OFFLINE, "Needs the MP3Quran CDN");
  await page.goto("/learn/surah-112");
  await page.getByRole("button", { name: "Continue" }).click();
  await page.getByRole("button", { name: "Listen to Al-Husary" }).click();
  await expect(page.getByRole("button", { name: "Stop" })).toBeVisible({ timeout: 20000 });
  const [[duration, offset, length]] = await starts(page);
  expect(duration).toBeGreaterThan(25);       // the whole surah was decoded
  expect(offset).toBeCloseTo(9.24, 2);        // ayah 1 starts at 9.24 s
  expect(length).toBeCloseTo(4.242, 2);       // and lasts 4.24 s
  await page.getByRole("button", { name: "Stop" }).click();
  await expect(page.getByRole("button", { name: "Listen to Al-Husary" })).toBeVisible();
});
