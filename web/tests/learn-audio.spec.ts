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

test("ayah plays its own local Huthaify clip, not another reader's timings", async ({ page }) => {
  await page.goto("/learn/surah-112");
  await page.getByRole("button", { name: "Continue" }).click();
  await expect(page.getByLabel("Reference reciter")).toHaveValue("huthaify");
  const fetched = page.waitForResponse(r => r.url().includes("/api/reference-audio?") && r.url().includes("reciter=huthaify"));
  await page.getByRole("button", { name: "Listen to ayah 1", exact: true }).click();
  expect((await fetched).status()).toBe(200);
  await expect(page.getByRole("button", { name: "Listen to ayah 1", exact: true })).toContainText("Stop", { timeout: 20000 });
  const [[duration, offset, length]] = await starts(page);
  expect(duration).toBeGreaterThan(1);
  expect(duration).toBeLessThan(30);          // decoded only the ayah
  expect(offset).toBe(0);                    // local clip starts at its own zero
  expect(length).toBeUndefined();            // no foreign full-surah timings
  await page.getByRole("button", { name: "Listen to ayah 1", exact: true }).click();
  await expect(page.getByRole("button", { name: "Listen to ayah 1", exact: true })).toContainText("Al-Huthaify");
});
