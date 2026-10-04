import { test, expect } from "@playwright/test";
import { tajweedSegments } from "../lib/tajweed";

test("reader views replace Arabic with large phonetics and retain canonical text", async ({ page }) => {
  await page.goto("/studio");
  await expect(page.getByLabel("Reference reciter")).toHaveValue("huthaify");
  const original = await page.locator('[data-ayah="3"] .ayah-reading-line').innerText();
  await page.getByRole("button", { name: "Phonetics", exact: true }).click();
  await expect(page.locator(".phonetic-text")).toBeVisible();
  await expect(page.locator('[data-ayah="3"] .ayah-reading-line')).toContainText("maliki");
  expect(await page.locator('[data-ayah="3"] .ayah-reading-line').innerText()).not.toMatch(/[\u0621-\u064a]/);
  await page.getByRole("button", { name: "Ayah view", exact: true }).click();
  expect(await page.locator('[data-ayah="3"] .ayah-reading-line').innerText()).toBe(original);
  await page.getByLabel("Draft tajweed colors").check();
  await expect(page.locator(".tajweed-legend")).toBeVisible();
  await expect(page.locator(".tajweed-palette li")).toHaveCount(8);
  await expect(page.locator(".tajweed-palette")).toContainText("Necessary madd · 6 counts");
  await expect(page.locator(".basmalah .tajweed-span").first()).toBeVisible();
  expect(await page.locator('[data-ayah="3"] .ayah-reading-line').innerText()).toBe(original);
  await expect(page.locator(".tajweed-span").first()).toBeVisible();
  await page.locator(".tajweed-legend summary").click();
  await expect(page.locator(".tajweed-legend")).toContainText("Natural madd · 2 harakat");
  await expect(page.locator(".tajweed-legend")).toContainText("Not a certified");
});

test("tajweed colors preserve Arabic font runs, joins and word widths", async ({ page }) => {
  await page.goto("/studio");
  await page.getByRole("button", { name: "Ayah view", exact: true }).click();
  await expect(page.locator(".ayah-reading-line .quran-word").first()).toBeVisible();
  await page.evaluate(() => document.fonts.ready);
  const plain = await page.locator(".ayah-reading-line .quran-word").evaluateAll(words => words.map(word => ({
    text: word.textContent, width: word.getBoundingClientRect().width, font: getComputedStyle(word).font,
  })));
  await page.getByLabel("Draft tajweed colors").check();
  await expect(page.locator(".ayah-reading-line .tajweed-span").first()).toBeVisible();
  const colored = await page.locator(".ayah-reading-line .quran-word").evaluateAll(words => words.map(word => ({
    text: word.textContent, width: word.getBoundingClientRect().width, font: getComputedStyle(word).font,
    piecesMatch: [...word.children].every(piece => {
      const parent = getComputedStyle(word), child = getComputedStyle(piece);
      return child.fontFamily === parent.fontFamily && child.fontWeight === parent.fontWeight
        && child.fontSize === parent.fontSize && child.display === "inline" && child.letterSpacing === parent.letterSpacing;
    }),
  })));
  expect(colored).toHaveLength(plain.length);
  colored.forEach((word, index) => {
    expect(word.text).toBe(plain[index].text);
    expect(word.font).toBe(plain[index].font);
    expect(Math.abs(word.width - plain[index].width)).toBeLessThan(.25);
    expect(word.piecesMatch).toBe(true);
  });
  await expect(page.locator(".mushaf-card > .tajweed-legend")).toBeVisible();
  await page.locator(".lh-lang").click();
  await expect(page.locator(".tajweed-palette")).toContainText("مد لازم · ٦ حركات");
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: "test-results/tajweed-arabic-mobile.png", fullPage: true });
});

test("Arabic switch updates direction, dashboard and commentary; reciter persists", async ({ page }) => {
  const requested: string[] = [];
  await page.route("**/api/tafsir?**", route => {
    requested.push(route.request().url());
    const ar = route.request().url().includes("lang=ar");
    return route.fulfill({ json: { kind: ar ? "Arabic tafsir" : "Translation of meanings", book: { name: "Test source", author: "Test author" }, entries: [{ providerAyah: 2, text: ar ? "شرح عربي تجريبي للاختبار" : "Test meaning" }], mappingNote: "Qaloon text alignment" } });
  });
  await page.goto("/studio");
  await page.getByLabel("Reference reciter").selectOption("husary");
  await page.getByRole("button", { name: "Meaning + phonetics", exact: true }).click();
  await page.locator(".ayah-meaning-pane").first().scrollIntoViewIfNeeded();
  await expect(page.getByText("Test meaning").first()).toBeVisible();
  await page.locator(".lh-lang").click();
  await expect(page.locator("html")).toHaveAttribute("dir", "rtl");
  await expect(page.getByRole("button", { name: "ابدأ التلاوة", exact: true })).toBeVisible();
  await expect(page.getByLabel("قارئ الاستماع")).toHaveValue("husary");
  await page.locator(".ayah-meaning-pane").first().scrollIntoViewIfNeeded();
  await expect(page.getByText("شرح عربي تجريبي للاختبار").first()).toBeVisible();
  expect(requested.some(url => url.includes("lang=ar"))).toBe(true);
  await page.reload();
  await expect(page.getByLabel("قارئ الاستماع")).toHaveValue("husary");
  await page.goto("/games");
  await expect(page.getByLabel("قارئ الاستماع")).toHaveValue("husary");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("تحدٍّ");
  await page.goto("/profile");
  await expect(page.getByLabel("الاسم", { exact: true })).toBeVisible();
});

test("actual reader clips differ; invalid and unavailable voices never silently fallback", async ({ request }) => {
  const defaultAudio = await request.get("/api/reference-audio?surah=1&ayah=1");
  const huthaify = await request.get("/api/reference-audio?surah=1&ayah=1&reciter=huthaify");
  const husary = await request.get("/api/reference-audio?surah=1&ayah=1&reciter=husary");
  expect(defaultAudio.status()).toBe(200); expect(husary.status()).toBe(200);
  expect((await defaultAudio.body()).equals(await huthaify.body())).toBe(true);
  expect((await husary.body()).equals(await huthaify.body())).toBe(false);
  expect((await request.get("/api/reference-audio?surah=2&ayah=1&reciter=huthaify")).status()).toBe(404);
  expect((await request.get("/api/reference-audio?surah=1&ayah=1&reciter=../../other")).status()).toBe(400);
  const response = await request.get("/api/tajweed?surah=1");
  expect(response.status()).toBe(200);
  const body = await response.json();
  expect(body.approved_ayahs).toBe(0); expect(body.surahs).toHaveLength(1); expect(body.surahs[0].ayahs).toHaveLength(7);
});

test("all navigation remains available and reader modes fit a phone in both languages", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/studio");
  for (const label of ["Ayah view", "Phonetics", "Meaning + phonetics", "Mushaf"]) {
    await page.getByRole("button", { name: label, exact: true }).click();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  }
  await expect(page.locator('nav a[href="/studio"]')).toHaveAttribute("aria-current", "page");
  await expect(page.locator('nav a[href="/profile"]')).toBeVisible();
  await expect(page.locator('nav a[href="/"]')).toBeVisible();
  await page.locator(".lh-lang").click();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test("overlapping tajweed spans preserve text and retain both rule IDs", () => {
  const text = "قْ";
  const spans = ["qalqala", "tafkhim"].map(rule => ({ start: 0, end: 2, rule, status: "needs-review" as const, context_note: "" }));
  const segments = tajweedSegments(text, spans, 0, 2);
  expect(segments.map(s => s.text).join("")).toBe(text);
  expect(segments[0].rules).toEqual(["qalqala", "tafkhim"]);
});
