import { test, expect } from "@playwright/test";
import { visibleRules, wordText, type TajweedRules } from "../lib/tajweed";

test("reader views replace Arabic with large phonetics and retain canonical text", async ({ page }) => {
  await page.goto("/studio");
  await expect(page.getByLabel("Reference reciter")).toHaveValue("huthaify");
  await expect(page.locator('[data-text-source="qaloon"]')).toBeVisible();
  const original = await page.locator('[data-ayah="3"] .ayah-reading-line').innerText();
  await page.getByRole("button", { name: "Phonetics", exact: true }).click();
  await expect(page.locator(".phonetic-text")).toBeVisible();
  await expect(page.locator('[data-ayah="3"] .ayah-reading-line')).toContainText("maliki");
  expect(await page.locator('[data-ayah="3"] .ayah-reading-line').innerText()).not.toMatch(/[\u0621-\u064a]/);
  await page.getByRole("button", { name: "Ayah view", exact: true }).click();
  expect(await page.locator('[data-ayah="3"] .ayah-reading-line').innerText()).toBe(original);
  await page.getByLabel("Tajweed colors").check();
  await expect(page.locator(".tajweed-legend")).toBeVisible();
  await expect(page.locator(".tajweed-palette li")).toHaveCount(9);
  await expect(page.locator(".tajweed-palette")).toContainText("Necessary madd · 6");
  await expect(page.locator(".tajweed-palette")).toContainText("Qālūn riwāyah point");
  await expect(page.locator(".basmalah .tajweed-span").first()).toBeVisible();
  expect(await page.locator('[data-ayah="3"] .ayah-reading-line').innerText()).toBe(original);
  await expect(page.locator(".tajweed-span").first()).toBeVisible();
  await page.locator(".tajweed-legend summary").click();
  await expect(page.locator(".tajweed-legend")).toContainText("Ghunna");
  await expect(page.locator(".tajweed-legend")).toContainText("qualified Qālūn teacher");
});

test("tajweed colors preserve Arabic font runs, joins and word widths", async ({ page }) => {
  await page.goto("/studio");
  await page.getByRole("button", { name: "Ayah view", exact: true }).click();
  await expect(page.locator('[data-text-source="qaloon"]')).toBeVisible();
  await page.evaluate(() => document.fonts.ready);
  const plain = await page.locator(".ayah-reading-line .quran-word").evaluateAll(words => words.map(word => ({
    text: word.textContent, width: word.getBoundingClientRect().width, font: getComputedStyle(word).font,
  })));
  await page.getByLabel("Tajweed colors").check();
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
  await expect(page.locator(".tajweed-palette")).toContainText("مد لازم · ٦");
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
  const fatiha = await (await request.get("/quran/tajweed/001.json")).json();
  expect(fatiha.ayahs).toHaveLength(7);
  const rules = await (await request.get("/quran/tajweed/rules.json")).json();
  expect(rules.riwayah).toContain("Qālūn"); expect(rules.topics.length).toBeGreaterThanOrEqual(14);
});

test("all navigation remains available and reader modes fit a phone in both languages", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/studio");
  for (const mode of ["text", "phonetic", "meaning", "mushaf"]) {
    await page.getByRole("combobox", { name: "Reader view", exact: true }).selectOption(mode);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  }
  await expect(page.locator('.mobile-audio-slot .start-button')).toBeVisible();
  await expect(page.locator('.mobile-nav a[href="/studio"]')).toHaveAttribute("aria-current", "page");
  await page.locator('.mobile-nav > button').click();
  await expect(page.locator('.mobile-menu a[href="/"]')).toBeVisible();
  await expect(page.locator('.mobile-menu a[href="/profile"]')).toBeVisible();
  await page.keyboard.press("Escape");
  await page.locator(".lh-lang").click();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test("tajweed filters hide groups and detail rules but keep the text", () => {
  const rules = { order: ["qalqala", "tafkhim_letter", "mim_jam"], rules: {
    qalqala: { group: "qalqala", level: "core" }, tafkhim_letter: { group: "heavy", level: "detail" }, mim_jam: { group: "riwaya", level: "core" },
  } } as unknown as TajweedRules;
  const base = { show: true, details: false, hidden: [] as string[], check: false };
  expect(visibleRules([0, 1, 2], rules, base)).toEqual(["qalqala", "mim_jam"]);
  expect(visibleRules([0, 1, 2], rules, { ...base, details: true, hidden: ["qalqala"] })).toEqual(["tafkhim_letter", "mim_jam"]);
  expect(wordText([["قْ", [0, 1]], ["لَ"]])).toBe("قْلَ");
});

test("tapping a colored letter explains its Qālūn rule", async ({ page }) => {
  await page.goto("/studio?surah=2&ayah=5");
  await page.getByRole("button", { name: "Ayah view", exact: true }).click();
  await page.getByLabel("Tajweed colors").check();
  const tasheel = page.locator('[data-ayah="5"] .tj-riwaya').filter({ hasText: "ا۬" }).first();
  await tasheel.click();
  await expect(page.locator(".tajweed-inspector")).toContainText("Tas-hīl");
  await expect(page.locator(".tajweed-inspector")).toContainText("book p. 93");
  await page.keyboard.press("Escape");
  await expect(page.locator(".tajweed-inspector")).toHaveCount(0);
});

test("tajweed page teaches letters, rules and the Qālūn topics", async ({ page }) => {
  await page.goto("/tajweed");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("Qālūn");
  await page.locator(".tj-letter-grid button", { hasText: "ق" }).click();
  await expect(page.locator(".tj-letter-card")).toContainText("Back of the tongue");
  await expect(page.locator(".tj-letter-card")).toContainText("Echo (qalqala)");
  await expect(page.locator("#rule-tasheel")).toContainText("book p. 93");
  const topic = page.locator("#topic-hamzatan_kalimatayn");
  await topic.locator("summary").click();
  await expect(topic).toContainText("Same vowel");
  await expect(topic.locator(".tj-found a").first()).toHaveAttribute("href", /\/studio\?surah=\d+&ayah=\d+/);
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});
