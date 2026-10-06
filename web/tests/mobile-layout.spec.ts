import { test, expect } from "@playwright/test";

test("phone navigation stays reachable on every main screen in both languages", async ({ page }) => {
  for (const width of [320, 390]) {
    await page.setViewportSize({ width, height: 844 });
    for (const route of ["/", "/learn", "/learn/welcome", "/studio", "/search", "/tajweed", "/games", "/profile"]) {
      await page.goto(route);
      for (let language = 0; language < 2; language++) {
        const nav = page.locator(".mobile-nav");
        await expect(nav).toBeVisible();
        for (const link of await nav.locator("a:visible, button:visible").all()) {
          await expect(link).toBeInViewport({ ratio: 1 });
        }
        await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
        const bounds = await nav.boundingBox();
        expect(bounds!.y + bounds!.height).toBeCloseTo(844, 0);
        expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
        await page.locator(".lh-lang").click();
      }
    }
  }
});

test("phone recording buttons stay centered in the app dock only on audio screens", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  for (const [route, selector] of [["/studio", ".start-button"], ["/search", ".find-star"]]) {
    await page.goto(route);
    const button = page.locator(`.mobile-audio-slot ${selector}`);
    await expect(button).toBeVisible();
    await expect(page.locator(".mobile-audio-slot button")).toHaveCount(1);
    for (const scroll of [0, 1000]) {
      await page.evaluate(y => window.scrollTo(0, y), scroll);
      const box = await button.boundingBox();
      const nav = await page.locator(".mobile-nav").boundingBox();
      expect(box!.x + box!.width / 2).toBeCloseTo(195, 0);
      expect(box!.y).toBeLessThan(nav!.y);
      expect(box!.y + box!.height).toBeLessThan(nav!.y + nav!.height);
      expect(box!.width).toBe(64);
      const icon = await button.locator("svg").boundingBox();
      expect(icon!.x + icon!.width / 2).toBeCloseTo(box!.x + box!.width / 2, 0);
      expect(icon!.y + icon!.height / 2).toBeCloseTo(box!.y + box!.height / 2, 0);
      await expect(button).toBeInViewport({ ratio: 1 });
    }
  }
  await page.setViewportSize({ width: 1280, height: 900 });
  await expect(page.locator(".find-stage .find-star")).toHaveCSS("position", "relative");
  await expect(page.locator(".mobile-nav")).toBeHidden();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/learn/welcome");
  await expect(page.locator(".mobile-audio-slot button")).toHaveCount(0);
  await expect(page.locator('.mobile-nav a[href="/studio"]')).toBeVisible();
  await expect(page.locator('.mobile-nav a[href="/learn"]')).toHaveCount(0);
});

test("phone More menu exposes remaining pages and supports keyboard dismissal", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/studio");
  const more = page.locator(".mobile-nav > button");
  await more.click();
  const dialog = page.getByRole("dialog", { name: "More navigation" });
  await expect(dialog).toBeVisible();
  for (const href of ["/", "/learn", "/tajweed", "/profile"]) {
    await expect(dialog.locator(`a[href="${href}"]`)).toBeInViewport({ ratio: 1 });
  }
  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
  await expect(more).toBeFocused();
  await more.click();
  await dialog.getByRole("link", { name: "Profile" }).click();
  await expect(page).toHaveURL(/\/profile$/);
  await expect(page.locator(".mobile-nav > button")).toHaveClass("active");
});

test("phone dock follows the requested five-slot order", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/studio");
  expect(await page.locator(".mobile-nav").evaluate(nav => [...nav.children].map(child => child.getAttribute("href") || child.id || child.tagName))).toEqual([
    "/studio", "/search", "mobile-audio-slot", "/games", "BUTTON",
  ]);
});

test("phone studio options collapse and reader views use a dropdown", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/studio");
  const options = page.locator(".studio-options");
  await expect(options).not.toHaveAttribute("open");
  await options.locator("summary").click();
  await page.getByLabel("Show draft Qālūn phonetics").check();
  await options.locator("summary").click();
  await expect(options).not.toHaveAttribute("open");
  await page.getByRole("combobox", { name: "Reader view", exact: true }).selectOption("phonetic");
  await expect(page.locator(".phonetic-text")).toBeVisible();
  await options.locator("summary").click();
  await expect(page.getByLabel("Show draft Qālūn phonetics")).toBeChecked();
});

test("phone surah picker defaults to trained coverage and gates the full Quran", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/studio");
  await page.locator(".surah-select").click();
  const toggle = page.getByLabel("Show the full Quran · work in progress");
  await expect(toggle).not.toBeChecked();
  await expect(page.locator(".surah-list > button")).toHaveCount(38);
  const numbers = await page.locator(".surah-list .list-number").allTextContents();
  expect(numbers.every(number => Number(number) === 1 || Number(number) >= 78)).toBe(true);
  await toggle.check();
  await expect(page.locator(".surah-list > button")).toHaveCount(114);
  await expect(page.locator(".picker-scope-warning")).toContainText("work in progress");
  await toggle.uncheck();
  await expect(page.locator(".surah-list > button")).toHaveCount(38);
  await page.locator(".surah-list > button").filter({ has: page.locator(".list-number", { hasText: /^78$/ }) }).click();
  const reader = page.locator(".text-mushaf");
  await expect(page.locator('.text-mushaf [data-ayah="40"]')).toBeVisible();
  await expect(reader).toHaveCSS("max-height", "none");
  await expect(reader).toHaveCSS("overflow-y", "visible");
  expect(await reader.evaluate(element => element.scrollHeight - element.clientHeight)).toBeLessThanOrEqual(1);
  expect((await reader.boundingBox())!.height).toBeGreaterThan(844);
});

test("phone challenge dropdowns share rows and whole Quran requires opt-in", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/games");
  const difficulty = page.getByLabel("Challenge difficulty");
  const scope = page.getByLabel("Challenge scope");
  await expect(scope.locator("option")).toHaveCount(1);
  expect(Math.abs((await difficulty.boundingBox())!.y - (await scope.boundingBox())!.y)).toBeLessThan(1);
  const choices = page.getByLabel("Where wrong answers come from");
  const reciter = page.getByLabel("Reference reciter");
  expect(Math.abs((await choices.boundingBox())!.y - (await reciter.boundingBox())!.y)).toBeLessThan(1);
  await page.getByLabel("Enable full Quran · work in progress").check();
  await expect(scope.locator("option")).toHaveCount(2);
  await expect(page.locator('.safety-note[role="status"]')).toContainText("work in progress");
  await scope.selectOption("all");
  await choices.selectOption("quran");
  await page.getByLabel("Enable full Quran · work in progress").uncheck();
  await expect(scope).toHaveValue("amma");
  await expect(choices).toHaveValue("scope");
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test("desktop also requires opting into the full Quran", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.goto("/studio");
  await page.locator(".surah-select").click();
  await expect(page.locator(".surah-list > button")).toHaveCount(38);
  await page.getByLabel("Show the full Quran · work in progress").check();
  await expect(page.locator(".surah-list > button")).toHaveCount(114);
  await expect(page.locator(".picker-scope-warning")).toBeVisible();
  await page.getByLabel("Show the full Quran · work in progress").uncheck();
  await expect(page.locator(".surah-list > button")).toHaveCount(38);
  await page.goto("/games");
  await expect(page.getByLabel("Challenge scope").locator("option")).toHaveCount(1);
  await page.getByLabel("Enable full Quran · work in progress").check();
  await expect(page.getByLabel("Challenge scope").locator("option")).toHaveCount(2);
  await expect(page.locator('.safety-note[role="status"]')).toBeVisible();
});

test("next challenge on a phone stays at the question rather than the page header", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  let round = 0;
  await page.route("**/api/challenges?**", async route => {
    round++;
    if (round > 1) await new Promise(resolve => setTimeout(resolve, 350));
    await route.fulfill({ json: {
      id: `round-${round}`, mode: "next", difficulty: "easy", prompt: "قُلْ هُوَ اللَّهُ أَحَدٌ", reference: `Question ${round}`,
      options: [{ label: "اللَّهُ الصَّمَدُ" }, { label: "قُلْ أَعُوذُ" }, { label: "مِنْ شَرِّ" }],
      answer: 0, target: "اللَّهُ الصَّمَدُ", explanation: "Review the next ayah.", similarity: "Test", surah: 112, ayah: 2,
    } });
  });
  await page.goto("/games");
  await expect(page.locator(".challenge-question-head")).toContainText("Question 1");
  await page.getByRole("button", { name: "Show answer · no XP" }).click();
  await page.getByRole("button", { name: "Next challenge", exact: true }).click();
  expect(await page.evaluate(() => window.scrollY)).toBeGreaterThan(100);
  await expect(page.locator(".challenge-question-head")).toContainText("Question 2");
  await expect.poll(async () => (await page.locator(".challenge-round").boundingBox())!.y).toBeLessThan(120);
  expect((await page.locator(".challenge-round").boundingBox())!.y).toBeGreaterThanOrEqual(0);
  await expect(page.locator(".challenge-round")).toBeFocused();
});

test("phone voice search retains its decorative main button as well as the dock microphone", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/search");
  const main = page.locator(".find-stage .find-star");
  await expect(main).toBeVisible();
  await expect(page.locator(".mobile-audio-slot .find-star")).toBeVisible();
  expect((await main.boundingBox())!.width).toBeGreaterThan(100);
  await expect(main.locator(".find-star-ring").first()).toHaveCSS("display", "block");
});

test("speaking lesson uses the center microphone and preserves recording actions", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.route("**/health", route => route.fulfill({ json: { ready: true } }));
  await page.route("**/api/practice", route => route.fulfill({ json: {
    verdict: "correct", score: 1, transcript: "قل هو الله احد",
    words: [{ index: 0, text: "قل", status: "correct", heard: "قل" }],
  } }));
  await page.addInitScript(() => {
    navigator.mediaDevices.getUserMedia = async () => {
      const context = new AudioContext();
      const tone = context.createOscillator();
      const output = context.createMediaStreamDestination();
      tone.connect(output);
      tone.start();
      return output.stream;
    };
  });
  await page.goto("/learn/surah-112");
  await expect(page.locator(".mobile-audio-slot button")).toHaveCount(0);
  await page.getByRole("button", { name: "Continue", exact: true }).click();
  const microphone = page.locator(".mobile-audio-slot .btn-mic");
  await expect(microphone).toBeEnabled();
  await expect(microphone).toBeInViewport({ ratio: 1 });
  const footer = await page.locator(".lesson-foot").boundingBox();
  const dock = await page.locator(".mobile-nav").boundingBox();
  expect(footer!.y + footer!.height).toBeLessThanOrEqual(dock!.y);
  await microphone.click();
  await expect(microphone).toHaveClass(/live/);
  await page.waitForTimeout(700);
  await microphone.click();
  await expect(page.locator(".say-feedback.correct")).toBeVisible();
  await expect(microphone).toContainText("Try again");
  await page.getByRole("button", { name: "Continue", exact: true }).click();
  await expect(page.locator(".mobile-audio-slot button")).toHaveCount(1);
});
