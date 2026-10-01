import { test, expect } from "@playwright/test";

const SHOTS = process.env.LEARN_SCREENSHOTS;

test("landing invites beginners and shows the alphabet", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("Read the Quran aloud");
  await expect(page.locator(".alpha-tile")).toHaveCount(29);
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/landing.png`, fullPage: true });
  await page.getByRole("button", { name: "العربية" }).click();
  await expect(page.getByRole("heading", { level: 1 })).toContainText("اقرأ القرآن");
  await expect(page.locator(".learn-shell")).toHaveAttribute("dir", "rtl");
  await page.getByRole("button", { name: "English" }).click();
  await page.getByRole("link", { name: "Start learning" }).click();
  await expect(page).toHaveURL(/\/learn\/welcome$/);
});

test("a lesson can be completed and is remembered", async ({ page }) => {
  await page.goto("/learn/welcome");
  for (let i = 0; i < 4; i++) await page.getByRole("button", { name: "Continue" }).click();
  // Answer wrongly once: the question is repeated at the end.
  await page.getByRole("radio", { name: "On the left" }).click();
  await page.getByRole("button", { name: "Check" }).click();
  await expect(page.locator(".lesson-foot.bad")).toContainText("Answer: On the right");
  await page.getByRole("button", { name: "Continue" }).click();
  await page.getByRole("radio", { name: "Short vowels" }).click();
  await page.getByRole("button", { name: "Check" }).click();
  await expect(page.locator(".lesson-foot.good")).toBeVisible();
  await page.getByRole("button", { name: "Continue" }).click();
  await expect(page.locator(".repeat-note")).toBeVisible();
  await page.getByRole("radio", { name: "On the right" }).click();
  await page.getByRole("button", { name: "Check" }).click();
  await page.getByRole("button", { name: "Continue" }).click();
  await expect(page.getByRole("heading", { name: "Lesson complete" })).toBeVisible();
  await expect(page.locator(".done-stats")).toContainText("50%");
  await page.getByRole("link", { name: "Back to course" }).click();
  await expect(page.locator(".lesson-list li.done")).toHaveCount(1);
  await expect(page.locator(".lesson-list li.next")).toContainText("Alif and the dotted family");
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/course.png`, fullPage: true });
});

test("speaking steps can be skipped when the model is offline", async ({ page }) => {
  await page.route("**/health", route => route.abort());
  await page.goto("/learn/surah-112");
  await page.getByRole("button", { name: "Continue" }).click();
  await expect(page.locator(".ayah-text")).toBeVisible();
  await expect(page.locator(".say-offline")).toBeVisible();
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/ayah.png`, fullPage: true });
  await page.getByRole("button", { name: "Skip" }).click();
  await expect(page.locator(".lesson-count")).toHaveText("3/6");
});

test("letter lesson renders on a phone without overflow", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/learn/letters-2");
  await expect(page.locator(".letter-glyph")).toHaveText("ج");
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/letter-mobile.png`, fullPage: true });
  expect(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth)).toBe(false);
  for (const path of ["/", "/learn"]) {
    await page.goto(path);
    expect(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth)).toBe(false);
  }
});
