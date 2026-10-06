import { test, expect } from "@playwright/test";

test("local Quran assets and surah selection", async ({ page }) => {
  await page.goto("/studio");
  await expect(page.getByRole("button", { name: "Begin recitation" })).toBeEnabled();
  await page.getByRole("button", { name: "Mushaf", exact: true }).click();
  await expect(page.getByRole("img", { name: "Quran page 001" })).toBeVisible();
  await page.locator(".surah-select").click();
  await page.getByRole("textbox", { name: "Search surahs" }).fill("112");
  await page.locator(".surah-list button").click();
  await expect(page.locator(".surah-select")).toContainText("Al-Ikhlāṣ");
  await page.getByRole("button", { name: "Ayah view", exact: true }).click();
  await expect(page.locator(".text-ayah")).toHaveCount(4);
});

test("clearly labeled demo highlights and reset", async ({ page }) => {
  await page.goto("/studio");
  await expect(page.getByRole("button", { name: "Try the presentation demo" })).toBeEnabled();
  await page.getByRole("button", { name: "Try the presentation demo" }).click();
  await expect(page.locator(".demo-banner")).toContainText("No microphone or model inference");
  const firstAyah = page.locator('.text-ayah[data-ayah="1"]');
  await expect(firstAyah.locator('[data-word-index="0"]')).toHaveAttribute("data-status", "correct");
  await expect(firstAyah.locator('[data-word-index="1"]')).toHaveAttribute("data-status", "missed");
  await expect(firstAyah.locator('[data-word-index="3"]')).toHaveAttribute("data-status", "pending");
  await expect(page.locator(".text-ayah.correct")).toHaveCount(2, { timeout: 8000 });
  await expect(page.locator(".text-ayah.missed")).toHaveCount(1, { timeout: 4000 });
  await page.getByRole("button", { name: "Finish recitation" }).click();
  await page.getByRole("button", { name: "Reset session" }).click();
  await expect(page.locator(".text-ayah.correct")).toHaveCount(0);
  await expect(page.locator(".text-mushaf .quran-word.correct, .text-mushaf .quran-word.missed")).toHaveCount(0);
  await expect(page.locator(".demo-banner")).toHaveCount(0);
});

test("mobile layout stays inside viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/studio");
  await expect(page.getByRole("button", { name: "Begin recitation" })).toBeEnabled();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
  expect(overflow).toBe(false);
});

test("help explains score limitations", async ({ page }) => {
  await page.goto("/studio");
  await page.getByRole("button", { name: "How it works", exact: true }).first().click();
  await expect(page.getByRole("dialog")).toContainText("not model probabilities");
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toHaveCount(0);
});

test("invalid uploads show a useful error and allow retry", async ({ page }) => {
  await page.goto("/studio");
  await expect(page.getByRole("button", { name: "Upload audio to test" })).toBeEnabled();
  await page.getByLabel("Choose audio recording").setInputFiles({ name: "invalid.wav", mimeType: "audio/wav", buffer: Buffer.from("not an audio recording") });
  await expect(page.locator(".error-message")).toContainText("Cannot decode this audio file");
  await expect(page.getByRole("button", { name: "Upload audio to test" })).toBeEnabled();
});

test("memorize hides each word until it is heard or hinted", async ({ page }) => {
  await page.goto("/studio");
  await page.getByLabel("Hide ayahs to memorize").check();
  await expect(page.getByRole("button", { name: "Mushaf", exact: true })).toBeDisabled();
  const words = page.locator('.text-ayah[data-ayah="1"] .quran-word');
  const hidden = "rgba(0, 0, 0, 0)";
  await expect(words.nth(0)).toHaveCSS("color", hidden);
  await expect(page.locator(".basmalah .quran-word").first()).not.toHaveCSS("color", hidden);
  await page.getByRole("button", { name: "Hint: show the next word" }).click();
  await expect(words.nth(0)).toHaveClass(/hinted/);
  await expect(words.nth(0)).not.toHaveCSS("color", hidden);
  await expect(words.nth(1)).toHaveCSS("color", hidden);
  await expect(page.getByText("1 hint used")).toBeVisible();
  await page.getByRole("button", { name: "Try the presentation demo" }).click();
  // Heard words appear; a missed word appears too (in red) so the learner sees what was skipped.
  await expect(words.nth(0)).toHaveAttribute("data-status", "correct");
  await expect(words.nth(1)).toHaveAttribute("data-status", "missed");
  await expect(words.nth(1)).not.toHaveCSS("color", hidden);
  await expect(words.nth(3)).toHaveCSS("color", hidden);
  await page.reload();
  await expect(page.getByLabel("Hide ayahs to memorize")).toBeChecked();
});
