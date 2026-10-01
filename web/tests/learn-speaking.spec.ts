import { test, expect } from "@playwright/test";

// A fake microphone and a mocked model: checks the client contract and feedback,
// not model quality.
// Headless Chromium's fake capture device can hang on macOS, so the microphone
// is a generated tone instead.
test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    navigator.mediaDevices.getUserMedia = async () => {
      const context = new AudioContext();
      const tone = context.createOscillator();
      const gain = context.createGain();
      gain.gain.value = 0.3;
      const out = context.createMediaStreamDestination();
      tone.connect(gain).connect(out);
      tone.start();
      return out.stream;
    };
  });
});

test("explains which confusable sound the model heard", async ({ page }) => {
  let request: { mode: string; target: string; alternatives: string[]; audio: string } | undefined;
  await page.route("**/health", route => route.fulfill({ json: { ready: true } }));
  await page.route("**/api/practice", async route => {
    request = route.request().postDataJSON();
    await route.fulfill({ json: { verdict: "other", heard: 1, confidence: 0.2, probabilities: [0.2, 0.8], transcript: "ها" } });
  });
  await page.goto("/learn/pairs-throat");
  for (let i = 0; i < 3; i++) await page.getByRole("button", { name: "Continue" }).click();
  for (let i = 0; i < 6; i++) {
    await page.getByRole("radio").first().click();
    await page.getByRole("button", { name: "Check" }).click();
    await page.getByRole("button", { name: "Continue" }).click();
  }
  await expect(page.locator(".say-arabic")).toHaveText("حَا");
  await expect(page.getByRole("button", { name: "Continue" })).toHaveCount(0);
  await page.getByRole("button", { name: "Your turn: say it" }).click();
  await page.waitForTimeout(700);
  await page.getByRole("button", { name: "Stop" }).click();
  const feedback = page.locator(".say-feedback.other");
  await expect(feedback).toContainText("closer to");
  await expect(feedback).toContainText("Hā’");
  expect(request?.mode).toBe("sound");
  expect(request?.target).toBe("حَا");
  expect(request?.alternatives).toEqual(["هَا"]);
  expect(request!.audio.length).toBeGreaterThan(1000);
  await expect(page.getByRole("button", { name: "Hear yourself" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Continue" })).toBeEnabled();
  if (process.env.LEARN_SCREENSHOTS) await page.screenshot({ path: `${process.env.LEARN_SCREENSHOTS}/say-feedback.png`, fullPage: true });
});
