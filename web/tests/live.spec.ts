import { test, expect } from "@playwright/test";

const audio = process.env.RECITER_LIVE_TEST_AUDIO;
test.use({
  permissions: ["microphone"],
  launchOptions: { args: audio ? ["--use-fake-device-for-media-stream", `--use-file-for-fake-audio-capture=${audio}`, "--use-fake-ui-for-media-stream"] : [] },
});

test("microphone worklet streams real audio to local GPU", async ({ page }) => {
  test.skip(!audio, "Set RECITER_LIVE_TEST_AUDIO to an absolute PCM WAV fixture and start the backend");
  test.setTimeout(45000);
  await page.goto("/studio");
  await expect(page.getByRole("button", { name: "Begin recitation", exact: true })).toBeEnabled();
  await page.getByRole("button", { name: "Begin recitation", exact: true }).click();
  await expect(page.getByRole("heading", { name: "We’re listening." })).toBeVisible();
  await expect(page.getByLabel("Reference reciter")).toBeDisabled();
  await expect(page.getByRole("button", { name: "Listen to ayah 1", exact: true })).toBeDisabled();
  await expect(page.locator('.text-ayah[data-ayah="1"]')).toHaveClass(/correct/, { timeout: 18000 });
  await expect(page.locator('.text-ayah[data-ayah="2"]')).toHaveClass(/missed/, { timeout: 15000 });
  await expect(page.locator('.text-ayah[data-ayah="1"] .quran-word.correct')).toHaveCount(4);
  await expect(page.locator('.text-ayah[data-ayah="2"] .quran-word.missed')).toHaveCount(2);
  await expect(page.locator('.text-ayah[data-ayah="5"] .quran-word.pending')).toHaveCount(3);
  await expect(page.locator('.text-ayah[data-ayah="3"]')).toHaveClass(/correct/);
  await page.getByRole("button", { name: "Finish recitation" }).click();
  await expect(page.getByRole("button", { name: "Begin recitation", exact: true })).toBeEnabled();
  await expect(page.locator(".error-message")).toHaveCount(0);
});

test("uploaded recording uses the GPU without requesting a microphone", async ({ page }) => {
  test.skip(!audio, "Set RECITER_LIVE_TEST_AUDIO and start the backend");
  test.setTimeout(60000);
  await page.addInitScript(() => {
    navigator.mediaDevices.getUserMedia = async () => { throw new Error("Upload must not request a microphone"); };
  });
  await page.goto("/studio");
  await expect(page.getByRole("button", { name: "Upload audio to test" })).toBeEnabled();
  await page.getByRole("button", { name: "Upload audio to test" }).click();
  await page.getByLabel("Choose audio recording").setInputFiles(audio!);
  await expect(page.locator(".upload-file-note")).toContainText("Accelerated local processing");
  await expect(page.locator('.text-ayah[data-ayah="1"]')).toHaveClass(/correct/, { timeout: 18000 });
  await expect(page.locator('.text-ayah[data-ayah="2"]')).toHaveClass(/missed/, { timeout: 15000 });
  await expect(page.getByRole("button", { name: "Upload audio to test" })).toBeEnabled({ timeout: 30000 });
  await expect(page.locator(".error-message")).toHaveCount(0);
});
