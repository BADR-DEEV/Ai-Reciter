import { test, expect } from "@playwright/test";
test("tafsir UI supports both languages, plain text and graceful failure", async ({ page }) => {
  await page.route("**/api/tafsir?**", async route => {
    const lang = new URL(route.request().url()).searchParams.get("lang");
    await route.fulfill({ json: { kind: lang === "ar" ? "Arabic tafsir" : "Translation of meanings", book: { name: "Fixture book", author: "Fixture author" }, entries: [{ providerAyah: 2, text: "<script>not executable</script> Commentary fixture" }], mappingNote: "Hafs-numbered provider / Qaloon text mapping" } });
  });
  await page.goto("/studio");
  await page.getByRole("button", { name: "Meaning & tafsir · Ayah 1" }).click();
  await expect(page.locator(".tafsir-panel")).toContainText("Translation of meanings");
  await expect(page.locator(".tafsir-panel script")).toHaveCount(0);
  await page.getByLabel("Tafsir language").selectOption("ar");
  await expect(page.locator(".tafsir-panel h3")).toContainText("التفسير العربي");
  await expect(page.locator('.tafsir-panel p[lang="ar"]')).toBeVisible();
});
