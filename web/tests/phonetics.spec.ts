import { test, expect } from "@playwright/test";
import { qaloonG2P } from "../lib/qaloon-g2p";

test("rule fixtures retain Qaloon malik, vowels and pause behavior", () => {
  expect(qaloonG2P("مَلِكِ يَوْمِ اِ۬لدِّينِ").text).toBe("maliki yawmi d-dīn");
  expect(qaloonG2P("مَٰلِكِ يَوْمِ ٱلدِّينِ").text).toBe("māliki yawmi d-dīn");
  expect(qaloonG2P("اِ۬لْحَمْدُ لِلهِ رَبِّ اِ۬لْعَٰلَمِينَ ١").text).toBe("al-ḥamdu lillāhi rabbi l-ʿālamīn");
  expect(qaloonG2P("اَ۬لرَّحْمَٰنِ اِ۬لرَّحِيمِ").text).toBe("ar-raḥmāni r-raḥīm");
  expect(qaloonG2P("اُ۪هْدِنَا اَ۬لصِّرَٰطَ اَ۬لْمُسْتَقِيمَ").text).toBe("ihdinā ṣ-ṣirāṭa l-mustaqīm");
  expect(qaloonG2P("اَلذِينَ", "connect").text).toBe("alladhīna");
  expect(qaloonG2P("وَلَا اَ۬لضَّآلِّينَ").text).toBe("walā ḍ-ḍāllīn");
  expect(qaloonG2P("بِسْمِ اَ۬للَّهِ").text).toBe("bismi llāh");
  expect(qaloonG2P("قُلْ هُوَ اَ۬للَّهُ أَحَدٌ").text).toBe("qul huwa llāhu ʾaḥad");
  expect(qaloonG2P("مَلِكِ", "connect").text).toBe("maliki");
  // Tanwin fatḥ written on the final alif, not a hamza before "an".
  expect(qaloonG2P("فَالْمُغِيرَٰتِ صُبْحاٗ").text).toBe("fal-mughīrāti ṣubḥā");
  expect(qaloonG2P("فَالْمُغِيرَٰتِ صُبْحاٗ", "connect").text).toBe("fal-mughīrāti ṣubḥan");
});

test("uncertain marks and disjoint letters are reviewable, never approved", () => {
  const letters = qaloonG2P("أَلَٓمِّٓ");
  expect(letters.text).toBe("alif lām mīm");
  expect(letters.warnings.join(" ")).toContain("Disjoint letters");
  expect(qaloonG2P("هُمْ").warnings.join(" ")).toContain("Mim al-jam");
  expect(qaloonG2P("نَبِيِّۧنَ").warnings.join(" ")).toContain("U+6E7");
  expect(qaloonG2P("text").warnings.length).toBeGreaterThan(0);
  expect(qaloonG2P("قُلْ").status).toBe("draft");
});

test("studio exposes opt-in phonetics and resume controls", async ({ page }) => {
  await page.goto("/studio");
  await expect(page.getByLabel("Starting ayah")).toBeVisible();
  await page.getByLabel("Show draft Qālūn phonetics").check();
  await expect(page.locator('[data-ayah="3"] .phonetic-aid')).toContainText("maliki yawmi d-dīn");
  await page.getByLabel("Starting ayah").selectOption("3");
  await expect(page.locator(".current-card")).toContainText("Ayah 3");
});
