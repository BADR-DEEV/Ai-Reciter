import { test, expect } from "@playwright/test";
import { emptySkill, recordChoice, recommendDifficulty, nextSkill } from "../lib/adaptive-challenges";

test("personalized difficulty starts gently, grows gradually and backs off errors", () => {
  let skill = emptySkill();
  for (let i = 0; i < 4; i++) skill = recordChoice(skill, true);
  expect(recommendDifficulty(skill).difficulty).toBe("easy");
  skill = recordChoice(skill, true);
  expect(recommendDifficulty(skill).difficulty).toBe("medium");
  skill = { ...skill, difficulty: "medium" };
  skill = recordChoice(recordChoice(skill, false), false);
  expect(recommendDifficulty(skill).difficulty).toBe("easy");
  expect(recommendDifficulty({ ...skill, difficulty: "hard" }).difficulty).toBe("medium");
});

test("history is bounded and does not use XP as competence", () => {
  let skill = emptySkill();
  for (let i = 0; i < 100; i++) skill = recordChoice(skill, i % 2 === 0);
  expect(skill.outcomes).toHaveLength(12);
  expect(skill.attempts).toBe(100);
  expect(recommendDifficulty(skill).difficulty).toBe("easy");
});

test("each difficulty level needs fresh evidence and does not promote repeatedly", () => {
  let skill = emptySkill();
  for (let i = 0; i < 5; i++) skill = recordChoice(skill, true);
  skill = nextSkill(skill);
  expect(skill.difficulty).toBe("medium");
  expect(skill.outcomes).toEqual([]);
  expect(nextSkill(skill).difficulty).toBe("medium");
  expect(skill.attempts).toBe(5);
  skill = recordChoice(recordChoice(skill, false), false);
  expect(nextSkill(skill).difficulty).toBe("easy");
});

test("choice policy is opt-in and changes difficulty only on the next question", async ({ page }) => {
  await page.route("**/api/challenges?**", route => route.fulfill({ json: {
    id: "adaptive-test", mode: "surah", difficulty: "easy", prompt: "قل هو الله أحد", reference: "112:1",
    surah: 112, ayah: 1, options: [{ label: "Al-Ikhlas" }, { label: "Al-Falaq" }, { label: "An-Nas" }],
    answer: 0, target: "قل هو الله أحد", explanation: "Test item", similarity: "text test fixture",
  } }));
  await page.goto("/games");
  await page.getByRole("button", { name: "Find the surah", exact: false }).click();
  const choose = async () => {
    await page.getByRole("radio", { name: "Al-Ikhlas", exact: true }).click();
    await page.getByRole("button", { name: "Check answer", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Well remembered!" })).toBeVisible();
  };
  const key = "rattil.learn.v1:adaptive-v1:surah:amma:text";
  await choose();
  expect(await page.evaluate(k => localStorage.getItem(k), key)).toBeNull();
  await page.getByRole("button", { name: "Next challenge", exact: true }).click();
  await page.getByLabel("Personalized difficulty · prototype").check();
  for (let i = 0; i < 5; i++) {
    await choose();
    await expect(page.getByLabel("Challenge difficulty")).toHaveValue("easy");
    await page.getByRole("button", { name: "Next challenge", exact: true }).click();
  }
  await expect(page.getByLabel("Challenge difficulty")).toHaveValue("medium");
  await expect(page.getByLabel("Challenge difficulty")).toBeDisabled();
  const state = await page.evaluate(k => JSON.parse(localStorage.getItem(k)!), key);
  expect(state.attempts).toBe(5);
  expect(state.outcomes).toEqual([]);
  await page.getByRole("button", { name: "Show answer · no XP", exact: true }).click();
  await page.getByRole("button", { name: "Next challenge", exact: true }).click();
  expect(await page.evaluate(k => JSON.parse(localStorage.getItem(k)!).attempts, key)).toBe(5);
});

test("learner histories are isolated and deleted only with their profile", async ({ page }) => {
  await page.goto("/profile");
  await page.getByLabel("Name", { exact: true }).fill("Adaptive Learner");
  await page.getByLabel("Demo password").fill("demo-passphrase");
  await page.getByRole("button", { name: "Create profile", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1 })).toContainText("Adaptive Learner");
  const ownKey = await page.evaluate(() => {
    const id = localStorage.getItem("rattil.active-profile.v1");
    const key = `rattil.learn.v1:${id}:adaptive-v1:surah:amma:text`;
    localStorage.setItem(key, "private-profile-history");
    localStorage.setItem("rattil.learn.v1:adaptive-v1:surah:amma:text", "guest-history");
    localStorage.setItem("rattil.learn.v1:other-profile:adaptive-v1:surah:amma:text", "other-history");
    return key;
  });
  page.once("dialog", dialog => dialog.accept());
  await page.getByRole("button", { name: "Delete local profile", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1 })).toContainText("Your practice profile");
  const stored = await page.evaluate(key => ({ own: localStorage.getItem(key),
    guest: localStorage.getItem("rattil.learn.v1:adaptive-v1:surah:amma:text"),
    other: localStorage.getItem("rattil.learn.v1:other-profile:adaptive-v1:surah:amma:text") }), ownKey);
  expect(stored).toEqual({ own: null, guest: "guest-history", other: "other-history" });
});
