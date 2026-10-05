import { test, expect } from "@playwright/test";
import { makeChallenge, wordKey, type Verse, type ChallengeMode, type EmbeddingTable, type TextIndex } from "../lib/challenges";
const verses: Verse[] = [
  { surah: 1, name: "One", ayah: 1, text: "word one alpha", normalized: "word one alpha", audio: "/1.wav", duration: 3 },
  { surah: 1, name: "One", ayah: 2, text: "word two beta", normalized: "word two beta", audio: "/2.wav", duration: 4 },
  { surah: 1, name: "One", ayah: 3, text: "word three gamma", normalized: "word three gamma", audio: "/3.wav", duration: 5 },
  { surah: 2, name: "Two", ayah: 1, text: "different second surah", normalized: "different second surah" },
  { surah: 3, name: "Three", ayah: 1, text: "another third surah", normalized: "another third surah" },
];
for (const mode of ["next", "audio", "surah", "missing", "order"] as ChallengeMode[]) {
  test(`${mode} questions have exactly one answer and distinct choices`, () => {
    const q = makeChallenge(verses, mode, "hard", null, () => 0.25);
    expect(q.options).toHaveLength(3);
    expect(new Set(q.options.map(o => o.audio || o.label)).size).toBe(3);
    expect(q.answer).toBeGreaterThanOrEqual(0);
    expect(q.answer).toBeLessThan(3);
    if (mode === "audio") expect(q.options[q.answer].audio).toBe(verses.find(v => v.surah === q.surah && v.ayah === q.ayah)?.audio);
  });
}
test("ambiguous repeated ayahs are excluded from surah questions", () => {
  const pool = [...verses, { ...verses[0], surah: 2, ayah: 2 }];
  for (let i = 0; i < 20; i++) expect(makeChallenge(pool, "surah", "easy").target).not.toBe(verses[0].text);
});
test("all Quran text and local audio challenges load", async ({ page, request }) => {
  await page.goto("/games");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("A little challenge");
  await expect(page.getByRole("button", { name: "Use answer choices instead" })).toBeVisible();
  await page.getByRole("button", { name: "Use answer choices instead" }).click();
  await expect(page.getByRole("radio")).toHaveCount(3);
  await page.getByRole("radio").first().click();
  await page.getByRole("button", { name: "Check answer" }).click();
  await expect(page.locator(".challenge-feedback")).toBeVisible();
  await page.getByRole("button", { name: /Listen & match/ }).click();
  await expect(page.getByRole("button", { name: "Play audio 1", exact: true })).toBeVisible();
  const response = await request.get("/api/challenges?mode=audio&difficulty=hard&scope=all");
  expect(response.ok()).toBeTruthy();
  const question = await response.json();
  expect(question.similarity).toContain("MFCC");
  const audio = await request.get(question.options[0].audio);
  expect(audio.headers()["content-type"]).toBe("audio/wav");
  expect((await audio.body()).subarray(0, 4).toString()).toBe("RIFF");
  expect((await request.get("/api/reference-audio?surah=../../&ayah=1")).status()).toBe(400);
});
test("local profiles hash passwords and isolate progress", async ({ page }) => {
  await page.goto("/profile");
  await page.getByLabel("Name", { exact: true }).fill("Learner");
  await page.getByLabel("Demo password").fill("demo-passphrase");
  await page.getByRole("button", { name: "Create profile", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1 })).toContainText("Learner");
  const saved = await page.evaluate(() => localStorage.getItem("rattil.profiles.v1"));
  expect(saved).not.toContain("demo-passphrase");
  expect(JSON.parse(saved!)[0].passwordHash).toHaveLength(64);
  await page.evaluate(() => {
    const id = localStorage.getItem("rattil.active-profile.v1");
    localStorage.setItem(`rattil.learn.v1:${id}`, JSON.stringify({ xp: 42, days: [], lessons: {} }));
  });
  await page.reload();
  await expect(page.locator(".profile-stats")).toContainText("42 XP");
  await page.getByRole("button", { name: "Sign out" }).click();
  await page.getByRole("button", { name: "I have a profile" }).click();
  await page.getByLabel("Name", { exact: true }).fill("Learner");
  await page.getByLabel("Demo password").fill("incorrect-password");
  await page.getByRole("button", { name: "Sign in", exact: true }).click();
  await expect(page.locator('main [role="alert"]')).toContainText("Incorrect");
  await page.getByLabel("Demo password").fill("demo-passphrase");
  await page.getByRole("button", { name: "Sign in", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1 })).toContainText("Learner");
  await expect(page.locator(".profile-stats")).toContainText("42 XP");
  await page.getByRole("button", { name: "Sign out" }).click();
  await page.getByRole("button", { name: "Create a profile instead" }).click();
  await page.getByLabel("Name", { exact: true }).fill("Other learner");
  await page.getByLabel("Demo password").fill("other-demo-password");
  await page.getByRole("button", { name: "Create profile", exact: true }).click();
  await expect(page.locator(".profile-stats")).toContainText("0 XP");
});
test("challenge and profile pages fit on a phone", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  for (const url of ["/games", "/profile"]) {
    await page.goto(url);
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(false);
  }
});

// Five verses whose 2-d "embeddings" put 1:3 next to 2:1 in meaning and 3:1 far away.
const fakeIndex = (vectors: Record<string, [number, number]>): TextIndex => {
  const table = (ids: string[]): EmbeddingTable => ({
    index: new Map(ids.map((id, i) => [id, i])), dim: 2,
    values: Int8Array.from(ids.flatMap(id => vectors[id].map(x => Math.round(x * 127)))), scales: new Float32Array(ids.length).fill(1 / 127),
  });
  return { model: "fake", method: "test", ayahs: table(Object.keys(vectors)), words: table([]) };
};
const ring: Verse[] = Array.from({ length: 12 }, (_, i) => ({ surah: 10 + i, name: `S${i}`, ayah: 1, text: `ayah ${i}`, normalized: `ayah ${i}` }));
const vectors = Object.fromEntries(ring.map((v, i) => [`${v.surah}:1`, [Math.cos(i / 4), Math.sin(i / 4)] as [number, number]]));

test("text embeddings rank the closest ayahs as hard choices and keep easy ones further away", () => {
  const index = fakeIndex(vectors);
  const pool = [{ surah: 9, name: "Prompt", ayah: 1, text: "start", normalized: "start" }, { surah: 9, name: "Prompt", ayah: 2, text: "ayah 0", normalized: "ayah 0x" }, ...ring.slice(1)];
  index.ayahs.index.set("9:2", 0);
  const hard = makeChallenge(pool, "next", "hard", null, () => 0, "en", index);
  const hardIDs = hard.options.filter((_, i) => i !== hard.answer).map(o => ring.findIndex(v => v.text === o.label));
  expect(Math.max(...hardIDs)).toBeLessThanOrEqual(6);
  expect(hard.similarity).toContain("embeddings");
  const surahs = makeChallenge(ring, "surah", "hard", null, () => 0, "en", index);
  const picked = ring.findIndex(v => v.name === surahs.options[surahs.answer].label);
  for (const option of surahs.options) expect(Math.abs(ring.findIndex(v => v.name === option.label) - picked)).toBeLessThanOrEqual(4);
});

test("the prompt is never offered as its own next ayah", () => {
  const pool: Verse[] = [
    { surah: 1, name: "A", ayah: 1, text: "alif one", normalized: "alif one" },
    { surah: 1, name: "A", ayah: 2, text: "alif two", normalized: "alif two" },
    { surah: 2, name: "B", ayah: 1, text: "ba one", normalized: "ba one" },
    { surah: 3, name: "C", ayah: 1, text: "jim one", normalized: "jim one" },
  ];
  for (let i = 0; i < 20; i++) expect(makeChallenge(pool, "next", "hard").options.map(o => o.label)).not.toContain("alif one");
});

test("missing-word choices never differ from the answer only by vowels or Qālūn marks", () => {
  expect(wordKey("اَ۬لنَّاسِ")).toBe(wordKey("اِ۬لنَّاسُ"));
  // Same expectations as src/learning/test_text_embeddings.py: the index keys must match.
  for (const [word, key] of [["اَ۬لنَّاسِ", "الناس"], ["يَوْمَئِذٖ", "يوميذ"], ["أَعْمَٰلَهُمْ", "اعملهم"], ["اُ۬لْقُرْءَانَ", "القرءان"]]) expect(wordKey(word)).toBe(key);
  const pool: Verse[] = ["قُلْ أَعُوذُ بِرَبِّ اِ۬لنَّاسِ", "مَلِكِ اِ۬لنَّاسُ كُلِّهِمْ", "مِن شَرِّ اِ۬لْوَسْوَاسِ", "إِلَٰهِ اِ۬لنَّاسَ جَمِيعًا"]
    .map((text, i) => ({ surah: 114, name: "An-Nas", ayah: i + 1, text, normalized: text }));
  for (let i = 0; i < 30; i++) {
    const q = makeChallenge(pool, "missing", "hard");
    const keys = q.options.map(o => wordKey(o.label));
    expect(new Set(keys).size).toBe(3);
  }
});

test("ayah order distractors get closer with difficulty and carry each ayah separately", () => {
  const hard = makeChallenge(verses, "order", "hard", null, () => 0.25);
  const right = hard.options[hard.answer].parts!;
  expect(right).toEqual(["word one alpha", "word two beta", "word three gamma"]);
  for (const option of hard.options) {
    expect(option.parts).toHaveLength(3);
    expect(option.parts!.filter((part, i) => part === right[i]).length).toBeGreaterThanOrEqual(1);
  }
  const easy = makeChallenge(verses, "order", "easy", null, () => 0.25);
  expect(easy.options.filter((_, i) => i !== easy.answer).every(o => o.parts![0] !== right[0])).toBe(true);
});
