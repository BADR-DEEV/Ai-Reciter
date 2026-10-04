import type { Difficulty } from "./challenges";

/** Explainable cold-start learner policy, NOT a trained Duolingo/neural model. */
export type SkillHistory = { outcomes: boolean[]; attempts: number; difficulty: Difficulty };
export const emptySkill = (): SkillHistory => ({ outcomes: [], attempts: 0, difficulty: "easy" });
const levels: Difficulty[] = ["easy", "medium", "hard"];
export function recordChoice(skill: SkillHistory, correct: boolean): SkillHistory {
  return { ...skill, attempts: skill.attempts + 1, outcomes: [...skill.outcomes, correct].slice(-12) };
}
export function recommendDifficulty(skill: SkillHistory): { difficulty: Difficulty; reason: "cold-start" | "grow" | "support" | "steady" } {
  const at = levels.indexOf(skill.difficulty);
  if (skill.outcomes.length >= 2 && skill.outcomes.slice(-2).every(value => !value))
    return { difficulty: levels[Math.max(0, at - 1)], reason: "support" };
  if (skill.outcomes.length < 5) return { difficulty: skill.difficulty, reason: "cold-start" };
  const accuracy = skill.outcomes.filter(Boolean).length / skill.outcomes.length;
  if (accuracy < .55)
    return { difficulty: levels[Math.max(0, at - 1)], reason: "support" };
  if (accuracy >= .8 && skill.outcomes.slice(-3).every(Boolean))
    return { difficulty: levels[Math.min(2, at + 1)], reason: "grow" };
  return { difficulty: skill.difficulty, reason: "steady" };
}

/** A new level needs fresh evidence, not repeated promotion from old successes. */
export function nextSkill(skill: SkillHistory): SkillHistory {
  const { difficulty } = recommendDifficulty(skill);
  return difficulty === skill.difficulty ? skill : { ...skill, difficulty, outcomes: [] };
}
