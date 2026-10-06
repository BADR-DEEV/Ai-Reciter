// First ayah of each juzʾ in this app's Qālūn numbering. Mirrors JUZ_STARTS in
// src/streaming/search.py (a Python test keeps the two identical).
export const JUZ_STARTS: readonly (readonly [number, number])[] = [
  [1, 1], [2, 141], [2, 251], [3, 93], [4, 24], [4, 147], [5, 84], [6, 112], [7, 87], [8, 41],
  [9, 94], [11, 6], [12, 53], [15, 1], [17, 1], [18, 74], [21, 1], [23, 1], [25, 21], [27, 58],
  [29, 46], [33, 31], [36, 27], [39, 31], [41, 46], [46, 1], [51, 31], [58, 1], [67, 1], [78, 1],
];

export const JUZ_NUMBERS = JUZ_STARTS.map((_, i) => i + 1);
