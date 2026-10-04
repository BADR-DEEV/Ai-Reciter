/** Whitelisted, actual Qaloon recordings. Never substitute another riwayah. */
export const RECITERS = [
  { id: "huthaify", name: "Al-Huthaify", arabic: "علي الحذيفي", folder: "dataset_qaloon_hutafi" },
  { id: "husary", name: "Al-Husary", arabic: "محمود خليل الحصري", folder: "dataset_qaloon_Husary" },
  { id: "dokali", name: "Al-Dokali", arabic: "الدكالي محمد العالم", folder: "dataset_qaloon_dokali" },
] as const;
export type ReciterID = typeof RECITERS[number]["id"];
export const DEFAULT_RECITER: ReciterID = "huthaify";
export const isReciter = (id: string): id is ReciterID => RECITERS.some(r => r.id === id);
export const referenceURL = (surah: number, ayah: number, reciter: ReciterID) => `/api/reference-audio?surah=${surah}&ayah=${ayah}&reciter=${reciter}`;
