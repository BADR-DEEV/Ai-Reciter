/** Local prototype profiles, NOT authentication or protection from device users/XSS. */
export type LocalProfile = { id: string; name: string; salt: string; passwordHash: string; iterations: number };
const KEY = "rattil.profiles.v1";
const ACTIVE = "rattil.active-profile.v1";
export const PROFILE_EVENT = "rattil:profile";

export function profiles(): LocalProfile[] {
  try {
    const rows: unknown = JSON.parse(localStorage.getItem(KEY) || "[]");
    return Array.isArray(rows) ? rows.filter((r): r is LocalProfile => r && typeof r.id === "string" && /^[\w-]{1,64}$/.test(r.id) && typeof r.name === "string" && typeof r.salt === "string" && typeof r.passwordHash === "string" && r.iterations === 210000) : [];
  } catch { return []; }
}
export function activeProfile(): LocalProfile | null {
  try { return profiles().find(p => p.id === localStorage.getItem(ACTIVE)) || null; } catch { return null; }
}
export function progressKey() { return `rattil.learn.v1${activeProfile() ? `:${activeProfile()!.id}` : ""}`; }
const hex = (array: Uint8Array) => Array.from(array, b => b.toString(16).padStart(2, "0")).join("");
async function derive(password: string, salt: string) {
  if (!crypto.subtle) throw new Error("Local profiles need HTTPS or localhost.");
  const key = await crypto.subtle.importKey("raw", new TextEncoder().encode(password), "PBKDF2", false, ["deriveBits"]);
  return hex(new Uint8Array(await crypto.subtle.deriveBits({ name: "PBKDF2", hash: "SHA-256", salt: new TextEncoder().encode(salt), iterations: 210000 }, key, 256)));
}
export function setActive(id: string | null) {
  if (id) localStorage.setItem(ACTIVE, id); else localStorage.removeItem(ACTIVE);
  window.dispatchEvent(new Event(PROFILE_EVENT));
}
export async function register(name: string, password: string) {
  name = name.trim().normalize("NFC");
  if (name.length < 2 || name.length > 32 || /[\u0000-\u001f]/u.test(name)) throw new Error("Use a name of 2–32 characters.");
  if (password.length < 8 || password.length > 128) throw new Error("Use a demo password of 8–128 characters.");
  if (profiles().some(p => p.name.toLocaleLowerCase() === name.toLocaleLowerCase())) throw new Error("That name already has a local profile. Sign in instead.");
  const salt = hex(crypto.getRandomValues(new Uint8Array(16)));
  const profile = { id: crypto.randomUUID(), name, salt, iterations: 210000, passwordHash: await derive(password, salt) };
  // Re-read after the asynchronous hash to avoid dropping another saved profile.
  const rows = profiles();
  if (rows.length >= 20) throw new Error("This demo supports 20 local profiles per browser.");
  if (rows.some(p => p.name.toLocaleLowerCase() === name.toLocaleLowerCase())) throw new Error("Name already exists.");
  localStorage.setItem(KEY, JSON.stringify([...rows, profile]));
  setActive(profile.id);
  return profile;
}
export async function signIn(name: string, password: string) {
  if (password.length > 128) throw new Error("Incorrect name or password.");
  const profile = profiles().find(p => p.name.toLocaleLowerCase() === name.trim().normalize("NFC").toLocaleLowerCase());
  if (!profile || await derive(password, profile.salt) !== profile.passwordHash) throw new Error("Incorrect name or password.");
  setActive(profile.id);
  return profile;
}
export function deleteActive() {
  const profile = activeProfile();
  if (!profile) return;
  localStorage.removeItem(progressKey());
  localStorage.setItem(KEY, JSON.stringify(profiles().filter(p => p.id !== profile.id)));
  setActive(null);
}
