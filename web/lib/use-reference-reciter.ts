"use client";
import { useEffect, useState } from "react";
import { DEFAULT_RECITER, isReciter, type ReciterID } from "./reciters";
import { stopAudio } from "./learn/speech";
const KEY = "rattil.reference-reciter.v1";
const EVENT = "rattil:reciter";
export function useReferenceReciter() {
  const [reciter, setReciter] = useState<ReciterID>(DEFAULT_RECITER);
  useEffect(() => {
    const refresh = () => { try { const id = localStorage.getItem(KEY); setReciter(id && isReciter(id) ? id : DEFAULT_RECITER); } catch { /* Default stays usable. */ } };
    refresh(); window.addEventListener(EVENT, refresh); window.addEventListener("storage", refresh);
    return () => { window.removeEventListener(EVENT, refresh); window.removeEventListener("storage", refresh); };
  }, []);
  const selectReciter = (id: ReciterID) => {
    stopAudio(); setReciter(id);
    try { localStorage.setItem(KEY, id); } catch { /* Still works in this tab. */ }
    window.dispatchEvent(new Event(EVENT));
  };
  return { reciter, selectReciter };
}
