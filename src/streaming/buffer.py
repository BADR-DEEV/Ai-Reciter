"""Small, pure audio-buffer policy; no model or CUDA dependency."""
import numpy as np


def after_advance(buffer, total, through, update, overlap=0, final=True):
    """Keep utterance context / next-ayah prefix; otherwise retain pending PCM.

    Audio arriving while GPU decoding is running must never be discarded.
    A window can complete ayah A and contain most of B. Clearing that window
    loses B's beginning and previously caused stalled/incorrect progression.
    """
    # An early terminal hypothesis can advance before the reader finishes the
    # sound. Decoding its tiny cropped tail encourages Whisper hallucinations.
    # Retain the full context until a quiet boundary (or an explicit stop).
    if not final or update.get("tentative_prefix"):
        return buffer
    current = update.get("current")
    result = update.get("results", {}).get(current, {})
    if any(word["status"] == "correct" for word in result.get("words", [])):
        return buffer
    retained = max(0, total - through) + overlap
    return buffer[-retained:].copy() if retained else np.empty(0, dtype=np.float32)
