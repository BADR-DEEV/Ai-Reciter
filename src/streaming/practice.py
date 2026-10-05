"""Short-utterance checks for the learn-to-read lessons.

Two complementary signals from the same Whisper model:

* A *blind* transcript (no expected text supplied), used for word reading.
* Teacher-forced log-likelihoods of a small closed set of candidates, e.g.
  ``حا`` vs ``ها``. The target is only ever one option among confusable
  alternatives, so this is a forced-choice discrimination test rather than
  leaking the answer into the decoder. It judges which written form the audio
  is closest to. It is not a tajweed or makhraj certification: the model was
  trained on unvowelled text, so short vowels (a/i/u) cannot be separated.
"""
import math
import re

import numpy as np
import torch

from .matcher import alignment, words

HARAKAT = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
SILENCE_RMS = 0.008


def plain(text):
    """Drop diacritics; the model was trained on unvowelled targets."""
    return re.sub(r"\s+", " ", HARAKAT.sub("", text)).strip()


def is_silent(audio):
    if not len(audio):
        return True
    frames = audio[: len(audio) // 1600 * 1600].reshape(-1, 1600) if len(audio) >= 1600 else audio[None]
    return float(np.sqrt((frames ** 2).mean(axis=1)).max()) < SILENCE_RMS


def candidate_logprobs(model, tokenizer, features, candidates):
    """Summed log-probability of each candidate transcript given the audio."""
    tokenizer.set_prefix_tokens(language="arabic", task="transcribe", predict_timestamps=False)
    prefix = len(tokenizer.prefix_tokens)
    sequences = [tokenizer(text).input_ids for text in candidates]
    width = max(map(len, sequences))
    pad = tokenizer.eos_token_id
    ids = torch.tensor([s + [pad] * (width - len(s)) for s in sequences], device=features.device)
    mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in sequences], device=features.device)
    with torch.inference_mode():
        encoded = model.get_encoder()(features).last_hidden_state
        logits = model(encoder_outputs=(encoded.expand(len(candidates), -1, -1),),
                       decoder_input_ids=ids[:, :-1]).logits.float()
    logp = torch.log_softmax(logits, dim=-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    # Only score the text and end-of-text tokens, not the fixed task prefix.
    valid = mask[:, 1:].clone()
    valid[:, : prefix - 1] = 0
    return (logp * valid).sum(dim=1).tolist()


def softmax(values):
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def assess_sound(candidates, target, logprobs, transcript):
    """Forced choice between a target and its confusable alternatives."""
    probabilities = softmax(logprobs)
    best = max(range(len(candidates)), key=probabilities.__getitem__)
    p = probabilities[target]
    verdict = "correct" if best == target and p >= 0.6 else "close" if best == target else "other"
    # If the blind transcript spells exactly another candidate, the two signals
    # disagree: never call that a clear success.
    spoken = plain(transcript).replace(" ", "")
    if verdict == "correct" and any(spoken == c.replace(" ", "") for i, c in enumerate(candidates) if i != target):
        verdict = "close"
    return {"verdict": verdict, "heard": best, "confidence": round(p, 3),
            "probabilities": [round(x, 3) for x in probabilities], "transcript": transcript}


def assess_reading(target_text, transcript, heard_tags=None):
    """Exact ordered word agreement; not certified learner pronunciation.

    `heard_tags` (tajweed model) lists each heard word's tags; matched words carry them.
    """
    expected, heard = words(target_text), words(transcript)
    matched = {i: j for i, j in alignment(expected, heard)}
    statuses = [{"index": i, "text": w, "status": "correct" if i in matched else "missed",
                 "heard": heard[matched[i]] if i in matched else None} for i, w in enumerate(expected)]
    if heard_tags is not None and len(heard_tags) == len(heard):
        for status in statuses:
            if status["index"] in matched and heard_tags[matched[status["index"]]]:
                status["tags"] = list(heard_tags[matched[status["index"]]])
    score = len(matched) / len(expected) if expected else 0.0
    verdict = "correct" if score == 1 else "close" if score >= 0.5 else "other"
    return {"verdict": verdict, "score": round(score, 3), "words": statuses, "transcript": transcript}
