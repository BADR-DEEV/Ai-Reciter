"""Blind decoding diagnostics: never trim or silently repair a Quran transcript.

Flags mean model uncertainty, not learner recitation mistakes. N-gram suppression
is a development-only ablation since it can suppress authentic repeats/restarts.
"""
from pathlib import Path
import json

DECODE_PROFILES = {
    "greedy": {"num_beams": 1, "repetition_penalty": 1.0, "no_repeat_ngram_size": 0},
    "beam3": {"num_beams": 3, "repetition_penalty": 1.0, "no_repeat_ngram_size": 0},
    "beam3-soft": {"num_beams": 3, "repetition_penalty": 1.1, "no_repeat_ngram_size": 0},
    "beam5-block3": {"num_beams": 5, "repetition_penalty": 1.2, "no_repeat_ngram_size": 3},
}


def repeated_cycle(ids, cycles=8, max_period=4):
    """Detect sustained exact token cycles; real repeated speech may also flag."""
    for period in range(1, max_period + 1):
        width = period * cycles
        for end in range(width, len(ids) + 1):
            segment = ids[end - width:end]
            if segment == segment[:period] * cycles:
                return True
    return False


def generation_diagnostics(sequence, tokenizer, max_length=448):
    ids = [int(t) for t in sequence]
    prefix = list(tokenizer.prefix_tokens)
    if ids[:len(prefix)] != prefix:
        return {"eos_emitted": False, "generated_tokens": len(ids), "decode_flags": ["unexpected_decoder_prefix"], "scorable": False}
    tail = ids[len(prefix):]
    eos = tokenizer.eos_token_id
    emitted = eos in tail
    content = tail[:tail.index(eos)] if emitted else tail
    flags = []
    if not emitted:
        flags.append("no_eos_or_length_limit")
    if repeated_cycle(content):
        flags.append("sustained_token_cycle")
    if hasattr(tokenizer, "decode") and repeated_cycle(tokenizer.decode(content, skip_special_tokens=True).split()):
        flags.append("sustained_word_cycle")
    if not content:
        flags.append("empty_decoding")
    return {"eos_emitted": emitted, "generated_tokens": len(content), "decode_flags": flags,
            "hit_context_limit": not emitted and len(ids) >= max_length, "scorable": not flags}


def save_decoding_bundle(model, processor, destination, profile):
    """Persist actual runtime config separately: PEFT doesn't save it for us."""
    destination = Path(destination)
    model.generation_config.save_pretrained(destination)
    (destination / "decoding_policy.json").write_text(json.dumps({"profile": profile,
        "kwargs": DECODE_PROFILES[profile], "max_length": model.config.max_target_positions,
        "expected_text_prompt": False, "fixed_60_token_cap": False,
        "flag_behavior": "abstain-from-learner-scoring; never silently delete repeat tokens",
        "not_calibrated_for_tajweed_or_learner_errors": True}, indent=2) + "\n", encoding="utf-8")


def load_private_adapter(directory, device="cuda", dtype=None):
    """Load pinned weights, matching processor and VERIFIED generation controls."""
    from peft import PeftConfig, PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor, GenerationConfig
    from .train_base_full import configure_generation, MODEL_REVISIONS
    directory = Path(directory)
    adapter = PeftConfig.from_pretrained(directory)
    revision = MODEL_REVISIONS.get(adapter.base_model_name_or_path)
    if not revision or adapter.revision != revision:
        raise ValueError("Adapter must identify the pinned supported base model/revision")
    processor = WhisperProcessor.from_pretrained(directory, trust_remote_code=False)
    processor.tokenizer.set_prefix_tokens(language="arabic", task="transcribe")
    kwargs = {"torch_dtype": dtype} if dtype is not None else {}
    base = WhisperForConditionalGeneration.from_pretrained(adapter.base_model_name_or_path, revision=revision, trust_remote_code=False, **kwargs)
    configure_generation(base, processor)
    if (directory / "generation_config.json").is_file():
        saved = GenerationConfig.from_pretrained(directory)
        # Compatibility template is reconstructible; no unverified replacement IDs.
        for field in ("decoder_start_token_id", "eos_token_id", "pad_token_id", "no_timestamps_token_id", "lang_to_id", "task_to_id", "suppress_tokens", "begin_suppress_tokens"):
            if getattr(saved, field, None) != getattr(base.generation_config, field, None):
                raise ValueError(f"Saved generation control mismatch: {field}")
        if (saved.max_length != base.config.max_target_positions or saved.forced_decoder_ids is not None
                or saved.max_new_tokens is not None or saved.return_timestamps is not False
                or saved.language != "arabic" or saved.task != "transcribe"):
            raise ValueError("Saved generation bounds/forced IDs differ")
        if (directory / "decoding_policy.json").is_file():
            policy = json.loads((directory / "decoding_policy.json").read_text(encoding="utf-8"))
            if (policy.get("profile") not in DECODE_PROFILES or policy.get("kwargs") != DECODE_PROFILES[policy["profile"]]
                    or policy.get("max_length") != saved.max_length or policy.get("expected_text_prompt") is not False):
                raise ValueError("Saved decoding policy differs from supported blind profiles")
            for field, value in policy["kwargs"].items():
                if getattr(saved, field, None) != value:
                    raise ValueError(f"Saved decoder differs from its documented policy: {field}")
        base.generation_config = saved
    model = PeftModel.from_pretrained(base, directory).to(device)
    model.eval()
    return model, processor
