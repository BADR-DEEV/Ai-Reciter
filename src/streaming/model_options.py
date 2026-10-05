"""Local server model selection, without downloading or loading GPU weights."""
from dataclasses import dataclass
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
MODEL_PRESETS = {
    "deepdml": ROOT / "runs/deepdml_qaloon_lora_base_v1/adapter",
    "gpu-full-base": ROOT / "runs/gpu_base_full",
    # Full fine-tune pulled by src/deployment/pull_hf_assets.py (Mathani-Ayat/rattil-qaloon-v3).
    "rattil-v3": ROOT / "runs/rattil_qaloon_v3",
    # Tajweed-tagged fine-tune (words carry <tj:...> tags). Optional: reported unavailable until trained.
    "rattil-tajweed-v1": ROOT / "runs/rattil_qaloon_tajweed_v1",
}
TAJWEED_PRESETS = {"rattil-tajweed-v1"}
LEGACY_DEFAULT = ROOT / "runs/deepdml_qaloon_lora_base_v1"
NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}")


@dataclass(frozen=True)
class ModelSpec:
    """One named server engine: a preset, a directory, or a preset with an overriding directory."""
    name: str
    path: Path
    preset: "str | None" = None

    @property
    def kind(self):
        return "tajweed" if self.name == "tajweed" or self.preset in TAJWEED_PRESETS else "plain"

    @property
    def label(self):
        return self.preset or (self.path.parent.name if self.path.name == "adapter" else self.path.name)

    @property
    def local(self):
        # A relative path that does not exist is a Hugging Face id (legacy RECITER_MODEL_PATH, development only).
        return self.path.is_absolute() or self.path.exists()

    def present(self):
        return self.path.is_dir() or not self.local

    def entry(self):
        return f"{self.name}={self.preset}:{self.path}" if self.preset else f"{self.name}={self.path}"


def model_spec(name, value):
    preset, _, override = value.partition(":")
    if preset not in MODEL_PRESETS:
        return ModelSpec(name, Path(value).expanduser().resolve())
    return ModelSpec(name, Path(override).expanduser().resolve() if override else MODEL_PRESETS[preset], preset)


def parse_models(items):
    """Repeated or comma-separated `PRESET`, `NAME=PRESET`, `NAME=PATH` or `NAME=PRESET:PATH`.

    A bare preset is named `tajweed` for a tajweed preset, else `plain` (then its own name).
    """
    specs = {}
    for item in (part.strip() for entry in items for part in entry.split(",")):
        if not item:
            continue
        name, sep, value = (part.strip() for part in item.partition("="))
        if not sep:
            if item not in MODEL_PRESETS:
                raise ValueError(f"Unknown model preset {item!r}. Presets: {', '.join(MODEL_PRESETS)}; or use NAME=PATH.")
            name, value = "tajweed" if item in TAJWEED_PRESETS else "plain" if "plain" not in specs else item, item
        if not NAME.fullmatch(name) or not value:
            raise ValueError(f"Invalid model entry {item!r}: use NAME=PRESET|PATH with a lowercase NAME.")
        if name in specs:
            raise ValueError(f"Model {name!r} is configured twice.")
        specs[name] = model_spec(name, value)
    return specs


def with_tajweed_slot(specs):
    """Always offer the tajweed engine; it plugs in once its directory exists."""
    if not any(spec.kind == "tajweed" for spec in specs.values()):
        specs = {**specs, "tajweed": model_spec("tajweed", "rattil-tajweed-v1")}
    return specs


def default_name(specs):
    return "plain" if "plain" in specs else next(iter(specs))


def default_models():
    """plain=rattil-v3 when it has been pulled, else None (callers keep their previous default)."""
    return {"plain": model_spec("plain", "rattil-v3")} if MODEL_PRESETS["rattil-v3"].is_dir() else None


def models_from_env(env):
    """Server registry: RECITER_MODELS, else legacy RECITER_MODEL_PATH, else rattil-v3, else the old DeepDML default."""
    if env.get("RECITER_MODELS"):
        specs = parse_models([env["RECITER_MODELS"]])
    elif env.get("RECITER_MODEL_PATH"):
        specs = {"plain": ModelSpec("plain", Path(env["RECITER_MODEL_PATH"]), env.get("RECITER_MODEL_PRESET") or None)}
    else:
        specs = default_models() or {"plain": ModelSpec("plain", LEGACY_DEFAULT)}
    if not specs:
        raise ValueError("RECITER_MODELS lists no models.")
    return with_tajweed_slot(specs)


def resolve_local_model(path):
    path = Path(path).expanduser().resolve()
    if (path / "adapter/adapter_config.json").is_file():
        path = path / "adapter"
    if not path.is_dir():
        raise ValueError(f"Selected local model is missing: {path}. No model fallback or download is performed.")
    return path


def select_model(preset, path=None):
    selected = resolve_local_model(path if path is not None else MODEL_PRESETS[preset])
    adapter = selected / "adapter_config.json"
    if preset == "deepdml":
        if not adapter.is_file() or not (selected / "adapter_model.safetensors").is_file():
            raise ValueError("DeepDML requires the trained adapter directory (or its parent run), not foundation weights.")
        configuration = json.loads(adapter.read_text(encoding="utf-8"))
        if not configuration.get("base_model_name_or_path", "").startswith("deepdml/"):
            raise ValueError("The selected adapter is not a DeepDML adapter.")
    elif adapter.is_file() or not (selected / "config.json").is_file():
        raise ValueError(f"{preset} requires a full saved Whisper model, not a LoRA adapter.")
    return selected


def check_model(spec, required=True):
    """Validated directory. An optional tajweed engine that is missing or incomplete returns None instead."""
    try:
        return select_model(spec.preset, spec.path) if spec.preset else resolve_local_model(spec.path)
    except ValueError:
        if required or spec.kind != "tajweed":
            raise
        return None
