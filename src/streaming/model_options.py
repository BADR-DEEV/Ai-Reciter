"""Local server model selection, without downloading or loading GPU weights."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_PRESETS = {
    "deepdml": ROOT / "runs/deepdml_qaloon_lora_base_v1/adapter",
    "gpu-full-base": ROOT / "runs/gpu_base_full",
    # Full fine-tune pulled by src/deployment/pull_hf_assets.py (Mathani-Ayat/rattil-qaloon-v3).
    "rattil-v3": ROOT / "runs/rattil_qaloon_v3",
}


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
