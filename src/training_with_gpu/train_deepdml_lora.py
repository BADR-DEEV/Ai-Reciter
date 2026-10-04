"""Five-reader DeepDML LoRA recipe for 12GB CUDA; no implicit model substitution."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.training_with_gpu.train_base_full import DEEPDML_BASE, DEEPDML_SMALL
from src.training_with_gpu.train_tarteel_lora import main as trainer_main

RECIPE = {"init_model": DEEPDML_BASE, "rank": 32, "alpha": 64,
    "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    "label_field": "text_asr_normalized", "repair_normalization": True,
    "noise_prob": .12, "speed_prob": .2, "tempo_min": .95, "tempo_max": 1.05,
    "batch_size": 2, "eval_batch_size": 2, "gradient_accumulation": 8,
    "precision": "bf16", "skip_heldout_eval": True, "require_training_audit": True,
    "supplement": [[str(ROOT / "src/dataset_collection" / f"dataset_qaloon_{r}"),
                    str(ROOT / "src/dataset_collection" / f"dataset_qaloon_{r}" / "experiment_decision.json")]
                   for r in ("trabulsi", "taha")]}


def main(argv=None):
    trainer_main(argv, recipe=RECIPE, allowed_models=(DEEPDML_BASE, DEEPDML_SMALL))


if __name__ == "__main__":
    main()
