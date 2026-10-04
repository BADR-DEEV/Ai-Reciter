"""Full Tarteel Whisper-tiny Qaloon adaptation; same audited corpus as base."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_base_full import main, TARTEEL_TINY_MODEL


if __name__ == "__main__":
    main(default_model=TARTEEL_TINY_MODEL, default_reciters=["dokali", "huthaify", "husary"],
         default_output="runs/tarteel_tiny_qaloon", default_dokali_weight=1.0)
