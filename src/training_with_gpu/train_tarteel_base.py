"""Fully adapt the pinned Tarteel Whisper-base checkpoint to Qaloon labels.

Default corpus: Dokali + Huthaify + Husary. No Hafs labels/audio are imported.
Trabulsi is opt-in only through a reviewed manifest AND a training rights grant.
No model is uploaded, deployed, or existing experiment overwritten by this entry.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_base_full import main, TARTEEL_MODEL


if __name__ == "__main__":
    main(default_model=TARTEEL_MODEL, default_reciters=["dokali", "huthaify", "husary"],
         default_output="runs/tarteel_base_qaloon", default_dokali_weight=1.0)
