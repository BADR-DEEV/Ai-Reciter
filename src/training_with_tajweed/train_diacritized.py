"""V2 stage 1: fine-tune a diacritized ASR adapter; not a tajweed judge."""

import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training_with_gpu"))
from train import main


if __name__ == "__main__":
    if "--label-field" in sys.argv:
        raise SystemExit("Label field is fixed to normalized_with_harakat for this stage")
    sys.argv.extend(["--label-field", "normalized_with_harakat"])
    main()
