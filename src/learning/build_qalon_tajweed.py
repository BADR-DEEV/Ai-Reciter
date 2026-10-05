"""Build the reader's Qālūn tajweed data (kept for existing commands).

Same as `python -m src.tajweed.build`; the rule engine lives in src/tajweed.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tajweed.build import main  # noqa: E402

if __name__ == "__main__":
    main()
