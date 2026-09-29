"""Add normalized_with_harakat to the four existing metadata.jsonl files.

Idempotent; preserves all original keys and writes each JSONL atomically.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "dataset_collection"
sys.path.insert(0, str(ROOT))
from qaloon_audio2text import normalize_with_harakat


def main():
    for folder in ("dataset_qaloon_hutafi", "dataset_qaloon_Husary",
                   "dataset_qaloon_dokali", "dataset_qaloon_waleed"):
        path = ROOT / folder / "metadata.jsonl"
        rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        for row in rows:
            row["normalized_with_harakat"] = normalize_with_harakat(row["text_raw_uthmani"])
        temporary = path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(path)
        print(f"{folder}: {len(rows)} labelled rows")


if __name__ == "__main__":
    main()
