# Save as fix_orphan.py and run it:
import json
from pathlib import Path

DIR = Path("E:/Ai-Reciter/dataset_qaloon_taha")
json_path = DIR / "metadata.json"
jsonl_path = DIR / "metadata.jsonl"

data = json.loads(json_path.read_text(encoding="utf-8"))

# Check if 092010 is missing from data
if not any(d["surah"] == 92 and d["ayah"] == 10 for d in data):
    entry = {
        "filename": "092010.wav",
        "path": str((DIR / "092010.wav").resolve()),
        "surah": 92,
        "ayah": 10,
        "reciter_key": "taha",
        "duration": 3.5,
        "start": 0.0,
        "end": 3.5,
        "coverage": 1.0,
        "text_asr_normalized": "فسنيسره للعسرى"
    }
    data.append(entry)
    data.sort(key=lambda x: (x["surah"], x["ayah"]))
    
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("✅ Successfully re-indexed 092010.wav into metadata!")