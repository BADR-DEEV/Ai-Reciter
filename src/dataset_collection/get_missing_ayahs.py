import os
import json

from dataset_collection.qaloon_audio2text import AUDIO_DIR, MISSING, download_audio

with open(MISSING, encoding="utf-8") as f:
    missing = json.load(f)

remaining = []

for item in missing:
    s = item["surah"]
    a = item["ayah"]

    dest = os.path.join(AUDIO_DIR, f"{s:03d}{a:03d}.mp3")

    reciter = download_audio(s, a, dest)

    if reciter:
        print(f"✅ recovered {s}:{a} from {reciter}")
    else:
        print(f"❌ still unavailable {s}:{a}")
        remaining.append(item)

with open(MISSING, "w", encoding="utf-8") as f:
    json.dump(remaining, f, ensure_ascii=False, indent=2)

print(f"{len(remaining)} still missing")