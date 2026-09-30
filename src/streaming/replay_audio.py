"""Replay mono 16 kHz WAVs at microphone speed against the live GPU backend.

python -m src.streaming.replay_audio --surah 1 --audio fatiha.wav
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path
import wave

import numpy as np
import websockets


def load_audio(paths):
    parts = []
    for path in paths:
        with wave.open(str(path), "rb") as wav:
            if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth()) != (1, 16000, 2):
                raise ValueError(f"Expected mono 16 kHz PCM16 WAV: {path}")
            parts.append(np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768)
        parts.append(np.zeros(16000, dtype=np.float32))
    return np.concatenate(parts)


async def main(args):
    audio = load_audio(args.audio)
    async with websockets.connect(args.url, origin="http://127.0.0.1:3000", max_size=1_000_000) as ws:
        await ws.send(json.dumps({"surah": args.surah}))
        ready = json.loads(await ws.recv())
        print(json.dumps(ready), flush=True)
        if ready["type"] != "ready":
            raise RuntimeError(ready)

        async def receive():
            async for message in ws:
                update = json.loads(message)
                print(json.dumps(update, ensure_ascii=False), flush=True)
                if update["type"] == "error":
                    raise RuntimeError(update["message"])
                if update["type"] == "finished":
                    return update

        receiver = asyncio.create_task(receive())
        loop = asyncio.get_running_loop()
        started = loop.time()
        for offset in range(0, len(audio), 2048):
            if receiver.done():
                break
            await ws.send(audio[offset:offset + 2048].astype("<f4").tobytes())
            await asyncio.sleep(max(0, started + (offset + 2048) / 16000 - loop.time()))
        if not receiver.done():
            await ws.send(json.dumps({"type": "stop"}))
        await asyncio.wait_for(receiver, timeout=30)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surah", type=int, required=True)
    parser.add_argument("--audio", type=Path, nargs="+", required=True)
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws/recite")
    asyncio.run(main(parser.parse_args()))
