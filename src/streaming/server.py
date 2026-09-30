"""Local PCM/WebSocket inference service for the trained full Whisper base model."""
import asyncio
from contextlib import asynccontextmanager, suppress
import json
import logging
import os
from pathlib import Path
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from .matcher import RecitationTracker

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.environ.get("RECITER_MODEL_PATH", ROOT / "runs" / "gpu_base_full"))
ASSETS = ROOT / "web" / "public" / "quran"
ORIGINS = set(os.environ.get("RECITER_ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(","))
SAMPLE_RATE = 16000
WINDOW = SAMPLE_RATE * 28
MAX_SESSIONS = 2
logger = logging.getLogger("reciter")


class Engine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = WhisperProcessor.from_pretrained(str(MODEL_PATH), local_files_only=True)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = WhisperForConditionalGeneration.from_pretrained(
            str(MODEL_PATH), local_files_only=True, torch_dtype=dtype).to(self.device).eval()
        self.lock = asyncio.Lock()

    def transcribe(self, audio):
        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", return_attention_mask=True)
        features = inputs.input_features.to(self.device, dtype=self.model.dtype)
        with torch.inference_mode():
            # Blind decoding avoids leaking the expected answer into scoring.
            ids = self.model.generate(input_features=features,
                                      attention_mask=inputs.attention_mask.to(self.device),
                                      language="arabic", task="transcribe", num_beams=1,
                                      max_new_tokens=180, do_sample=False)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


@asynccontextmanager
async def lifespan(app):
    if not (ASSETS / "manifest.json").exists():
        raise RuntimeError("First run: python src/dataset_collection/cache_quran_pages.py")
    app.state.engine = await asyncio.to_thread(Engine)
    app.state.sessions = 0
    logger.info("Loaded %s on %s", MODEL_PATH, app.state.engine.device)
    yield


app = FastAPI(title="Qaloon Live Recitation", lifespan=lifespan)


@app.get("/health")
def health():
    return {"ready": True, "device": app.state.engine.device, "model": MODEL_PATH.name,
            "sample_rate": SAMPLE_RATE, "sessions": app.state.sessions}


@app.websocket("/ws/recite")
async def recite(ws: WebSocket):
    if ws.headers.get("origin") not in ORIGINS:
        await ws.close(code=1008, reason="Origin not allowed")
        return
    if app.state.sessions >= MAX_SESSIONS:
        await ws.close(code=1013, reason="Inference service is busy")
        return
    await ws.accept()
    app.state.sessions += 1
    worker = None
    try:
        config = await asyncio.wait_for(ws.receive_json(), timeout=15)
        surah = config.get("surah")
        if type(surah) is not int or not 1 <= surah <= 114:
            raise ValueError("Choose a valid surah (1–114)")
        path = ASSETS / "surahs" / f"{surah:03d}.json"
        if not path.exists():
            raise ValueError("This surah has not been cached")
        data = json.loads(path.read_text(encoding="utf-8"))
        tracker = RecitationTracker(data["ayahs"], surah=surah)
        engine = app.state.engine
        await ws.send_json({"type": "ready", "device": engine.device, "model": MODEL_PATH.name,
                            "trained": data["trained"], "current": data["ayahs"][0]["ayah"]})
        buffer = np.empty(0, dtype=np.float32)
        total = 0
        voiced_at = 0
        last_decoded = 0
        final_voice_decoded = 0
        wake = asyncio.Event()
        stopping = False

        async def decode_loop():
            nonlocal buffer, last_decoded, final_voice_decoded
            while not tracker.done:
                try:
                    await asyncio.wait_for(wake.wait(), timeout=0.2)
                except asyncio.TimeoutError:
                    pass
                wake.clear()
                final = stopping or (voiced_at > 0 and total - voiced_at >= SAMPLE_RATE * 0.65)
                if final and voiced_at == final_voice_decoded:
                    if stopping:
                        break
                    continue
                if total == last_decoded or len(buffer) < SAMPLE_RATE * 0.7 or voiced_at == 0:
                    if stopping:
                        break
                    continue
                if not final and total - last_decoded < SAMPLE_RATE * 1.2:
                    continue
                audio, through, voice_through = buffer.copy(), total, voiced_at
                start = time.perf_counter()
                async with engine.lock:
                    inference = asyncio.create_task(asyncio.to_thread(engine.transcribe, audio))
                    try:
                        transcript = await asyncio.shield(inference)
                    except asyncio.CancelledError:
                        # A disconnected client must not release the GPU lock
                        # while its non-cancellable torch thread is still running.
                        with suppress(Exception):
                            await inference
                        raise
                last_decoded = through
                if final:
                    final_voice_decoded = voice_through
                update = tracker.feed(transcript, final=final)
                update["latency_ms"] = round((time.perf_counter() - start) * 1000)
                await ws.send_json(update)
                if update["advanced"]:
                    # Preserve audio received during decoding. Do not erase the
                    # beginning of the next ayah while inference was running.
                    pending = max(0, total - through)
                    buffer = buffer[-pending:].copy() if pending else np.empty(0, dtype=np.float32)
                if stopping:
                    break
            await ws.send_json({**tracker.snapshot(tracker.last_partial), "type": "finished"})

        worker = asyncio.create_task(decode_loop())
        started = time.monotonic()
        while not tracker.done:
            message = await asyncio.wait_for(ws.receive(), timeout=60)
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                payload = message["bytes"]
                if len(payload) > 16384 or len(payload) % 4:
                    raise ValueError("Expected small Float32 mono PCM frames")
                audio = np.frombuffer(payload, dtype="<f4").copy()
                if not np.isfinite(audio).all() or np.max(np.abs(audio), initial=0) > 1.01:
                    raise ValueError("Invalid PCM values")
                total += len(audio)
                # Prevent an untrusted client from flooding GPU decoding.
                if total / SAMPLE_RATE > time.monotonic() - started + 5:
                    raise ValueError("Audio must be sent at real-time speed")
                if np.sqrt(np.mean(audio ** 2)) > 0.008:
                    voiced_at = total
                buffer = np.concatenate((buffer, audio))[-WINDOW:]
                wake.set()
            elif message.get("text"):
                control = json.loads(message["text"])
                if control.get("type") == "stop":
                    stopping = True
                    wake.set()
                    await worker
                    break
            if worker.done():
                await worker  # Surface errors from inference.
                break
    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    except ValueError as exc:
        with suppress(Exception):
            await ws.send_json({"type": "error", "message": str(exc)})
    except Exception:
        logger.exception("Recitation session failed")
        with suppress(Exception):
            await ws.send_json({"type": "error", "message": "Inference failed. Check the local backend console."})
    finally:
        if worker is not None:
            worker.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await worker
        app.state.sessions -= 1
        with suppress(Exception):
            await ws.close()
