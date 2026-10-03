"""Local PCM/WebSocket inference service for the trained full Whisper base model."""
import asyncio
import base64
from contextlib import asynccontextmanager, suppress
import json
import logging
import os
from pathlib import Path
import time

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel, Field
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from .matcher import RecitationTracker
from .buffer import after_advance
from . import practice as lessons

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.environ.get("RECITER_MODEL_PATH", ROOT / "runs" / "gpu_base_full"))
# A missing local directory is treated as a Hugging Face model id (development only).
LOCAL_MODEL = MODEL_PATH.is_dir()
ASSETS = ROOT / "web" / "public" / "quran"
ORIGINS = set(os.environ.get("RECITER_ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(","))
SAMPLE_RATE = 16000
WINDOW = SAMPLE_RATE * 28
MAX_SESSIONS = 2
logger = logging.getLogger("reciter")


class Engine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = WhisperProcessor.from_pretrained(str(MODEL_PATH), local_files_only=LOCAL_MODEL)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = WhisperForConditionalGeneration.from_pretrained(
            str(MODEL_PATH), local_files_only=LOCAL_MODEL, torch_dtype=dtype).to(self.device).eval()
        self.lock = asyncio.Lock()
        self.model.config.use_cache = True
        self.last_latency_ms = None

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

    def check(self, audio, candidates):
        """Blind transcript plus closed-set candidate likelihoods for one short clip."""
        transcript = self.transcribe(audio)
        if len(candidates) < 2:
            return transcript, []
        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        features = inputs.input_features.to(self.device, dtype=self.model.dtype)
        return transcript, lessons.candidate_logprobs(self.model, self.processor.tokenizer, features, candidates)


@asynccontextmanager
async def lifespan(app):
    if not (ASSETS / "manifest.json").exists():
        # Lessons work without the Mushaf cache; the studio needs it.
        logger.warning("Quran assets missing. For the studio run: python src/dataset_collection/cache_quran_pages.py")
    app.state.engine = await asyncio.to_thread(Engine)
    app.state.sessions = 0
    logger.info("Loaded %s on %s", MODEL_PATH, app.state.engine.device)
    yield


app = FastAPI(title="Qaloon Live Recitation", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=sorted(ORIGINS), allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type"])
MAX_PRACTICE_SECONDS = 15


class PracticeRequest(BaseModel):
    mode: str = Field(pattern="^(sound|reading)$")
    audio: str = Field(max_length=SAMPLE_RATE * MAX_PRACTICE_SECONDS * 2 * 4 // 3 + 8)  # base64 PCM16
    target: str = Field(min_length=1, max_length=400)
    alternatives: list[str] = Field(default_factory=list, max_length=6)


@app.post("/api/practice")
async def practice(body: PracticeRequest, request: Request):
    """Score one short learner recording for a lesson exercise."""
    origin = request.headers.get("origin")
    if origin is not None and origin not in ORIGINS:
        raise HTTPException(403, "Origin not allowed")
    try:
        raw = base64.b64decode(body.audio, validate=True)
    except ValueError:
        raise HTTPException(400, "Audio must be base64 PCM16")
    if len(raw) % 2 or len(raw) < SAMPLE_RATE // 5:
        raise HTTPException(400, "Recording is too short")
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
    if lessons.is_silent(audio):
        return {"verdict": "silent"}
    candidates = [lessons.plain(body.target)]
    if body.mode == "sound" and not 0 < len(candidates[0]) <= 60:
        raise HTTPException(400, "Sound checks need a short target")
    for text in map(lessons.plain, body.alternatives):
        if text and len(text) <= 60 and text not in candidates:
            candidates.append(text)
    engine = app.state.engine
    async with engine.lock:
        transcript, logprobs = await asyncio.to_thread(
            engine.check, audio, candidates if body.mode == "sound" else [])
    if body.mode == "reading" or len(candidates) < 2:
        return lessons.assess_reading(body.target, transcript)
    return lessons.assess_sound(candidates, 0, logprobs, transcript)


@app.get("/health")
def health():
    return {"ready": True, "device": app.state.engine.device, "model": MODEL_PATH.name,
             "sample_rate": SAMPLE_RATE, "sessions": app.state.sessions,
             "busy": app.state.engine.lock.locked(), "max_sessions": MAX_SESSIONS,
             "last_latency_ms": app.state.engine.last_latency_ms}


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
        start_ayah = config.get("start_ayah", data["ayahs"][0]["ayah"])
        if type(start_ayah) is not int or not any(a["ayah"] == start_ayah for a in data["ayahs"]):
            raise ValueError("Choose a valid starting ayah")
        tracker.index = next(i for i, a in enumerate(data["ayahs"]) if a["ayah"] == start_ayah)
        engine = app.state.engine
        await ws.send_json({"type": "ready", "device": engine.device, "model": MODEL_PATH.name,
                             "trained": data["trained"], "current": start_ayah})
        buffer = np.empty(0, dtype=np.float32)
        total = 0
        voiced_at = 0
        last_decoded = 0
        final_voice_decoded = 0
        wake = asyncio.Event()
        stopping = False
        transition_pending = False

        async def decode_loop():
            nonlocal buffer, last_decoded, final_voice_decoded, transition_pending
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
                update = tracker.feed(transcript, final=final, continuous=True)
                update["latency_ms"] = round((time.perf_counter() - start) * 1000)
                engine.last_latency_ms = update["latency_ms"]
                await ws.send_json(update)
                if update["advanced"] or transition_pending:
                    # Preserve audio received during decoding. Do not erase the
                    # beginning of the next ayah while inference was running.
                    before = buffer
                    buffer = after_advance(buffer, total, through, update, final=final)
                    if buffer is not before:
                        tracker.clear_context()
                    transition_pending = not final
                if stopping:
                    break
            await ws.send_json({**tracker.snapshot(tracker.last_partial), "type": "finished"})

        worker = asyncio.create_task(decode_loop())
        started = time.monotonic()
        while not tracker.done:
            # Surface completion/errors immediately even if no new PCM arrives.
            receive = asyncio.create_task(ws.receive())
            try:
                completed, _ = await asyncio.wait({receive, worker}, timeout=60,
                                                   return_when=asyncio.FIRST_COMPLETED)
                if worker in completed:
                    await worker
                    break
                if receive not in completed:
                    raise asyncio.TimeoutError
                message = receive.result()
            finally:
                if not receive.done():
                    receive.cancel()
                    with suppress(asyncio.CancelledError):
                        await receive
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
                if control.get("type") == "ping":
                    await ws.send_json({"type": "pong", "busy": engine.lock.locked(),
                                        "audio_seconds": round(total / SAMPLE_RATE, 1),
                                        "decoded_seconds": round(last_decoded / SAMPLE_RATE, 1)})
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
