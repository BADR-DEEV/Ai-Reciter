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

from .matcher import RecitationTracker, words
from .live_buffer import AudioQueue, TranscriptOverlap
from . import practice as lessons
from src.training_with_gpu.decoding_safety import load_private_adapter, generation_diagnostics
from .model_options import resolve_local_model

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.environ.get("RECITER_MODEL_PATH", ROOT / "runs" / "deepdml_qaloon_lora_base_v1"))
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
        if MODEL_PATH.is_absolute() and not MODEL_PATH.is_dir():
            raise RuntimeError(f"Local full model is missing: {MODEL_PATH}. Set RECITER_MODEL_PATH to your verified model directory, or restore the pinned private release with python src/deployment/restore_local_full.py. No automatic model substitution is performed.")
        self.model_path = resolve_local_model(MODEL_PATH) if LOCAL_MODEL else MODEL_PATH
        requested_device = os.environ.get("RECITER_DEVICE", "auto")
        if requested_device not in {"auto", "cuda", "cpu"}:
            raise ValueError("RECITER_DEVICE must be auto, cuda or cpu")
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA explicitly requested but unavailable")
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if requested_device == "auto" else requested_device
        precision = os.environ.get("RECITER_DTYPE", "auto")
        if precision == "auto":
            precision = "fp16" if self.device == "cuda" else "fp32"
        if precision not in {"fp16", "bf16", "fp32"} or self.device == "cpu" and precision != "fp32":
            raise ValueError("Use fp16/bf16/fp32 on CUDA, or fp32 on CPU")
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 requested but unsupported by this GPU")
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
        self.precision = precision
        self.model_name = os.environ.get("RECITER_MODEL_PRESET") or (self.model_path.parent.name if self.model_path.name == "adapter" else self.model_path.name)
        self.experimental_adapter = (self.model_path / "adapter_config.json").is_file()
        if self.experimental_adapter:
            if os.environ.get("RECITER_ALLOW_EXPERIMENTAL_ADAPTER") != "1":
                raise RuntimeError("Experimental adapters require explicit private-staging opt-in: RECITER_ALLOW_EXPERIMENTAL_ADAPTER=1. This is not production or learner-grading approval.")
            self.model, self.processor = load_private_adapter(self.model_path, self.device, dtype=dtype)
            logger.warning("Private experimental adapter enabled; qualified learner/riwayah validation is not established")
            self.num_beams = self.model.generation_config.num_beams
        else:
            self.processor = WhisperProcessor.from_pretrained(str(self.model_path), local_files_only=LOCAL_MODEL)
            self.model = WhisperForConditionalGeneration.from_pretrained(
                str(self.model_path), local_files_only=LOCAL_MODEL, torch_dtype=dtype).to(self.device).eval()
            self.num_beams = 1
        if os.environ.get("RECITER_NUM_BEAMS") is not None:
            self.num_beams = int(os.environ["RECITER_NUM_BEAMS"])
            if self.num_beams not in {1, 3, 5}:
                raise ValueError("RECITER_NUM_BEAMS must be 1, 3 or 5")
        self.processor.tokenizer.set_prefix_tokens(language="arabic", task="transcribe")
        self.lock = asyncio.Lock()
        self.model.config.use_cache = True
        self.last_latency_ms = None
        self.last_diagnostics = None

    def transcribe(self, audio):
        return self._transcribe(audio, self.num_beams)

    def transcribe_live(self, audio, final=False):
        # Frequent provisional hypotheses use greedy; pauses/recovery/finish use
        # the selected quality decoder. No expected-text prompt in either mode.
        beams = self.num_beams if final else 1
        budget = None if final else min(self.model.config.max_target_positions, max(64, int(len(audio) / SAMPLE_RATE * 25) + 32))
        return self._transcribe(audio, beams, budget, early_stop=True)

    def _transcribe(self, audio, beams, budget=None, early_stop=False):
        if audio.ndim != 1 or not np.isfinite(audio).all() or len(audio) > SAMPLE_RATE * 30:
            raise ValueError("Expected finite <=30s mono PCM; never silently truncate")
        if not len(audio) or float(np.sqrt(np.mean(audio ** 2))) < 1e-5:
            self.last_diagnostics = {"scorable": False, "decode_flags": ["near_silent_audio"]}
            return ""
        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", return_attention_mask=True)
        features = inputs.input_features.to(self.device, dtype=self.model.dtype)
        from .loop_stop import LoopStop
        from transformers import StoppingCriteriaList
        stopping = StoppingCriteriaList([LoopStop(self.processor.tokenizer)] if early_stop else [])
        with torch.inference_mode():
            # Blind decoding avoids leaking the expected answer into scoring.
            ids = self.model.generate(input_features=features,
                                      attention_mask=inputs.attention_mask.to(self.device),
                                       language="arabic", task="transcribe", num_beams=beams,
                                       max_length=budget or self.model.config.max_target_positions, do_sample=False,
                                       return_dict_in_generate=True, use_cache=True,
                                       stopping_criteria=stopping).sequences
        self.last_diagnostics = generation_diagnostics(ids[0].tolist(), self.processor.tokenizer, budget or self.model.config.max_target_positions)
        if not self.last_diagnostics["scorable"]:
            logger.warning("ASR abstained on decode flags: %s", self.last_diagnostics["decode_flags"])
            return ""  # Empty evidence cannot turn a model loop into learner penalties.
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def check(self, audio, candidates):
        """Blind transcript plus closed-set candidate likelihoods for one short clip."""
        transcript = self.transcribe(audio)
        if not transcript or len(candidates) < 2:
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
        diagnostics = getattr(engine, "last_diagnostics", None)
    if not transcript or diagnostics is not None and not diagnostics["scorable"]:
        # A decoder failure is NOT an incorrect learner attempt. Non-2xx also
        # prevents the existing client from recording a zero adaptive score/XP.
        raise HTTPException(503, "The model could not reliably score this recording. No mistake or score was recorded; please try again. / تعذّر تقييم التسجيل بثقة؛ لم يُسجّل خطأ أو نتيجة، حاول مجددًا.")
    if body.mode == "reading" or len(candidates) < 2:
        return lessons.assess_reading(body.target, transcript)
    if len(logprobs) != len(candidates):
        raise HTTPException(503, "The model could not reliably compare these sounds. Please try again.")
    return lessons.assess_sound(candidates, 0, logprobs, transcript)


@app.get("/health")
def health():
    return {"ready": True, "device": app.state.engine.device, "model": getattr(app.state.engine, "model_name", MODEL_PATH.name),
             "dtype": getattr(app.state.engine, "precision", None), "beams": getattr(app.state.engine, "num_beams", None),
             "experimental_adapter": getattr(app.state.engine, "experimental_adapter", False),
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
        input_mode = config.get("input_mode", "microphone")
        if input_mode not in {"microphone", "file"}:
            raise ValueError("Input mode must be microphone or file")
        tracker = RecitationTracker(data["ayahs"], surah=surah, context_limit=600)
        start_ayah = config.get("start_ayah", data["ayahs"][0]["ayah"])
        if type(start_ayah) is not int or not any(a["ayah"] == start_ayah for a in data["ayahs"]):
            raise ValueError("Choose a valid starting ayah")
        tracker.index = next(i for i, a in enumerate(data["ayahs"]) if a["ayah"] == start_ayah)
        engine = app.state.engine
        await ws.send_json({"type": "ready", "device": engine.device, "model": getattr(engine, "model_name", MODEL_PATH.name),
                             "trained": data["trained"], "current": start_ayah,
                             "upload_credit_seconds": 12, "window_seconds": 8})
        queue = AudioQueue()
        overlap = TranscriptOverlap()
        total = 0
        voiced_at = 0
        last_decoded = 0
        final_voice_decoded = 0
        wake = asyncio.Event()
        stopping = False
        origin = 0
        recovery = False
        previous_stitched = []

        async def decode_loop():
            nonlocal origin, last_decoded, final_voice_decoded, recovery, previous_stitched
            while not tracker.done:
                try:
                    await asyncio.wait_for(wake.wait(), timeout=0.2)
                except asyncio.TimeoutError:
                    pass
                wake.clear()
                final = stopping or (voiced_at > 0 and total - voiced_at >= SAMPLE_RATE * 0.45)
                if voiced_at == 0 or final and voiced_at == final_voice_decoded:
                    # Credit quiet PCM too: a long silent upload must not deadlock
                    # the bounded sender or trigger repeated model hallucinations.
                    if total > last_decoded and (input_mode == "file" or total - last_decoded >= SAMPLE_RATE * .65 or stopping):
                        last_decoded = total
                        origin = max(queue.start, total - SAMPLE_RATE // 2)
                        queue.discard_before(origin)
                        overlap.commit_boundary(origin)
                        await ws.send_json({**tracker.snapshot(), "audio_seconds": total / SAMPLE_RATE,
                            "decoded_seconds": total / SAMPLE_RATE, "pending_seconds": 0,
                            "upload_credit_seconds": 12, "window_seconds": 0,
                            "tracking_uncertain": tracker.uncertain_audio})
                    if stopping:
                        break
                    continue
                if (total == last_decoded and not final) or (total - origin < SAMPLE_RATE * 0.5 and not stopping):
                    if stopping:
                        break
                    continue
                if not recovery and not final and total - last_decoded < SAMPLE_RATE * 0.65:
                    continue
                start = time.perf_counter()
                async with engine.lock:
                    # Snapshot AFTER acquiring GPU access: don't decode a stale
                    # copy queued behind another session's inference.
                    window = SAMPLE_RATE * (24 if recovery else 8)
                    through = min(total, origin + window)
                    audio = queue.read(origin, through)
                    voice_through = min(voiced_at, through)
                    boundary = (final and through == total)
                    if lessons.is_silent(audio):
                        last_decoded = max(last_decoded, through)
                        origin = max(queue.start, through - SAMPLE_RATE // 2)
                        queue.discard_before(origin)
                        overlap.commit_boundary(origin)
                        if boundary:
                            final_voice_decoded = voice_through
                        await ws.send_json({**tracker.snapshot(), "audio_seconds": total / SAMPLE_RATE,
                            "decoded_seconds": last_decoded / SAMPLE_RATE,
                            "pending_seconds": max(0, total - last_decoded) / SAMPLE_RATE,
                            "upload_credit_seconds": 12, "window_seconds": len(audio) / SAMPLE_RATE,
                            "tracking_uncertain": tracker.uncertain_audio})
                        if stopping and boundary:
                            break
                        continue
                    decoder = getattr(engine, "transcribe_live", None)
                    inference = asyncio.create_task(asyncio.to_thread(decoder, audio, boundary or recovery) if decoder
                                                    else asyncio.to_thread(engine.transcribe, audio))
                    try:
                        transcript = await asyncio.shield(inference)
                        diagnostics = getattr(engine, "last_diagnostics", None)
                    except asyncio.CancelledError:
                        # A disconnected client must not release the GPU lock
                        # while its non-cancellable torch thread is still running.
                        with suppress(Exception):
                            await inference
                        raise
                silent_overlap = False
                if overlap.start is not None and origin > overlap.start:
                    common_end = min(last_decoded, through)
                    common = queue.read(origin, common_end) if common_end > origin else np.empty(0)
                    silent_overlap = bool(len(common)) and lessons.is_silent(common)
                stitched = overlap.accept(origin, transcript, silent_overlap=silent_overlap) if transcript else None
                if recovery and through <= last_decoded:
                    stitched = None  # A bounded retry must never cycle without audio progress.
                if stitched is None and not recovery and overlap.start is not None:
                    origin = overlap.start
                    recovery = True
                    wake.set()
                    continue
                uncertain = stitched is None
                if uncertain:
                    # Don't turn an ambiguous window join or model loop into
                    # skipped-ayah verdicts. Expose the gap and resume fresh.
                    overlap.reset()
                    tracker.clear_context()
                    tracker.uncertain_audio = True
                heard = words(stitched) if stitched is not None else []
                stable_prefix = 0
                for old, new in zip(previous_stitched, heard):
                    if old != new:
                        break
                    stable_prefix += 1
                previous_stitched = heard
                last_decoded = max(last_decoded, through)
                if boundary:
                    final_voice_decoded = voice_through
                update = tracker.feed(stitched, final=boundary, continuous=True, stable_prefix=stable_prefix) if stitched is not None else tracker.snapshot()
                update["tracking_uncertain"] = uncertain or tracker.uncertain_audio
                if diagnostics is not None:
                    update["asr_scorable"] = diagnostics["scorable"]
                    update["decode_flags"] = diagnostics["decode_flags"]
                update["latency_ms"] = round((time.perf_counter() - start) * 1000)
                update["audio_seconds"] = round(total / SAMPLE_RATE, 3)
                update["decoded_seconds"] = round(last_decoded / SAMPLE_RATE, 3)
                update["pending_seconds"] = round(max(0, total - last_decoded) / SAMPLE_RATE, 3)
                update["window_seconds"] = round(len(audio) / SAMPLE_RATE, 3)
                update["upload_credit_seconds"] = 12
                engine.last_latency_ms = update["latency_ms"]
                await ws.send_json(update)
                was_recovery = recovery
                recovery = False
                if boundary and not stopping and total - voiced_at >= SAMPLE_RATE * .45:
                    # A confirmed quiet tail permits an acoustic boundary cut;
                    # carry the text prefix so internal waqf doesn't erase half
                    # an ayah. PCM arriving during inference remains queued.
                    origin = max(queue.start, through - int(SAMPLE_RATE * .45))
                    queue.discard_before(origin)
                    overlap.commit_boundary(origin)
                elif through - origin >= window or uncertain or was_recovery:
                    origin = max(origin, through - SAMPLE_RATE * 4)
                    # Keep the last accepted origin available for join recovery.
                    retain = min(origin, overlap.start) if overlap.start is not None else origin
                    queue.discard_before(max(queue.start, retain))
                if stopping and boundary and through == total:
                    break
            await ws.send_json({**tracker.snapshot(tracker.last_partial), "type": "finished",
                "tracking_uncertain": tracker.uncertain_audio, "audio_seconds": total / SAMPLE_RATE,
                "decoded_seconds": last_decoded / SAMPLE_RATE,
                "pending_seconds": max(0, total - last_decoded) / SAMPLE_RATE})

        worker = asyncio.create_task(decode_loop())
        started = time.monotonic()
        while True:
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
                if input_mode == "microphone" and total / SAMPLE_RATE > time.monotonic() - started + 5:
                    raise ValueError("Audio must be sent at real-time speed")
                if input_mode == "file" and total > SAMPLE_RATE * 600:
                    raise ValueError("Uploaded recordings are limited to 10 minutes")
                if len(audio) and lessons.is_silent(audio) is False:
                    voiced_at = total
                queue.append(audio)
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
