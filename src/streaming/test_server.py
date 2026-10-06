"""WebSocket lifecycle tests with fake ASR, never GPU or simulated model claims."""
import asyncio
import base64
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
import torch
from fastapi.testclient import TestClient
from . import server

TEXT = "قل هو الله احد الله الصمد لم يلد ولم يولد ولم يكن له كفؤا احد"
TAGGED = "قل<qalqala> هو الله احد<qalqala> الله الصمد <qalqala> لم يلد<qalqala> ولم يولد ولم يكن له كفؤا احد<qalqala>"
SURAH = {"id": 112, "trained": True, "ayahs": [
    {"ayah": 1, "text": "قُلْ هُوَ اَ۬للَّهُ أَحَدٌۖ\xa0١", "normalized": "قل هو الله احد"},
    {"ayah": 2, "text": "اِ۬للَّهُ اُ۬لصَّمَدُۖ\xa0٢", "normalized": "الله الصمد"},
    {"ayah": 3, "text": "لَمْ يَلِدْ وَلَمْ يُولَدْۖ\xa0٣", "normalized": "لم يلد ولم يولد"},
    {"ayah": 4, "text": "وَلَمْ يَكُن لَّهُۥ كُفُؤاً أَحَدُۢۖ\xa0٤", "normalized": "ولم يكن له كفؤا احد"}]}
ORIGIN = {"origin": "http://127.0.0.1:3000"}


class FakeEngine:
    def __init__(self, spec=None):
        self.device = "test"
        self.lock = asyncio.Lock()
        self.last_latency_ms = None
        self.kind = spec.kind if spec else "plain"
        self.model_name = spec.label if spec else "fake"

    def transcribe(self, audio):
        return TAGGED if self.kind == "tajweed" else TEXT

    def check(self, audio, candidates):
        return self.transcribe(audio), [-1.0] + [-6.0] * (len(candidates) - 1) if len(candidates) > 1 else []


def pcm16(seconds=1.0, value=3200):
    return base64.b64encode(np.full(int(16000 * seconds), value, dtype="<i2").tobytes()).decode()


class ServerCase(unittest.TestCase):
    tajweed = False  # whether the tajweed model directory exists

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "plain").mkdir()
        if self.tajweed:
            (self.root / "tajweed").mkdir()
        (self.root / "quran/surahs").mkdir(parents=True)
        (self.root / "quran/surahs/112.json").write_text(json.dumps(SURAH, ensure_ascii=False), encoding="utf-8")
        (self.root / "quran/manifest.json").write_text("{}", encoding="utf-8")
        env = {"RECITER_MODELS": f"plain=rattil-v3:{self.root / 'plain'},tajweed=rattil-tajweed-v1:{self.root / 'tajweed'}"}
        for active in (patch.object(server, "Engine", FakeEngine), patch.object(server, "ASSETS", self.root / "quran"),
                       patch.dict(os.environ, env)):
            active.start()
            self.addCleanup(active.stop)
        self.client = TestClient(server.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def ready(self, ws, **config):
        ws.send_json({"surah": 112, **config})
        message = ws.receive_json()
        self.assertEqual(message["type"], "ready", message)
        return message

    def drain(self, ws, seconds=2):
        for _ in range(int(seconds * 4)):
            ws.send_bytes(np.full(4000, .1, dtype="<f4").tobytes())
        ws.send_json({"type": "stop"})
        messages = []
        for _ in range(100):
            messages.append(ws.receive_json())
            self.assertNotEqual(messages[-1]["type"], "error", messages[-1])
            if messages[-1]["type"] == "finished":
                return messages
        self.fail("Session did not finish")


class ServerTests(ServerCase):
    def test_health_reports_capacity(self):
        body = self.client.get("/health").json()
        self.assertEqual(body["max_sessions"], 2)
        self.assertFalse(body["busy"])

    def test_flagged_or_empty_asr_abstains_without_scoring_practice(self):
        audio = base64.b64encode(np.full(16000, 3200, dtype="<i2").tobytes()).decode()
        with patch.object(FakeEngine, "check", return_value=("", [])):
            for mode in ("sound", "reading"):
                response = self.client.post("/api/practice", json={"mode": mode, "target": "حا", "alternatives": ["ها"], "audio": audio})
                self.assertEqual(response.status_code, 503)
                self.assertNotIn("score", response.json())
                self.assertNotIn("verdict", response.json())

    def test_search_finds_the_recited_passage(self):
        body = self.client.post("/api/search", json={"audio": pcm16()}, headers=ORIGIN).json()
        self.assertEqual(body["verdict"], "found")
        self.assertEqual(body["transcript"], TEXT)
        best = body["results"][0]
        self.assertEqual((best["start"], best["end"]), ({"surah": 112, "ayah": 1}, {"surah": 112, "ayah": 4}))
        self.assertEqual((best["confidence"], best["juz"], body["model_used"]), ("high", [30], "plain"))

    def test_search_juz_scope_and_openings(self):
        body = self.client.post("/api/search", json={"audio": pcm16(), "juz": [1, 1]}).json()
        self.assertEqual((body["verdict"], body["results"], body["juz"]), ("none", [], [1]))
        with patch.object(FakeEngine, "transcribe", return_value="بسم الله الرحمن الرحيم"):
            body = self.client.post("/api/search", json={"audio": pcm16()}).json()
        self.assertEqual((body["verdict"], body["opening"]), ("opening_only", ["basmala"]))

    def test_search_rejects_bad_requests_and_skips_silence(self):
        for request, status in (({"audio": pcm16(.5), "juz": [31]}, 400), ({"audio": pcm16(.2)}, 400),
                                ({"audio": pcm16(31)}, 422), ({"audio": "not base64!"}, 400)):
            self.assertEqual(self.client.post("/api/search", json=request).status_code, status, request)
        self.assertEqual(self.client.post("/api/search", json={"audio": pcm16()}, headers={"origin": "http://evil.test"}).status_code, 403)
        with patch.object(FakeEngine, "transcribe", side_effect=AssertionError("Silence must not decode")):
            body = self.client.post("/api/search", json={"audio": pcm16(value=0)}).json()
        self.assertEqual(body["verdict"], "silent")
        with patch.object(FakeEngine, "transcribe", return_value=""):
            self.assertEqual(self.client.post("/api/search", json={"audio": pcm16()}).status_code, 503)

    def test_start_ayah_and_ping(self):
        with self.client.websocket_connect("/ws/recite", headers=ORIGIN) as ws:
            ws.send_json({"surah": 112, "start_ayah": 3})
            self.assertEqual(ws.receive_json()["current"], 3)
            ws.send_json({"type": "ping"})
            self.assertEqual(ws.receive_json()["type"], "pong")
            ws.send_json({"type": "stop"})
            self.assertEqual(ws.receive_json()["type"], "finished")
        self.assertEqual(self.client.get("/health").json()["sessions"], 0)

    def test_invalid_start_ayah_rejected(self):
        with self.client.websocket_connect("/ws/recite", headers=ORIGIN) as ws:
            ws.send_json({"surah": 112, "start_ayah": 999})
            self.assertEqual(ws.receive_json()["type"], "error")

    def test_file_mode_accepts_faster_than_realtime_and_drains_at_eof(self):
        with self.client.websocket_connect("/ws/recite", headers=ORIGIN) as ws:
            ws.send_json({"surah": 112, "input_mode": "file"})
            self.assertEqual(ws.receive_json()["upload_credit_seconds"], 12)
            for _ in range(32):  # 8.192s delivered immediately, not wall-clock replay.
                ws.send_bytes(np.full(4096, .1, dtype="<f4").tobytes())
            ws.send_json({"type": "stop"})
            for _ in range(100):
                message = ws.receive_json()
                self.assertNotEqual(message["type"], "error")
                if message["type"] == "finished":
                    self.assertTrue(message["complete"])
                    break
            else:
                self.fail("EOF failed to drain uploaded audio")

    def test_silent_upload_returns_credit_without_model_inference(self):
        with patch.object(FakeEngine, "transcribe", side_effect=AssertionError("Silence must not decode")):
            with self.client.websocket_connect("/ws/recite", headers=ORIGIN) as ws:
                ws.send_json({"surah": 112, "input_mode": "file"})
                ws.receive_json()
                for _ in range(32):
                    ws.send_bytes(np.zeros(4096, dtype="<f4").tobytes())
                ws.send_json({"type": "stop"})
                credited = 0
                for _ in range(100):
                    message = ws.receive_json()
                    self.assertNotEqual(message["type"], "error")
                    credited = max(credited, message.get("decoded_seconds", 0))
                    if message["type"] == "finished":
                        self.assertFalse(message["complete"])
                        break
                self.assertAlmostEqual(credited, 32 * 4096 / 16000)

    def test_long_upload_wraps_queue_and_drains_without_growing_windows(self):
        seen = []
        def decode(audio):
            seen.append(len(audio))
            return "كلام غير مطابق"
        count = 96 * 16000
        with patch.object(FakeEngine, "transcribe", side_effect=decode):
            with self.client.websocket_connect("/ws/recite", headers=ORIGIN) as ws:
                ws.send_json({"surah": 112, "input_mode": "file"})
                ws.receive_json()
                sent = decoded = 0
                stopped = False
                for _ in range(1000):
                    while sent < count and min(sent + 4096, count) - decoded <= 12 * 16000:
                        end = min(sent + 4096, count)
                        ws.send_bytes(np.full(end - sent, .1, dtype="<f4").tobytes())
                        sent = end
                    if sent == count and not stopped:
                        ws.send_json({"type": "stop"})
                        stopped = True
                    message = ws.receive_json()
                    self.assertNotEqual(message["type"], "error")
                    decoded = max(decoded, round(message.get("decoded_seconds", 0) * 16000))
                    if message["type"] == "finished":
                        self.assertEqual(decoded, count)
                        self.assertFalse(message["complete"])
                        break
                else:
                    self.fail("Long file stalled")
        self.assertTrue(seen)
        self.assertLessEqual(max(seen), 24 * 16000)

    def test_stop_flushes_and_completes_without_extra_pcm(self):
        with self.client.websocket_connect("/ws/recite", headers=ORIGIN) as ws:
            ws.send_json({"surah": 112})
            self.assertEqual(ws.receive_json()["type"], "ready")
            for _ in range(4):
                ws.send_bytes(np.full(3200, 0.1, dtype="<f4").tobytes())
            ws.send_json({"type": "stop"})
            messages = []
            for _ in range(5):
                message = ws.receive_json()
                messages.append(message)
                if message["type"] == "finished":
                    break
            self.assertEqual(messages[-1]["type"], "finished")
            self.assertTrue(messages[-1]["complete"])

    def test_invalid_pcm_surfaces_error(self):
        with self.client.websocket_connect("/ws/recite", headers=ORIGIN) as ws:
            ws.send_json({"surah": 112})
            ws.receive_json()
            ws.send_bytes(np.array([np.nan], dtype="<f4").tobytes())
            self.assertEqual(ws.receive_json()["type"], "error")

    def test_worker_failure_surfaces_without_another_client_message(self):
        with patch.object(FakeEngine, "transcribe", side_effect=RuntimeError("Test inference failure")):
            with self.client.websocket_connect("/ws/recite", headers=ORIGIN) as ws:
                ws.send_json({"surah": 112})
                ws.receive_json()
                for _ in range(6):
                    ws.send_bytes(np.full(3200, 0.1, dtype="<f4").tobytes())
                self.assertEqual(ws.receive_json()["type"], "error")

    def test_health_lists_models_and_keeps_existing_fields(self):
        body = self.client.get("/health").json()
        self.assertEqual((body["status"], body["default_model"], body["model"], body["device"]), ("ok", "plain", "rattil-v3", "test"))
        for field in ("ready", "dtype", "beams", "experimental_adapter", "sample_rate", "sessions", "busy", "max_sessions", "last_latency_ms"):
            self.assertIn(field, body)
        models = {m["name"]: m for m in body["models"]}
        self.assertEqual(models["plain"], {"name": "plain", "label": "rattil-v3", "path": str((self.root / "plain").resolve()), "kind": "plain",
                                           "available": True, "loaded": True, "device": "test"})
        self.assertEqual({k: models["tajweed"][k] for k in ("kind", "available", "loaded", "device")}, {"kind": "tajweed", "available": False, "loaded": False, "device": None})

    def test_practice_defaults_to_plain_and_falls_back_from_missing_tajweed(self):
        body = self.client.post("/api/practice", json={"mode": "reading", "target": "قُلْ هُوَ", "audio": pcm16()}).json()
        self.assertEqual((body["model_used"], body["fallback_reason"], body["tajweed_tags"]), ("plain", None, []))
        body = self.client.post("/api/practice", json={"mode": "reading", "target": "قُلْ هُوَ", "audio": pcm16(), "model": "tajweed"}).json()
        self.assertEqual((body["model_used"], body["fallback_reason"], body["verdict"]), ("plain", "missing", "correct"))
        self.assertNotIn("tajweed_feedback", body)
        silent = self.client.post("/api/practice", json={"mode": "reading", "target": "قل", "audio": pcm16(value=0), "model": "tajweed"}).json()
        self.assertEqual(silent, {"verdict": "silent", "model_used": "plain", "fallback_reason": "missing"})

    def test_unknown_or_unavailable_models_are_rejected(self):
        response = self.client.post("/api/practice", json={"mode": "reading", "target": "قل", "audio": pcm16(), "model": "other"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("plain, tajweed", response.json()["detail"])
        self.assertEqual(self.client.post("/api/practice", json={"mode": "reading", "target": "قل", "audio": pcm16(), "model": "Bad Name"}).status_code, 422)
        with self.client.websocket_connect("/ws/recite", headers=ORIGIN) as ws:
            ws.send_json({"surah": 112, "model": "other"})
            message = ws.receive_json()
            self.assertEqual(message["type"], "error")
            self.assertIn("Unknown model", message["message"])
        with self.client.websocket_connect("/ws/recite", headers=ORIGIN) as ws:
            ws.send_json({"surah": 112, "model": 3})
            self.assertEqual(ws.receive_json()["type"], "error")

    def test_websocket_falls_back_from_missing_tajweed_with_flag(self):
        with self.client.websocket_connect("/ws/recite", headers=ORIGIN) as ws:
            message = self.ready(ws, model="tajweed")
            self.assertEqual((message["model_used"], message["model_kind"], message["fallback_reason"]), ("plain", "plain", "missing"))
            finished = self.drain(ws)[-1]
            self.assertTrue(finished["complete"])
            self.assertFalse(any("tags" in w for r in finished["results"].values() for w in r["words"]))

    def test_tajweed_model_plugs_in_when_its_directory_appears(self):
        (self.root / "tajweed").mkdir()
        self.assertTrue({m["name"]: m for m in self.client.get("/health").json()["models"]}["tajweed"]["available"])
        body = self.client.post("/api/practice", json={"mode": "reading", "target": "قل هو", "audio": pcm16(), "model": "tajweed"}).json()
        self.assertEqual((body["model_used"], body["fallback_reason"]), ("tajweed", None))
        models = {m["name"]: m for m in self.client.get("/health").json()["models"]}
        self.assertTrue(models["tajweed"]["loaded"])
        engines = self.client.app.state.models.engines
        self.assertIs(engines["plain"].lock, engines["tajweed"].lock)  # one decode at a time per device


class TajweedServerTests(ServerCase):
    tajweed = True

    def test_practice_strips_tags_before_matching_and_returns_them(self):
        target = "قُلْ هُوَ اَ۬للَّهُ أَحَدٌ"
        tagged = self.client.post("/api/practice", json={"mode": "reading", "target": target, "audio": pcm16(), "model": "tajweed"}).json()
        plain = self.client.post("/api/practice", json={"mode": "reading", "target": target, "audio": pcm16()}).json()
        self.assertEqual((tagged["model_used"], tagged["fallback_reason"]), ("tajweed", None))
        self.assertEqual((tagged["verdict"], tagged["score"], tagged["transcript"]), (plain["verdict"], plain["score"], plain["transcript"]))
        self.assertEqual(tagged["tajweed_tags"][:2], [{"word_index": 0, "tags": ["qalqala"]}, {"word_index": 3, "tags": ["qalqala"]}])
        self.assertEqual([w.get("tags") for w in tagged["words"]], [["qalqala"], None, None, ["qalqala"]])
        self.assertNotIn("<", tagged["transcript"])
        sound = self.client.post("/api/practice", json={"mode": "sound", "target": "قل", "alternatives": ["كل"], "audio": pcm16(), "model": "tajweed"}).json()
        self.assertEqual((sound["verdict"], sound["model_used"]), ("correct", "tajweed"))

    def test_feedback_hook_is_optional_and_isolated(self):
        calls = []
        def compare(target_text, hyp_words, hyp_tags, surah=None, ayah=None):
            calls.append((target_text, hyp_words, hyp_tags))
            return {"rules": len(hyp_tags)}
        request = {"mode": "reading", "target": "قل هو", "audio": pcm16(), "model": "tajweed"}
        with patch.dict(sys.modules, {"src.tajweed.feedback": SimpleNamespace(compare=compare)}):
            body = self.client.post("/api/practice", json=request).json()
            self.assertEqual(body["tajweed_feedback"], {"rules": 15})
            self.assertEqual(calls[0][0], "قل هو")
            self.assertEqual(calls[0][1][:2], ["قل", "هو"])
            self.assertNotIn("tajweed_feedback", self.client.post("/api/practice", json={**request, "model": "plain"}).json())
        for broken in (SimpleNamespace(compare=lambda *a: None), SimpleNamespace(compare=lambda *a: 1 / 0), None):
            with patch.dict(sys.modules, {"src.tajweed.feedback": broken}), patch.object(server.logger, "exception"):
                response = self.client.post("/api/practice", json=request)
                self.assertEqual(response.status_code, 200)
                self.assertNotIn("tajweed_feedback", response.json())

    def test_websocket_reports_word_tags_and_ayah_feedback(self):
        def compare(target_text, hyp_words, hyp_tags, surah=None, ayah=None):
            return {"target": target_text, "tags": hyp_tags}
        with patch.dict(sys.modules, {"src.tajweed.feedback": SimpleNamespace(compare=compare)}):
            with self.client.websocket_connect("/ws/recite", headers=ORIGIN) as ws:
                message = self.ready(ws, model="tajweed")
                self.assertEqual((message["model_used"], message["model_kind"], message["fallback_reason"]), ("tajweed", "tajweed", None))
                messages = self.drain(ws)
        updates = [m for m in messages if m["type"] == "update" and m.get("tajweed_tags")]
        self.assertTrue(updates)
        self.assertIn({"word_index": 0, "tags": ["qalqala"]}, updates[0]["tajweed_tags"])
        results = messages[-1]["results"]
        self.assertTrue(messages[-1]["complete"])
        self.assertEqual([w.get("tags") for w in results["1"]["words"]], [["qalqala"], None, None, ["qalqala"]])
        self.assertEqual([w.get("tags") for w in results["2"]["words"]], [None, ["qalqala"]])
        self.assertEqual(results["1"]["tajweed_feedback"], {"target": SURAH["ayahs"][0]["text"], "tags": [["qalqala"], [], [], ["qalqala"]]})
        self.assertIn("tajweed_feedback", results["4"])

    def test_plain_session_has_identical_results_without_tags(self):
        def results(model):
            with self.client.websocket_connect("/ws/recite", headers=ORIGIN) as ws:
                self.ready(ws, model=model)
                return self.drain(ws)[-1]["results"]
        plain, tagged = results("plain"), results("tajweed")
        for result in tagged.values():
            result.pop("tajweed_feedback", None)
            for word in result["words"]:
                word.pop("tags", None)
        self.assertEqual(tagged, plain)


class RegistryStartupTests(unittest.TestCase):
    def test_missing_default_model_fails_startup(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(server, "Engine", FakeEngine), \
                patch.dict(os.environ, {"RECITER_MODELS": f"plain={Path(directory) / 'missing'}"}):
            with self.assertRaisesRegex(RuntimeError, "Default model 'plain' is missing"):
                with TestClient(server.app):
                    pass

    def test_unavailable_plain_engine_is_503_and_failed_tajweed_load_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "plain").mkdir()
            (root / "tajweed").mkdir()
            def engine(spec):
                if spec.kind == "tajweed":
                    raise OSError("corrupt weights")
                return FakeEngine(spec)
            env = {"RECITER_MODELS": f"plain={root / 'plain'},tajweed={root / 'tajweed'},extra={root / 'missing'}"}
            with patch.object(server, "Engine", side_effect=engine), patch.dict(os.environ, env), patch.object(server.logger, "exception"):
                with TestClient(server.app) as client:
                    request = {"mode": "reading", "target": "قل", "audio": pcm16()}
                    self.assertEqual(client.post("/api/practice", json={**request, "model": "extra"}).status_code, 503)
                    body = client.post("/api/practice", json={**request, "model": "tajweed"}).json()
                    self.assertEqual((body["model_used"], body["fallback_reason"]), ("plain", "load_failed"))
                    models = {m["name"]: m for m in client.get("/health").json()["models"]}
                    self.assertEqual((models["tajweed"]["available"], models["tajweed"]["error"]), (False, "corrupt weights"))
                    self.assertFalse(models["extra"]["available"])


class EngineTagDecodingTests(unittest.TestCase):
    def test_tag_tokens_registered_as_special_survive_decoding(self):
        vocab = {1: "قل", 2: "هو", 9: "<|endoftext|>", 10: "<|startoftranscript|>", 11: "<|ar|>", 50: "<qalqala>"}
        tokenizer = SimpleNamespace(prefix_tokens=[10, 11], eos_token_id=9, all_special_ids=[9, 10, 11, 50],
                                    decode=lambda ids, **kwargs: " ".join(vocab[i] for i in ids))
        class Processor:
            def __init__(self):
                self.tokenizer = tokenizer
            def __call__(self, audio, **kwargs):
                return SimpleNamespace(input_features=torch.zeros(1, 80, 3000), attention_mask=torch.ones(1, 3000))
        engine = server.Engine.__new__(server.Engine)
        engine.device, engine.num_beams, engine.tag_ids, engine.processor = "cpu", 1, {50}, Processor()
        engine.model = SimpleNamespace(dtype=torch.float32, config=SimpleNamespace(max_target_positions=448),
                                       generate=Mock(return_value=SimpleNamespace(sequences=torch.tensor([[10, 11, 1, 50, 2, 9]]))))
        transcript = engine.transcribe(np.ones(16000, dtype=np.float32) * .1)
        self.assertEqual(transcript, "قل <qalqala> هو")
        self.assertEqual(server.heard_with_tags(transcript), ("قل هو", ["قل", "هو"], [["qalqala"], []]))
