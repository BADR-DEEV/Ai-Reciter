"""WebSocket lifecycle tests with fake ASR, never GPU or simulated model claims."""
import asyncio
import base64
import unittest
from unittest.mock import patch
import numpy as np
from fastapi.testclient import TestClient
from . import server


class FakeEngine:
    def __init__(self):
        self.device = "test"
        self.lock = asyncio.Lock()
        self.last_latency_ms = None

    def transcribe(self, audio):
        return "قل هو الله احد الله الصمد لم يلد ولم يولد ولم يكن له كفؤا احد"


@unittest.skipUnless((server.ASSETS / "surahs/112.json").exists(), "Requires locally cached Qaloon text")
class ServerTests(unittest.TestCase):
    def setUp(self):
        self.patch = patch.object(server, "Engine", FakeEngine)
        self.patch.start()
        self.client = TestClient(server.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.patch.stop()

    def test_health_reports_capacity(self):
        body = self.client.get("/health").json()
        self.assertEqual(body["max_sessions"], 2)
        self.assertFalse(body["busy"])

    def test_flagged_or_empty_asr_abstains_without_scoring_practice(self):
        audio = base64.b64encode(np.full(16000, 3200, dtype="<i2").tobytes()).decode()
        with patch.object(FakeEngine, "check", return_value=("", []), create=True):
            for mode in ("sound", "reading"):
                response = self.client.post("/api/practice", json={"mode": mode, "target": "حا", "alternatives": ["ها"], "audio": audio})
                self.assertEqual(response.status_code, 503)
                self.assertNotIn("score", response.json())
                self.assertNotIn("verdict", response.json())

    def test_start_ayah_and_ping(self):
        with self.client.websocket_connect("/ws/recite", headers={"origin": "http://127.0.0.1:3000"}) as ws:
            ws.send_json({"surah": 112, "start_ayah": 3})
            self.assertEqual(ws.receive_json()["current"], 3)
            ws.send_json({"type": "ping"})
            self.assertEqual(ws.receive_json()["type"], "pong")
            ws.send_json({"type": "stop"})
            self.assertEqual(ws.receive_json()["type"], "finished")
        self.assertEqual(self.client.get("/health").json()["sessions"], 0)

    def test_invalid_start_ayah_rejected(self):
        with self.client.websocket_connect("/ws/recite", headers={"origin": "http://127.0.0.1:3000"}) as ws:
            ws.send_json({"surah": 112, "start_ayah": 999})
            self.assertEqual(ws.receive_json()["type"], "error")

    def test_file_mode_accepts_faster_than_realtime_and_drains_at_eof(self):
        with self.client.websocket_connect("/ws/recite", headers={"origin": "http://127.0.0.1:3000"}) as ws:
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
            with self.client.websocket_connect("/ws/recite", headers={"origin": "http://127.0.0.1:3000"}) as ws:
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
            with self.client.websocket_connect("/ws/recite", headers={"origin": "http://127.0.0.1:3000"}) as ws:
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
        with self.client.websocket_connect("/ws/recite", headers={"origin": "http://127.0.0.1:3000"}) as ws:
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
        with self.client.websocket_connect("/ws/recite", headers={"origin": "http://127.0.0.1:3000"}) as ws:
            ws.send_json({"surah": 112})
            ws.receive_json()
            ws.send_bytes(np.array([np.nan], dtype="<f4").tobytes())
            self.assertEqual(ws.receive_json()["type"], "error")

    def test_worker_failure_surfaces_without_another_client_message(self):
        with patch.object(FakeEngine, "transcribe", side_effect=RuntimeError("Test inference failure")):
            with self.client.websocket_connect("/ws/recite", headers={"origin": "http://127.0.0.1:3000"}) as ws:
                ws.send_json({"surah": 112})
                ws.receive_json()
                for _ in range(6):
                    ws.send_bytes(np.full(3200, 0.1, dtype="<f4").tobytes())
                self.assertEqual(ws.receive_json()["type"], "error")
