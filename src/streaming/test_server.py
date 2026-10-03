"""WebSocket lifecycle tests with fake ASR, never GPU or simulated model claims."""
import asyncio
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
