"""Exercise real process trees and listening ports without loading Rattil models."""
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import unittest

import run


def unused_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def listening(port):
    with socket.socket() as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_for(predicate):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return False


@unittest.skipUnless(os.name == "nt", "Windows Job Object lifecycle tests")
class LauncherShutdownTests(unittest.TestCase):
    def test_job_closes_descendants_after_their_parent_exits(self):
        port = unused_port()
        server = f"from http.server import HTTPServer, BaseHTTPRequestHandler; HTTPServer(('127.0.0.1', {port}), BaseHTTPRequestHandler).serve_forever()"
        parent = f"import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', {server!r}]); time.sleep(0.5)"
        job = run.WindowsServerJob()
        process = run.start([sys.executable, "-c", parent], run.ROOT, run.child_env(), job)
        try:
            self.assertTrue(wait_for(lambda: listening(port)))
            process.wait(timeout=10)
            self.assertTrue(listening(port), "Grandchild should outlive its direct parent before job cleanup")
        finally:
            job.close()
        self.assertTrue(wait_for(lambda: not listening(port)), "Job close must stop orphaned listener")

    def test_abrupt_launcher_exit_releases_both_server_ports(self):
        ports = [unused_port(), unused_port()]
        launcher = "import run, sys, time; job = run.WindowsServerJob(); "
        for port in ports:
            server = f"from http.server import HTTPServer, BaseHTTPRequestHandler; HTTPServer(('127.0.0.1', {port}), BaseHTTPRequestHandler).serve_forever()"
            parent = f"import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', {server!r}]); time.sleep(60)"
            launcher += f"run.start([sys.executable, '-c', {parent!r}], run.ROOT, run.child_env(), job); "
        launcher += "time.sleep(60)"
        process = subprocess.Popen([sys.executable, "-c", launcher], cwd=Path(run.ROOT))
        try:
            self.assertTrue(wait_for(lambda: all(listening(port) for port in ports)))
        finally:
            # TerminateProcess bypasses Python's finally block, as abrupt console closure can.
            process.terminate()
            process.wait(timeout=10)
        self.assertTrue(wait_for(lambda: all(not listening(port) for port in ports)), "Both server descendants must exit with their launcher")


if __name__ == "__main__":
    unittest.main()
