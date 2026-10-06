#!/usr/bin/env python3
"""Set up and run Rattil with one command (macOS, Windows, Linux).

    python run.py           # or ./run.sh (macOS/Linux), run.bat (Windows)

The first run creates .venv and installs the Python packages, installs the web
app's packages, downloads the models from Hugging Face (Mathani-Ayat), the Quran
text and pages and the reference readers' audio, then starts the recognition API
and the web app and opens http://127.0.0.1:3000. Later runs skip what is already
there and start in seconds. Ctrl+C stops everything.

Needs Python 3.10-3.13 (3.14 works when PyTorch has wheels for it) and Node.js 20+.
If the Hugging Face repos are private, log in first (.venv's `hf auth login`) or
set HF_TOKEN.
"""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
QURAN = WEB / "public/quran"
VENV = ROOT / ".venv"
WINDOWS = os.name == "nt"
PYTHON_RANGE = ((3, 10), (3, 14))
NODE_MIN = (18, 18)
# (pull_hf_assets.py name, folder, required to start the API)
MODELS = [("v4", "runs/rattil_qaloon_v4", True),
          ("tajweed-v2", "runs/rattil_qaloon_tajweed_v2", False),
          ("ayah-embed", "runs/rattil_ayah_embed", False)]
READERS = ["dataset_qaloon_hutafi", "dataset_qaloon_Husary", "dataset_qaloon_dokali"]
HF_HELP = ("The Mathani-Ayat repos on Hugging Face may still be private. Log in with an account that can read\n"
           f"them ({VENV / ('Scripts/hf.exe' if WINDOWS else 'bin/hf')} auth login) or set HF_TOKEN, then run again.")


def say(message):
    print(f"[rattil] {message}", flush=True)


def fail(message):
    print(f"\n[rattil] {message}", file=sys.stderr, flush=True)
    sys.exit(1)


def venv_python():
    return VENV / ("Scripts/python.exe" if WINDOWS else "bin/python")


def child_env(**extra):
    # UTF-8 everywhere: Windows would otherwise read the Arabic JSON files as cp1252.
    return {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "HF_HUB_DISABLE_TELEMETRY": "1",
            "NEXT_TELEMETRY_DISABLED": "1", **extra}


def run(command, cwd=ROOT, **extra):
    """Run a setup step with its output shown; True when it succeeded."""
    return subprocess.run([str(c) for c in command], cwd=cwd, env=child_env(**extra)).returncode == 0


def digest(*paths):
    h = hashlib.sha256()
    for path in paths:
        h.update(str(path.relative_to(ROOT)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def stamp_ok(stamp, value):
    return stamp.is_file() and stamp.read_text(encoding="utf-8") == value


# ---------- Python ----------

def ensure_python():
    if not venv_python().exists():
        low, high = PYTHON_RANGE
        if not low <= sys.version_info[:2] <= high:
            fail(f"Python {sys.version.split()[0]} is not supported. Install Python 3.12 from "
                 "https://www.python.org/downloads/ (Windows: tick \"Add python.exe to PATH\") and run again.")
        say(f"Creating the Python environment in {VENV.name} (Python {sys.version.split()[0]})")
        if subprocess.run([sys.executable, "-m", "venv", str(VENV)]).returncode != 0:
            fail("Could not create .venv. On Debian/Ubuntu install python3-venv first.")
    requirements = ROOT / "requirements.txt"
    stamp = VENV / ".rattil-requirements"
    wanted = digest(requirements)
    if stamp_ok(stamp, wanted):
        return
    say("Installing Python packages (PyTorch, Transformers, FastAPI...). The first time takes a few minutes.")
    python = venv_python()
    run([python, "-m", "pip", "install", "--upgrade", "pip"])
    if sys.platform.startswith("linux") and not shutil.which("nvidia-smi"):
        # PyPI's Linux PyTorch bundles CUDA (several GB); use the CPU build when there is no NVIDIA GPU.
        run([python, "-m", "pip", "install", "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"])
    if not run([python, "-m", "pip", "install", "-r", requirements]):
        fail("Installing the Python packages failed (see the messages above).")
    stamp.write_text(wanted, encoding="utf-8")


# ---------- Node / web packages ----------

def npm_command():
    node, npm = shutil.which("node"), shutil.which("npm")
    if not node or not npm:
        fail("Node.js is missing. Install Node.js 22 LTS from https://nodejs.org and run again.")
    version = subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip().lstrip("v")
    parts = tuple(int(p) for p in version.split(".")[:2] if p.isdigit())
    if parts < NODE_MIN:
        fail(f"Node.js {version} is too old. Install Node.js 22 LTS from https://nodejs.org and run again.")
    return npm


def ensure_web_packages(npm):
    lock = WEB / "package-lock.json"
    stamp = WEB / "node_modules/.rattil-lock"
    wanted = digest(lock)
    if stamp_ok(stamp, wanted):
        return
    say("Installing the web app's packages (npm ci)")
    if not run([npm, "ci", "--no-audit", "--no-fund"], cwd=WEB):
        fail("npm ci failed (see the messages above).")
    stamp.write_text(wanted, encoding="utf-8")


# ---------- Models and data ----------

def model_present(folder):
    return (ROOT / folder / "config.json").is_file() and any((ROOT / folder).glob("*.safetensors"))


def ensure_models(skip):
    python = venv_python()
    for name, folder, required in MODELS:
        if model_present(folder) or skip:
            continue
        say(f"Downloading model {name} from Hugging Face -> {folder}")
        if run([python, "src/deployment/pull_hf_assets.py", "--models", name, "--datasets"], HF_HUB_DISABLE_PROGRESS_BARS="0"):
            continue
        if required:
            fail(f"Could not download the recognition model ({name}).\n{HF_HELP}")
        say(f"Skipped {name}: download failed. The app runs without it.")


def ensure_data():
    python = venv_python()
    if not (QURAN / "manifest.json").is_file():
        say("Downloading the Quran pages and text (about 350 MB, once)")
        if not run([python, "src/dataset_collection/cache_quran_pages.py", "--skip-metadata"]):
            fail("Caching the Quran failed (see above). Check the internet connection and run again.")
    if not (QURAN / "tajweed/rules.json").is_file():
        say("Generating Qalun tajweed colours")
        if not run([python, "-m", "src.tajweed.build"]):
            fail("Generating the tajweed data failed (see above).")
    if not all((ROOT / "src/dataset_collection" / r / "metadata.jsonl").is_file() for r in READERS):
        say("Downloading the reference readers' audio (about 330 MB, once)")
        if not run([python, "src/deployment/pull_hf_assets.py", "--skip-model", "--readers-only"]):
            say(f"Reader audio not downloaded; listening and audio challenges are off until it is.\n{HF_HELP}")
    if not (QURAN / "text-embeddings.json").is_file():
        say("Embedding ayahs and words for the challenges (a few minutes, once)")
        if not run([python, "src/learning/build_text_embeddings.py"]):
            say("Embeddings not built; challenges use spelling similarity instead.")


# ---------- Web build ----------

def web_sources():
    files = [p for d in ("app", "components", "lib") for p in (WEB / d).rglob("*") if p.is_file()]
    files += [p for p in WEB.iterdir() if p.is_file() and p.suffix in {".ts", ".json", ".mjs", ".js"}
              and p.name not in {"tsconfig.tsbuildinfo", "next-env.d.ts"}]  # Next rewrites these itself
    return sorted(files)


def ensure_web_build(npm, env):
    build = WEB / ".next-production"
    stamp = build / "rattil-build"
    wanted = digest(*web_sources()) + env["NEXT_PUBLIC_RECITER_WS"]
    if (build / "BUILD_ID").is_file() and stamp_ok(stamp, wanted):
        return
    say("Building the web app (about a minute, again only after the code changes)")
    if subprocess.run([npm, "run", "build"], cwd=WEB, env=env).returncode != 0:
        fail("The web build failed (see above).")
    stamp.write_text(wanted, encoding="utf-8")


# ---------- Start and stop ----------

class WindowsServerJob:
    """Own server descendants: Windows kills them even if the console closes abruptly."""

    def __init__(self):
        self.handle = None
        if not WINDOWS:
            return
        import ctypes
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                        ("PerJobUserTimeLimit", ctypes.c_longlong), ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", wintypes.DWORD), ("Affinity", ctypes.c_size_t),
                        ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

        class IOCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in
                        ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                         "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", IOCounters),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.kernel.SetInformationJobObject.restype = wintypes.BOOL
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.get_last_error()
            self.close()
            raise ctypes.WinError(error)

    def assign(self, process):
        if self.handle:
            import ctypes
            if not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
                raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def port_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if not WINDOWS:  # like uvicorn and Node: a port left in TIME_WAIT by the last run is free
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def start(command, cwd, env, job=None):
    group = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if WINDOWS else {"start_new_session": True}
    process = subprocess.Popen([str(c) for c in command], cwd=cwd, env=env, **group)
    try:
        if job is not None:
            job.assign(process)
    except BaseException:
        stop(process)
        raise
    return process


def stop(process):
    if WINDOWS:
        if process.poll() is None:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, timeout=15)
            process.wait(timeout=15)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=15)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def answers(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status == 200
    except OSError:
        return False


def wait_until(label, url, processes, timeout):
    say(f"Waiting for the {label}...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if any(p.poll() is not None for p in processes):
            return False
        if answers(url):
            return True
        time.sleep(2)
    return False


def interrupted(*_):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setup-only", action="store_true", help="Install and download everything, then exit")
    parser.add_argument("--check", action="store_true", help="Start everything, confirm both servers answer, then stop")
    parser.add_argument("--dev", action="store_true", help="Run the web app with `next dev` (hot reload) instead of a production build")
    parser.add_argument("--skip-models", action="store_true", help="Do not download models; without the recognition model only the web app starts")
    parser.add_argument("--web-port", type=int, default=3000)
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Where the model runs (auto: an NVIDIA GPU if present, else CPU)")
    parser.add_argument("--beams", type=int, choices=[1, 3, 5], default=3, help="Beam search width: 3 is more accurate, 1 is faster on slow computers")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser")
    parser.add_argument("--public-url", help="Server deployment: the HTTPS address users open (e.g. https://rattil.example.com). "
                        "A reverse proxy must send /ws/recite, /health, /api/practice and /api/search to the API (see deploy/Caddyfile)")
    args = parser.parse_args()
    # Ctrl+C, Ctrl+Break and closing the terminal must all stop the servers, however run.py was launched.
    for name in ("SIGINT", "SIGTERM", "SIGHUP", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), interrupted)

    ensure_python()
    npm = npm_command()
    ensure_web_packages(npm)
    ensure_models(args.skip_models)
    ensure_data()
    if args.setup_only:
        say("Setup complete. Start Rattil with the same command, without --setup-only.")
        return

    web_url = f"http://127.0.0.1:{args.web_port}"
    public = (args.public_url or "").rstrip("/")
    # Behind the proxy the browser reaches the API on the same origin (https -> wss).
    socket_url = f"{public.replace('http', 'ws', 1)}/ws/recite" if public else f"ws://127.0.0.1:{args.api_port}/ws/recite"
    api_env = child_env(RECITER_ALLOWED_ORIGINS=",".join([web_url, f"http://localhost:{args.web_port}"] + [public] * bool(public)))
    web_env = child_env(NEXT_PUBLIC_RECITER_WS=socket_url, RATTIL_PYTHON=str(venv_python()))
    with_api = model_present(MODELS[0][1])
    if not with_api:
        say("No recognition model: starting the web app only (recitation checks are off).")
    for port, flag in ((args.web_port, "--web-port"), (args.api_port, "--api-port"))[: 2 if with_api else 1]:
        if not port_free(port):
            fail(f"Port {port} is already in use (is Rattil already running?). Stop that program or pick another port with {flag}.")
    if not args.dev:
        ensure_web_build(npm, web_env)

    processes = []
    job = WindowsServerJob()
    try:
        if with_api:
            processes.append(start([venv_python(), "-m", "src.streaming.serve", "--model", "rattil-v4", "--port", args.api_port,
                                    "--device", args.device, "--beams", args.beams], ROOT, api_env, job))
        processes.append(start([npm, "run", "dev" if args.dev else "start", "--", "-p", args.web_port], WEB, web_env, job))
        ready = (not with_api or wait_until("recognition API", f"http://127.0.0.1:{args.api_port}/health", processes, 600)) \
            and wait_until("web app", web_url, processes, 300)
        if not ready:
            fail("Rattil did not start (see the messages above).")
        if args.check:
            say("Check passed: " + ("the API and the web app answer." if with_api else "the web app answers."))
            return
        say(f"Rattil is running at {public or web_url}  (Ctrl+C to stop)")
        if not args.no_browser and not public:
            webbrowser.open(web_url)
        while all(p.poll() is None for p in processes):
            time.sleep(1)
        fail("A server stopped unexpectedly (see the messages above).")
    except KeyboardInterrupt:
        say("Stopping...")
    finally:
        try:
            for process in reversed(processes):
                try:
                    stop(process)
                except (OSError, subprocess.TimeoutExpired) as error:
                    say(f"Server shutdown: {error}")
        finally:
            # The job also owns descendants left behind if npm/the API parent already exited.
            # If Windows terminates this launcher, it closes this handle automatically.
            job.close()


if __name__ == "__main__":
    main()
