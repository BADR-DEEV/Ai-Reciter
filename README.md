# Rattil · رَتِّل

Learn to read and recite the Quran in the Qālūn riwāyah, from the first letter to whole surahs.
The name comes from 73:4, *wa rattili l-qurʾāna tartīlā*: "and recite the Quran with measured care."

- **Learn to read:** a beginner course from the alphabet to short surahs, in English or Arabic.
- **Recitation studio:** recite into the microphone and see each word turn green as it is heard,
  red when it is skipped. Follow a reference reader ayah by ayah.
- **Check my ahkam:** a second model listens for tajweed (ghunna, ikhfāʾ, qalqala, madd, tas-hīl,
  ṣilah) and lists what to fix.
- **Find an ayah:** recite a few words and Rattil finds where they are in the Quran.
- **Tajweed:** letter articulation, every rule with Qālūn's counts, and the topics of the riwāyah,
  with the rules coloured in the text.
- **Challenges:** next-ayah recall, listening, surah and word quizzes.

Everything runs on your own computer. Recordings are processed locally and never uploaded.

## Run it with one command

You need [Git](https://git-scm.com/downloads), [Python](https://www.python.org/downloads/) 3.10 to
3.13 (3.12 recommended) and [Node.js](https://nodejs.org) 20 or newer (22 LTS recommended).
On Windows, tick **"Add python.exe to PATH"** in the Python installer.

```bash
git clone https://github.com/BADR-DEEV/Ai-Reciter.git
cd Ai-Reciter
```

| System | Command |
| --- | --- |
| macOS / Linux | `./run.sh` |
| Windows | `run.bat` (or double-click it) |
| Any | `python run.py` (`python3 run.py` on macOS/Linux) |

The browser opens **http://127.0.0.1:3000** when everything is ready. Press **Ctrl+C** in the
terminal to stop.

On phones, the app uses a bottom navigation bar. Audio-input screens show a large
recording button in its centre. The five slots are studio, ayah search, recording,
challenges and **Menu**. The menu opens home, learning/letters, tajweed, profile
and other pages. Voice
search also keeps its decorative listening button in the page content on phones.

Closing the launcher terminal also stops its servers. On Windows, the launcher
owns both server process trees through a kill-on-close Job Object, including
children started by npm. Closing only the browser tab leaves the launcher running.

The first run takes 10 to 20 minutes, depending on your connection. It downloads about 3 GB and
needs about 5 GB of disk space:

1. Creates a Python environment in `.venv` and installs PyTorch, Transformers and FastAPI.
2. Installs the web app's packages (`npm ci`).
3. Downloads the models from Hugging Face: the recognition model, the tajweed model and the ayah
   embedder (about 1.1 GB, see [Models](#models)).
4. Downloads the Quran pages and text (about 350 MB) and the reference readers' audio (about 330 MB),
   and builds the tajweed colours and the challenge index.
5. Builds the web app, starts the recognition API (port 8000) and the web app (port 3000).

Later runs skip what is already there and start in under a minute.

### Options

| Option | What it does |
| --- | --- |
| `--beams 1` | Faster recognition on slow computers (the default 3 is more accurate) |
| `--device cuda` | Run the model on an NVIDIA GPU (`auto`, the default, uses one when PyTorch sees it) |
| `--web-port 3001 --api-port 8001` | Use other ports |
| `--dev` | Run the web app with hot reload, for development |
| `--setup-only` | Install and download everything, then exit |
| `--check` | Start everything, confirm both servers answer, then stop |
| `--no-browser` | Do not open the browser |
| `--skip-models` | Do not download models; only the web app starts (no recitation checks) |
| `--public-url https://…` | Run on a server behind HTTPS (see [Run it on a server](#run-it-on-a-server)) |

Pass them to any of the commands, for example `./run.sh --beams 1` or `run.bat --beams 1`.

### If something goes wrong

- **"Could not download the recognition model".** The models and the readers' audio come from the
  [Mathani-Ayat](https://huggingface.co/Mathani-Ayat) organisation on Hugging Face. While those repos
  are private, log in with an account that can read them, then run again:
  `.venv/bin/hf auth login` (Windows: `.venv\Scripts\hf.exe auth login`), or set `HF_TOKEN`.
- **"Port 3000 is already in use".** Rattil is probably already running in another terminal. Stop it
  or use `--web-port`.
- **Python or Node.js not found.** Install them from the links above, open a new terminal and run
  again. On Windows, use the python.org installer rather than the Microsoft Store.
- **The microphone does not work.** Open the app at `http://127.0.0.1:3000` or
  `http://localhost:3000` (browsers only allow the microphone there or over HTTPS) and allow access.
- **Recognition is slow.** Use `--beams 1`. The model runs on the CPU unless you have an NVIDIA GPU.
- **Start over.** Delete `.venv` (Python packages) or `web/node_modules` (web packages) and run
  again. Downloaded models live in `runs/` and data in `data/` and `web/public/quran/`.

## Run it on a server

On a Linux server (Ubuntu 24.04 tested, 2+ vCPUs and 4 GB RAM; no GPU needed) install `python3-venv`,
Node.js 22 and [Caddy](https://caddyserver.com/docs/install), log in to Hugging Face
(`HF_TOKEN` or `~/.cache/huggingface/token`), open ports 80 and 443, then:

```bash
python3 run.py --setup-only
# Replace RATTIL_HOST with your domain, or <ip-with-dashes>.sslip.io without one
sed "s/RATTIL_HOST/rattil.example.com/" deploy/Caddyfile | sudo tee /etc/caddy/Caddyfile && sudo systemctl reload caddy
sed "s/RATTIL_HOST/rattil.example.com/" deploy/rattil.service | sudo tee /etc/systemd/system/rattil.service
sudo systemctl daemon-reload && sudo systemctl enable --now rattil
```

Caddy gets the HTTPS certificate (browsers need HTTPS for the microphone) and sends the recognition
API's paths to port 8000 and everything else to the web app on 3000. `--public-url` points the
browser's WebSocket at that address and allows it as an origin. Logs: `journalctl -u rattil -f`.

Every push to `main` deploys to the team's EC2 server (`.github/workflows/deploy.yml`): GitHub signs in
to AWS with OIDC (no stored secrets), Systems Manager checks out the commit on the server and
`deploy/update.sh` restarts Rattil, which is down for a minute or two while it rebuilds.

## Run the parts by hand

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/deployment/pull_hf_assets.py               # models + readers' audio + datasets
python src/dataset_collection/cache_quran_pages.py --skip-metadata

# Terminal 1: recognition API on :8000 (plain model + tajweed model)
python -m src.streaming.serve --model rattil-v4 --beams 3

# Terminal 2: web app on :3000
cd web && npm ci && npm run dev
```

The web server also fetches anything missing when it starts (`web/lib/first-run.ts`).
[`docs/LOCAL_SETUP.md`](docs/LOCAL_SETUP.md) lists every Hugging Face source, pinned revisions and
the `/dev` page.

## Models

All three are fine-tuned for Qālūn and downloaded by `run.py` into `runs/`:

| Model | Size | What it does |
| --- | --- | --- |
| [`Mathani-Ayat/rattil-qaloon-v4`](https://huggingface.co/Mathani-Ayat/rattil-qaloon-v4) | 290 MB | Whisper-base: recognises the recited words. 2.0% word error on a reader it never heard. |
| [`Mathani-Ayat/rattil-qaloon-tajweed-v2`](https://huggingface.co/Mathani-Ayat/rattil-qaloon-tajweed-v2) | 290 MB | Writes a token after each word for every tajweed rule it hears ("Check my ahkam"). |
| [`Mathani-Ayat/rattil-ayah-embed`](https://huggingface.co/Mathani-Ayat/rattil-ayah-embed) | 520 MB | Ayah embeddings that pick close, confusable answers for the challenges. |

How they were trained and tested: [`docs/QALOON_TAJWEED.md`](docs/QALOON_TAJWEED.md),
[`docs/CHALLENGE_DISTRACTORS.md`](docs/CHALLENGE_DISTRACTORS.md),
[`docs/QALOON_MODEL_V2.md`](docs/QALOON_MODEL_V2.md).

## Project layout

| Path | Contents |
| --- | --- |
| `run.py`, `run.sh`, `run.bat` | One-command setup and start |
| `web/` | Next.js app: course, studio, tajweed, search, challenges |
| `src/streaming/` | FastAPI recognition server: live WebSocket recitation, word matching, search |
| `src/tajweed/` | Qālūn tajweed rule engine, colours, tajweed-model labels and evaluation |
| `src/learning/` | Challenge indexes (ayah embeddings, audio similarity) |
| `src/dataset_collection/` | Quran text and page cache, dataset preparation |
| `src/training_with_gpu/`, `src/training/` | Model training ([`docs/SPEAKER_ROBUST_TRAINING.md`](docs/SPEAKER_ROBUST_TRAINING.md)) |
| `src/deployment/` | Hugging Face download and publishing |
| `docs/` | Design notes, results and guides |

## More documentation

- [Learn-to-read course](docs/LEARN_TO_READ.md), [Find an ayah](docs/QURAN_SEARCH.md),
  [Qālūn tajweed](docs/QALOON_TAJWEED.md), [Challenge answers](docs/CHALLENGE_DISTRACTORS.md)
- [Reviewer guide](docs/REVIEWER_GUIDE.md), [Project map](docs/PROJECT_MAP.md)
- [Detailed project notes](docs/PROJECT_NOTES.md): Quran assets, streaming behaviour, training
  experiments and the audio review workflow (the earlier README)

## Checks

```bash
.venv/bin/python -m pip install pytest                      # Windows: .venv\Scripts\python
.venv/bin/python -m pytest -q src/streaming
.venv/bin/python -m unittest src.tajweed.test_engine src.tajweed.test_targets
cd web && npm run typecheck
```

The **One-command run** workflow (GitHub Actions, started by hand) runs `run.bat` and `run.sh` on
clean Windows and macOS machines.

## Limits

- Recognition was trained on al-Fātiḥah and surahs 78 to 114. Other surahs work but are less accurate.
- The tajweed model has heard correct recitation and plain synthetic readings, not learners' real
  mistakes: "missed" means "not heard". The models were tested on professional readers, not
  learners. Rattil helps you practise; it does not replace a teacher.
- Tajweed colours come from a rule engine based on the Libyan Awqaf curriculum for Qālūn; they are
  machine-applied, not a certified mushaf.

## Credits

Quran page images and ayah timings from [MP3Quran](https://www.mp3quran.net/); Qālūn text from the
King Fahd Complex (KFGQPC) edition; tafsir and translations from the Quran Tafseer API; numbering
reference from [risan/quran-json](https://github.com/risan/quran-json). Fonts (SIL Open Font
License, in `web/public/fonts`): Amiri, Cairo, DM Sans, Manrope. Respect each source's terms when
you reuse them.
