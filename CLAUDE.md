# Rattil (Ai-Reciter)

Qālūn Quran recitation learning app: a Python Whisper inference API (`src/streaming`) and a Next.js web app (`web/`).

**Start here:** [`docs/LOCAL_SETUP.md`](docs/LOCAL_SETUP.md) lists which Hugging Face model and datasets to pull (private org `Mathani-Ayat`), where they go locally, how to run both servers, Mac-specific workarounds and the `/dev` page.

- Default model: `Mathani-Ayat/rattil-qaloon-v3` → `runs/rattil_qaloon_v3`, served with `python -m src.streaming.serve --model rattil-v3`.
- Pull model + data: `python src/deployment/pull_hf_assets.py` (pinned revisions inside).
- Weights, audio and `data/` are Git-ignored; never commit them.
- First run: the web server fetches/generates missing Quran cache, tajweed draft and reader audio (`web/lib/first-run.ts`; `RATTIL_AUTO_SETUP=0` to skip).
- Dev mode: `NEXT_PUBLIC_DEV_MODE=1` in `web/.env.local` enables `/dev` (off by default).
- Reviewer and evaluation background: `docs/REVIEWER_GUIDE.md`, `docs/PROJECT_MAP.md`.
