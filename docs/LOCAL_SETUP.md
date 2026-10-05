# Local setup: where the model and data come from

Read this first in a new session. It covers what to download, where it goes and
how to run Rattil. The `/dev` page (dev mode only) shows the same information live.

## Hugging Face sources (private, org `Mathani-Ayat`)

You need a Hugging Face login with access to the org: `hf auth login`.
Pinned revisions live in `src/deployment/pull_hf_assets.py`. Update them there when
the team publishes a new version.

| Repo | Type | Local path | What it is |
| --- | --- | --- | --- |
| `Mathani-Ayat/rattil-qaloon-v3` | model | `runs/rattil_qaloon_v3/` | **Default model.** Full Whisper-base fine-tune from `deepdml/whisper-base-ar-quran-mix-norm`, 10 readers, 5,205 clips. Waleed (held-out voice): 2.2% WER at normal speed, 13.4% at 1.25×, 21.2% at 1.5×. |
| `Mathani-Ayat/rattil-qaloon-v2`, `-v1` | model | not pulled | Earlier versions (v2: 2.0% Waleed WER at normal speed, worse on fast recitation). |
| `Mathani-Ayat/qalon-reciter` | model | not pulled | Mirror of the original team release. |
| `Mathani-Ayat/qaloon-reciter-dataset` | dataset | `data/hf/qaloon-reciter-dataset/` | Approved readers Huthaify, Husary, Dokali (569 ayahs each) plus Garu, with `train/validation/test.jsonl` split manifests. |
| `Mathani-Ayat/qaloon-reciter-experiments` | dataset | `data/hf/qaloon-reciter-experiments/` | Waleed (held-out test voice) and Trabulsi. Its `sources.json` marks them unauthorized: private research only. |
| `Mathani-Ayat/qaloon-new-reciters` | dataset | `data/hf/qaloon-new-reciters/` | Abusnaina, Akri, Daawob, Deeban, Kshidan, Qeniwa (from mp3quran, used for v2/v3) plus `unapproved/`. |

The older `BadrSh/qalon-reciter` full model (about 26% test WER) restores to
`runs/gpu_base_full/` via `src/deployment/restore_local_full.py`. It is kept only
for comparison.

All of these paths are Git-ignored. Never commit audio or weights.

## Pull everything

```bash
python src/deployment/pull_hf_assets.py                   # model + all 3 datasets (~1.6 GB)
python src/deployment/pull_hf_assets.py --datasets        # model only
python src/deployment/pull_hf_assets.py --skip-model --datasets qaloon-reciter-dataset
python src/dataset_collection/cache_quran_pages.py --skip-metadata   # Quran page SVGs/text (~350 MB)
```

The pull script symlinks `src/dataset_collection/dataset_qaloon_{hutafi,Husary,dokali}`
to the Hub copy. The web app reads reference audio from there (see
`web/lib/reciters.ts`). It never replaces an existing real folder.

### First run sets itself up

None of the generated or downloaded data is in Git. When `next dev` or `next start`
starts, `web/instrumentation.ts` → `web/lib/first-run.ts` runs these steps in the
background. Each one runs only if its output is missing:

1. `cache_quran_pages.py --skip-metadata`: Quran page SVGs, geometry and per-surah text (~350 MB).
2. `src/learning/build_qalon_tajweed.py`: the tajweed draft `web/public/quran/qalon_majwad_mushaf.json` (about 1 s).
3. `pull_hf_assets.py --readers-only`: Huthaify, Husary and Dokali reference audio (~330 MB).
   Progress goes to `data/hf/pull-status.json` and is served at `/api/reader-audio`.
4. Once the readers are present, `build_audio_similarity.py` builds listening-distractor
   indexes. This step is optional; challenges fall back to text similarity. It needs a
   working torchaudio, and Anaconda's on the Mac is broken, so it fails there harmlessly.

Watch for `[rattil]` lines in the `next dev` output. Set `RATTIL_AUTO_SETUP=0` to skip all
of this. Pressing **Listen** on an ayah with missing audio then offers a
**Download reader audio** button. Any failure, usually a missing `hf auth login`, also
shows that button. Set `RATTIL_PYTHON` if `python3` isn't the Python with
`huggingface_hub`.

`cache_quran_pages.py` fetches the Hafs numbering reference from the jsdelivr mirror of
`quran-json`. The original risan/quran-json GitHub raw URL returns 404.

## Run

```bash
# Terminal 1: inference API on :8000 (plain model + tajweed model side by side)
python -m src.streaming.serve --model rattil-v3
#   also offers tajweed=rattil-tajweed-v1 (runs/rattil_qaloon_tajweed_v1); it loads on first
#   use once that folder exists, and requests fall back to plain until then.
#   explicit: --models plain=rattil-v3,tajweed=runs/rattil_qaloon_tajweed_v1
#   other presets: gpu-full-base (old team model), deepdml (adapter, training PC only)

# Terminal 2: web on :3000 (the API only accepts origins on port 3000)
cd web && npm install && npx next dev --hostname 127.0.0.1 -p 3000
```

Pages: `/` course, `/learn`, `/studio`, `/tajweed` (letter sounds, rules, Qālūn topics),
`/games`, `/profile`, and `/dev` (dev mode only).

Tajweed colors need `web/public/quran/tajweed/`. The first start builds it; to rebuild after
changing rules run `python -m src.tajweed.build` (about 7 s). See `docs/QALOON_TAJWEED.md`.

### Mac notes

- Use `--device cpu`. There is no CUDA, and whisper-base is fast enough on CPU.
- Anaconda's torchvision is broken. Prefix the server command with
  `PYTHONPATH=src/deployment/mac_shim`, which hides torchvision from transformers.
- Another local project (`rayaq`) listens on `*:3000` over IPv6. Pass `-p 3000`
  explicitly so Next binds 127.0.0.1:3000 instead of moving to 3001.
- Python tests that compare temp paths fail under macOS's `/var` symlink.
  Run them with `TMPDIR=/private/tmp`.

```bash
PYTHONPATH=src/deployment/mac_shim python3 -m src.streaming.serve --model rattil-v3 --device cpu
```

## Dev mode

`web/.env.local` (Git-ignored) controls it. The default in `web/.env.example` is off:

```
NEXT_PUBLIC_DEV_MODE=1   # shows /dev and a "Dev" header link; 0 or unset = hidden (404)
```

`/dev` shows the live `/health` of the API (loaded model, device, dtype, beams), every
local model's specs and scores, and every data file or folder with its location, size,
what reads it and where it comes from. Restart `next dev` after changing the flag.

## Limits

The plain models recognise recited words; they do not judge tajweed. The tajweed model
(`docs/QALOON_TAJWEED.md`) adds tags for audible rules, but it is an early model trained only
on correct recitations. They were tested on
professional reciters, not learners. Source recordings' licenses are not
established, so keep the models and datasets private.
