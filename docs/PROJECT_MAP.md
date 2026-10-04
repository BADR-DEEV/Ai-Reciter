# Project map and preservation-first cleanup

## Authoritative locations

| Concern | Location |
| --- | --- |
| Canonical Qālūn source | `src/dataset_collection/QaloonData_v10(1).json` |
| Core readers | `src/dataset_collection/dataset_qaloon_hutafi`, `dataset_qaloon_dokali`, `dataset_qaloon_Husary` |
| Latest Trabulsi candidate package | `src/dataset_collection/dataset_qaloon_trabulsi` |
| Latest rebuilt Taha candidate package | `src/dataset_collection/dataset_qaloon_taha` |
| Original Taha dataset, preserved | `data/archives/datasets/dataset_qaloon_taha_original_v1` |
| Immutable full source recordings | `ahmad_tarabulsi/mp3`, `taha_al_fahad_juz_amma/mp3` |
| Historical cuts/review evidence | original root dataset folders and `data/segmentation_review` (not deleted) |
| Production admission | `src/training/reviewed_audio.py` |
| Explicit private experimental admission/splits | `src/training/experiment_data.py` |
| Current LoRA entry | `src/training_with_gpu/train_tarteel_lora.py` |
| Existing full/tiny entries | `src/training_with_gpu/train_base_full.py`, `train_tarteel_base.py`, `train_tarteel_tiny.py` |
| ASR metrics/comparison | `src/training/asr_metrics.py`, `src/training_with_gpu/compare_asr_runs.py` |
| Live recitation/product | `src/streaming`, `web/app/studio`, `web/lib/playback.ts` |
| Challenge/learner policy | `web/lib/challenges.ts`, `web/lib/adaptive-challenges.ts`, `web/app/games` |

Dataset folders contain the other readers' usual `metadata.jsonl`/`audio/` schema,
paired `audio_raw/`, honest `coverage.jsonl`, review/provenance bundles and audits.
A candidate folder name is **not approval**. Trabulsi has a publisher-policy-based
private decision; Taha now has a separate user-directed private-research decision
with **unverified source-use status**, not a manufactured license or release grant.
Coverage inventories
are never training manifests. Source/paired WAV hashes were verified on relocation.

Generated audio, model weights, runs, conversion staging and review assets stay
Git-ignored. `models/segmentation/` is now explicitly ignored so its tokenizer/
conversion files do not accidentally enter a source commit alongside ignored
weights. The deployed/private model layouts are unchanged.

This cleanup establishes authoritative datasets and a map; it does **not** delete
older experiments, user edits, source recordings, useful caches or checkpoints.
Legacy scripts remain for reproducibility. In particular `collect_taha.py` is a
historically misleading filename for a Ganayni downloader and is **not** used to
establish Taha provenance or obtain a rights grant. Do not automate blocked
Assabile resolution routes to “fix” that legacy script.

See [the two AI approaches/current run](AI_APPROACHES_AND_CURRENT_RUN.md) for the
actual training membership, experimental status, comparison limitations and
personalization roadmap. Commit/push, model deployment and public publication
remain separate explicit operations.
