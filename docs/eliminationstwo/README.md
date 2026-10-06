# Eliminations round 2: pitch decks

Two Arabic decks for Rattil. They tell the same story with the same numbers, and every number comes from the repo docs. Each slide has Arabic speaker notes.

| File | What it is |
| --- | --- |
| `rattil_official_template.pptx` / `.pdf` | 16 slides on the challenge's official participant template (Readex Pro is embedded in the template). |
| `rattil_custom_design.pptx` / `.pdf` | 13 slides in Rattil's own colours: deep green, gold, and the studio's green/red word tiles. |

The PDFs embed their fonts, so they look the same on any machine. To edit the custom deck you need two free Google Fonts installed: **Amiri** and **IBM Plex Sans Arabic**.

The screenshots are real app output, recorded with Sheikh Waleed's voice, which was never used in training:
- the studio running Al-Ikhlāṣ with ayah 3 cut out on purpose;
- search on Al-Bayyina 5.

## Fill in before presenting

- **Team name**: shown as `[اسم الفريق]` on the cover, team and closing slides.
- **Team roles and photos**: the names and roles on the team slide were guessed from git history. Confirm them and add photos.
- **Demo link or contact**: shown as `[رابط ...]` on the last slide. The GitHub repo is private, so judges can't open it.

## Gaps against the rules pack (`rules.pdf`)

- **Tafsir and translation source**: the app uses `api.quran-tafseer.com`, which is not on the approved list. Moving to quranenc.com would turn the "قيد التنفيذ" row into "مطبّق".
- **No teacher review yet**: no certified Qālūn teacher has signed off the tajweed colours.
- **No learner testing**: every accuracy number comes from professional reciters.
- **Quran text check**: the KFGQPC Qālūn text was checked against a mirror. The check against the official portal is still open.
- **Out-of-date repo docs**: `README.md` still describes the old model and says the app does not assess tajweed. The old `presentation/` deck says it covers the whole Quran.

Sources for the numbers: `docs/QALOON_MODEL_V1.md`, `docs/QALOON_MODEL_V2.md`, `docs/QALOON_TAJWEED.md`, `docs/QURAN_SEARCH.md`, `docs/LEARN_TO_READ.md`.
