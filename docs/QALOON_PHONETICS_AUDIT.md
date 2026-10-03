# Qālūn phonetics: audit before teaching

**Status: experimental / requires qualified Qālūn review. No scholar-approved
coverage is claimed.** The Latin text is a reading aid, not a substitute for
listening, Arabic-script learning or instruction by a qualified teacher.

## Files

- Rules: `web/lib/qaloon-g2p.ts` (`qaloon-orthographic-0.1`).
- Reader component: `web/components/phonetic-aid.tsx`; opt-in in the studio.
- Audit generator: `web/scripts/phonetics-audit.cjs`.
- Full local audit: **`docs/generated/qaloon-phonetics-audit.csv`** (6,210 cached
  Qālūn ayahs), plus `phonetics-summary.json`.
- Regression fixtures: `web/tests/phonetics.spec.ts`.

Regenerate with `cd web; npm.cmd run phonetics:audit` (run as separate commands
if using a shell that does not support separators). It reads the exact cached
Qālūn source text, not ASR-normalized Arabic, and never changes Quran text or
labels. The CSV is UTF-8 with BOM for spreadsheet tools; it is a generated,
Git-ignored local artifact. Back up reviewer-edited sheets **before** regenerating:
the generator intentionally rebuilds draft columns, not an approval database.

## What follows Hafs-style transliteration design

We consulted Quran JSON's published transliteration methodology: explicit
vowels, consonant doubling, sun-letter assimilation, joining at hamzat al-wasl,
and end-of-ayah pause forms. That project warns that its own machine outputs
are not qualified-reader reviewed. Its Hafs outputs are not Qālūn labels.
Our implementation is independently written; no transliteration code or Hafs
verse Latin output was copied. Hafs Arabic is used only as a mapped comparison.

Qālūn source determines vowels and words. Mapping is by existing aligned regions,
not `Qaloon ayah N == Hafs ayah N`. One source ayah can map to multiple provider
verses or only a portion of a provider verse. A difference is not necessarily an
error: e.g. **maliki** versus **māliki**, or different verse divisions.

## Human-checkable regression cases

These are engineering checks, **not a religious sign-off**.

| Source/case | Draft pause reading | Check |
|---|---|---|
| Qālūn Fātiḥah 1 | al-ḥamdu lillāhi rabbi l-ʿālamīn | Long ā in ʿālamīn; not simply copying normalized consonants |
| Qālūn Fātiḥah 2 | ar-raḥmāni r-raḥīm | Sun-letter assimilation, doubled r; no tripled r |
| Qālūn Fātiḥah 3: مَلِكِ | maliki yawmi d-dīn | Do **not** introduce Hafs māliki |
| Fātiḥah 5 | ihdinā ṣ-ṣirāṭa l-mustaqīm | Reviewed-starting-vowel candidate; Maghrebi initial notation flagged |
| Fātiḥah 6: اَلذِينَ | lladhīna in connected context | Relative pronoun is not dh-dhīna |
| Fātiḥah 7: ضَّآلِّينَ | ḍ-ḍāllīn after article | Madd mark on alif must not become a new hamza |
| قُلْ هُوَ اَللَّهُ أَحَدٌ | qul huwa llāhu ʾaḥad | Allah's orthographic long vowel and end pause |
| Disjoint letters أَلَٓمِّٓ | alif lām mīm | Letter names, not an invented word "alammi" |

`ending="pause"` stops at the final word of the supplied ayah; `"connect"`
retains final inflection. These do **not** model arbitrary internal stop choices
or all across-ayah liaison. A reviewer must define the exact wasl/waqf context.

## Rules present

- Arabic consonants with scholarly-style Latin distinctions (ḥ/ṣ/ḍ/ṭ/ẓ, ʿ, ʾ).
- Fatḥa/kasra/ḍamma, common tanwin and source Maghrebi tanwin forms.
- Shadda, common long-alif/waw/ya forms, dagger alif and basic diphthongs.
- Common initial/prefixed articles and sun letters, Allah, relative pronouns,
  ihdinā and disjoint-letter names.
- Basic final short-vowel/tanwin removal and tāʾ marbūṭah in simple pause forms.
- Version/status/warnings in every result; unsupported characters/marks are
  flagged instead of being silently declared supported.

These are **not** proof of complete G2P coverage. Unvowelled/silent letters,
Maghrebi notation and contextual pronunciations can need lexical or route-aware
exceptions. A warning-free result is still `draft` and may be wrong.

## Explicitly not solved

- Complete Qālūn ṭarīq selection and allowable madd/hamza variants.
- Mīm al-jam and pronoun ṣilah; selected-route behavior, not Hafs defaults.
- Full nūn/tanwin assimilation, ikhfa, iqlab, nasalization, phonetic transitions.
- Madd timing, qalqalah, emphatic/throat articulation, stopping signs and
  arbitrary restart vowels. Latin spelling cannot represent acoustic quality.
- All Quran spelling exceptions, small hamza/ya forms, silent alifs, liaison
  shortening, and arbitrary Maghrebi signs. Ambiguities are flagged for review.
- IPA/acoustic targets, teacher acceptance thresholds and production approval.

Examples with nūn/tanwin keep orthographic segments and carry a warning; their
connected articulation may differ. Do not present them as exact spoken English.
The chosen symbols are a consistent aid, not an "English pronunciation" promise.

## Audit workflow

1. Appoint a qualified Qālūn teacher; record the exact source edition and ṭarīq.
2. Freeze generator version, source hash/version, audio provenance and context.
3. Review Fātiḥah and the five beginner lesson surahs first, then Juz ʿAmma.
4. Compare source text to an authoritative Qālūn mushaf **and** verified audio.
   Existing Hafs-mapped SVG artwork is not authoritative Qālūn spelling.
5. Fill `reviewer`, `approved_pause`, `approved_connected`, `audio_reference`
   and `status`. Prefer a second independent review for release examples.
6. Record approved variants instead of rejecting permissible route choices.
7. Correct the rule/exception with a reproducible fixture; regenerate only a
   separate draft sheet and reconcile it against preserved approved records.
8. Release a separately versioned, signed approval dataset; only reviewed
   entries should be labeled teacher-approved. This approval layer is **future
   work**, not implied by the current UI or an automated test passing.

The audit includes `hafs_reference_ids`, `hafs_reference_text` and
`hafs_rule_output_NOT_authority`. The last is our **same rule engine run on
Hafs Arabic**, not a certified external transliteration. Some mapped Hafs text
covers more content; compare corresponding spans, not raw whole-row equality.
Missing mappings remain empty—no verse numbering is guessed.

## Gate before public learner instruction

- Source + selected route approved by qualified reviewers.
- Zero unsupported/unresolved notation in entries released as reviewed.
- Per-word pause/connected outputs, accepted variants and reference clips checked.
- Fail safely on unreleased verses; no silent substitution of Hafs Latin text.
- Pedagogical testing with foreign Muslim readers: spelling must help, not teach
  an incorrect English sound. Begin Arabic-letter/vowel learning early.
- Keep explicit "recognition aid, not tajweed assessment" messaging.

Source consulted: https://github.com/risan/quran-json and this project's
QaloonData-derived source/cache. Permissions and scholarly validation are
separate obligations; a comparison table is not a license or certification.
