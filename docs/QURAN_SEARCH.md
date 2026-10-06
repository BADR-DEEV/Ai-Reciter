# Find an ayah: search the Quran by recitation

`/search` answers "which ayah is this?". You recite a few words, or play or upload a
recording, and the page shows the best place in the Quran. The next two places are
behind "Show more matches". A switch limits the search to the juzʾ you pick.

## How it works

| File | What it does |
| --- | --- |
| `src/streaming/server.py` → `POST /api/search` | Takes up to 30 s of 16 kHz PCM16 audio and an optional `juz` list. Transcribes the clip with the default (plain) model, without any expected text, then searches. |
| `src/streaming/search.py` | Indexes every word of the 6,210 Qālūn ayahs from `web/public/quran/surahs/` (~0.3 s, once). Strips a leading taʿawwudh and basmalah, then searches. Holds the juzʾ table. |
| `web/app/search/page.tsx`, `web/lib/search.ts` | The page: Shazam-style listen button, upload, juzʾ switch and picker, result cards. |
| `web/lib/juz.ts` | The juzʾ table for labels. A Python test keeps it identical to `search.py`. |

Search runs in two steps, seed then align:

1. Every heard word votes for places in the Quran where it appears, weighted by how
   rare it is. Votes that line up on the same diagonal mark a candidate place.
2. The heard words are aligned in order against up to 24 candidate places. The best 8
   are aligned again with partial credit for near spellings (يعلمون/تعلمون).
3. Evidence is weighted by how rare each word is. A place that shares only قل and الله
   with the clip is noise and is dropped (under 45% of the weighted words). One rare
   word such as وشاهد is enough to keep a place, and it wins ties against common words.
   Results never repeat an ayah.

When two places tie, a whole ayah beats the tail of a longer one. «فعّال لما يريد» is
85:16 first and the end of 11:107 second. A juzʾ scope makes words outside it
unmatchable, so every result lies inside the scope.

Confidence is about the words, not the voice. "Strong" means at least three heard words
and three quarters of what was heard line up, both by count and by rarity. A whole short
ayah with no close rival, such as «الله الصمد», also counts as strong.

The juzʾ starts use the standard Ḥafṣ boundaries mapped to Qālūn numbering through
`text-mapping.json` (for example juzʾ 2 starts at Al-Baqarah 141 in Qālūn).

## Accuracy (CPU, 2026-10-05)

| Test | rattil-v4 (default) | rattil-v3 |
| --- | --- | --- |
| 36 whole-ayah clips from 6 readers (Fātiḥah/Juz ʿAmma), top 1 | 36 | 36 |
| 13 three-second fragments of those clips, top 1 / top 3 | 10 / 10 | 9 / 10 |
| Al-Ikhlāṣ 1 searched only in juzʾ 1 (it is in juzʾ 30) | no match | no match |

The misses are a fragment that holds only the basmalah (the page asks for a few more
words) and fragments whose one distinctive word was misheard. For example, «إنه هو يبدل»
for 85:13 «إنه هو يبدئ» then fits many places equally.

On simulated transcripts (200 trials each), 8 heard words with 25% word errors give 97%
top 1 and 98% top 3; 4 words with 25% errors give 84% and 91%. A median request takes
0.18 s end to end, including Whisper on CPU. The search step alone takes under 80 ms.

`web/tests/live.spec.ts` checks a real microphone search when `RECITER_LIVE_TEST_AUDIO`
points at a Fātiḥah recording and the backend is running.

The readers were part of training, and the model was trained only on Al-Fātiḥah and
Juz ʿAmma. Elsewhere in the Quran the transcript is
less reliable, so the page labels those results as a guide.

## Limits

- It compares recognised words, not sound, so it cannot tell two identical wordings
  apart. The page says so when the same words appear in several places.
- One clip is at most 30 seconds (one Whisper window). Longer uploads are cut, and
  the page says so.
- A clip of only the basmalah or taʿawwudh has no place to find. The page asks for a
  few words of the ayah itself.
