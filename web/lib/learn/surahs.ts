// Short surahs taught in the final unit. Arabic is the project's Qālūn source
// text (QaloonData_v10). Historical Husary timing/Latin fields are retained as
// source notes only; playback now uses reciters.ts + validated local ayah clips,
// and visible phonetics come from qaloon-g2p.ts, never these old Latin fields.
// Meanings are short plain-English renderings to give context, not a
// scholarly translation.
export type LessonAyah = { ayah: number; text: string; translit: string; meaning: string; start: number; end: number };
export type LessonSurah = { id: number; name: string; arabic: string; meaning: string; audio: string; about: string; ayahs: LessonAyah[] };

const audio = (id: number) => `https://server13.mp3quran.net/husr/Rewayat-Qalon-A-n-Nafi/${String(id).padStart(3, "0")}.mp3`;

export const SURAHS: LessonSurah[] = [
  {
    id: 1, name: "Al-Fātiḥah", arabic: "الفاتحة", meaning: "The Opening", audio: audio(1),
    about: "The opening chapter of the Quran. Muslims recite it in every unit of every prayer, so it is the first surah most people learn.",
    ayahs: [
      { ayah: 1, text: "اِ۬لْحَمْدُ لِلهِ رَبِّ اِ۬لْعَٰلَمِينَ", translit: "al-ḥamdu lillāhi rabbi l-ʿālamīn", meaning: "All praise belongs to God, Lord of all the worlds,", start: 13140, end: 17840 },
      { ayah: 2, text: "اَ۬لرَّحْمَٰنِ اِ۬لرَّحِيمِ", translit: "ar-raḥmāni r-raḥīm", meaning: "the Most Compassionate, the Most Merciful,", start: 17840, end: 21820 },
      { ayah: 3, text: "مَلِكِ يَوْمِ اِ۬لدِّينِ", translit: "maliki yawmi d-dīn", meaning: "King of the Day of Judgement.", start: 21820, end: 25640 },
      { ayah: 4, text: "إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ", translit: "iyyāka naʿbudu wa iyyāka nastaʿīn", meaning: "You alone we worship, and You alone we ask for help.", start: 25640, end: 32479 },
      { ayah: 5, text: "اُ۪هْدِنَا اَ۬لصِّرَٰطَ اَ۬لْمُسْتَقِيمَ", translit: "ihdinā ṣ-ṣirāṭa l-mustaqīm", meaning: "Guide us along the straight path,", start: 32479, end: 37640 },
      { ayah: 6, text: "صِرَٰطَ اَ۬لذِينَ أَنْعَمْتَ عَلَيْهِمْ", translit: "ṣirāṭa lladhīna anʿamta ʿalayhim", meaning: "the path of those You have blessed,", start: 37640, end: 42760 },
      { ayah: 7, text: "غَيْرِ اِ۬لْمَغْضُوبِ عَلَيْهِمْ وَلَا اَ۬لضَّآلِّينَ", translit: "ghayri l-maghḍūbi ʿalayhim wa la ḍ-ḍāllīn", meaning: "not of those who earned anger, nor of those who went astray.", start: 42760, end: 51680 },
    ],
  },
  {
    id: 112, name: "Al-Ikhlāṣ", arabic: "الإخلاص", meaning: "Sincerity", audio: audio(112),
    about: "Four short ayahs describing the oneness of God. Short and full of the letters you have practised.",
    ayahs: [
      { ayah: 1, text: "قُلْ هُوَ اَ۬للَّهُ أَحَدٌ", translit: "qul huwa llāhu aḥad", meaning: "Say: He is God, the One;", start: 9240, end: 13482 },
      { ayah: 2, text: "اِ۬للَّهُ اُ۬لصَّمَدُ", translit: "allāhu ṣ-ṣamad", meaning: "God, the Eternal Refuge.", start: 13482, end: 17850 },
      { ayah: 3, text: "لَمْ يَلِدْ وَلَمْ يُولَدْ", translit: "lam yalid wa lam yūlad", meaning: "He neither begets, nor was He begotten,", start: 17850, end: 23226 },
      { ayah: 4, text: "وَلَمْ يَكُن لَّهُۥ كُفُؤاً أَحَدُۢ", translit: "wa lam yakul lahū kufuʾan aḥad", meaning: "and there is none comparable to Him.", start: 23226, end: 29337 },
    ],
  },
  {
    id: 108, name: "Al-Kawthar", arabic: "الكوثر", meaning: "Abundance", audio: audio(108),
    about: "The shortest surah in the Quran: three ayahs.",
    ayahs: [
      { ayah: 1, text: "إِنَّا أَعْطَيْنَٰكَ اَ۬لْكَوْثَرَ", translit: "innā aʿṭaynāka l-kawthar", meaning: "Indeed, We have given you abundance.", start: 8085, end: 14952 },
      { ayah: 2, text: "فَصَلِّ لِرَبِّكَ وَانْحَرْ", translit: "fa-ṣalli li-rabbika wa-nḥar", meaning: "So pray to your Lord and sacrifice.", start: 14952, end: 20853 },
      { ayah: 3, text: "إِنَّ شَانِئَكَ هُوَ اَ۬لْأَبْتَرُ", translit: "inna shāniʾaka huwa l-abtar", meaning: "Indeed, the one who hates you is the one cut off.", start: 20853, end: 26901 },
    ],
  },
  {
    id: 113, name: "Al-Falaq", arabic: "الفلق", meaning: "Daybreak", audio: audio(113),
    about: "A prayer for protection, often recited before sleep together with An-Nās.",
    ayahs: [
      { ayah: 1, text: "قُلْ أَعُوذُ بِرَبِّ اِ۬لْفَلَقِ", translit: "qul aʿūdhu bi-rabbi l-falaq", meaning: "Say: I seek refuge in the Lord of daybreak", start: 7780, end: 12820 },
      { ayah: 2, text: "مِن شَرِّ مَا خَلَقَ", translit: "min sharri mā khalaq", meaning: "from the evil of what He created,", start: 12820, end: 17500 },
      { ayah: 3, text: "وَمِن شَرِّ غَاسِقٍ إِذَا وَقَبَ", translit: "wa min sharri ghāsiqin idhā waqab", meaning: "and from the evil of darkness when it settles,", start: 17500, end: 24440 },
      { ayah: 4, text: "وَمِن شَرِّ اِ۬لنَّفَّٰثَٰتِ فِے اِ۬لْعُقَدِ", translit: "wa min sharri n-naffāthāti fi l-ʿuqad", meaning: "and from the evil of those who blow on knots,", start: 24440, end: 32920 },
      { ayah: 5, text: "وَمِن شَرِّ حَاسِدٍ إِذَا حَسَدَ", translit: "wa min sharri ḥāsidin idhā ḥasad", meaning: "and from the evil of an envier when he envies.", start: 32920, end: 39580 },
    ],
  },
  {
    id: 114, name: "An-Nās", arabic: "الناس", meaning: "Mankind", audio: audio(114),
    about: "The final surah of the Quran: a prayer for protection from whispered evil.",
    ayahs: [
      { ayah: 1, text: "قُلْ أَعُوذُ بِرَبِّ اِ۬لنَّاسِ", translit: "qul aʿūdhu bi-rabbi n-nās", meaning: "Say: I seek refuge in the Lord of people,", start: 8500, end: 14180 },
      { ayah: 2, text: "مَلِكِ اِ۬لنَّاسِ", translit: "maliki n-nās", meaning: "the King of people,", start: 14180, end: 18240 },
      { ayah: 3, text: "إِلَٰهِ اِ۬لنَّاسِ", translit: "ilāhi n-nās", meaning: "the God of people,", start: 18240, end: 22560 },
      { ayah: 4, text: "مِن شَرِّ اِ۬لْوَسْوَاسِ اِ۬لْخَنَّاسِ", translit: "min sharri l-waswāsi l-khannās", meaning: "from the evil of the whisperer who withdraws,", start: 22560, end: 30500 },
      { ayah: 5, text: "اِ۬لذِے يُوَسْوِسُ فِے صُدُورِ اِ۬لنَّاسِ", translit: "alladhī yuwaswisu fī ṣudūri n-nās", meaning: "who whispers into the hearts of people,", start: 30500, end: 39020 },
      { ayah: 6, text: "مِنَ اَ۬لْجِنَّةِ وَالنَّاسِ", translit: "mina l-jinnati wa n-nās", meaning: "from among jinn and people.", start: 39020, end: 45160 },
    ],
  },
];
