# Rattil · رَتِّل — Project Idea | فكرة المشروع

## English

### Short description

Rattil is an AI-assisted Quran learning application focused on the Qālūn narration. It helps learners move from Arabic letters to Quran recitation through interactive lessons, reference audio, live word-by-word feedback, tajweed practice, and memorization challenges, with an English and Arabic interface.

### Medium-length description

Rattil brings Quran reading, recitation, and revision into one interactive learning experience, with a focus on the Qālūn narration. It supports beginners learning the Arabic alphabet as well as learners practising complete ayahs and surahs. The beginner course introduces letter sounds, connected forms, vowels, and short surahs through listening exercises and guided reading activities.

In the recitation studio, learners can recite into a microphone and follow feedback that matches the recognized words against the Quran text, highlighting progress and skipped words. Reference recitations help learners listen and repeat ayah by ayah. The application also provides colour-coded tajweed rules, explanations of articulation and Qālūn-specific rules, and a dedicated AI model that offers feedback on the tajweed patterns it hears. Voice-based ayah search lets users find a passage by reciting a few words, while quizzes and recall challenges support revision and memorization.

The project combines a Next.js, React, and TypeScript web interface with a Python FastAPI backend and Whisper models fine-tuned for Qālūn recitation. Its repository includes tools for Quran data preparation, model training, tajweed analysis, and learning challenges. In the local setup, recordings are processed on the user's computer. The central idea is to make guided Quran practice more accessible through bilingual learning materials, immediate feedback, and a clear progression from first letters to independent recitation.

## العربية

### الوصف المختصر

رَتِّل هو تطبيق لتعلّم قراءة القرآن الكريم وتلاوته بمساعدة الذكاء الاصطناعي، مع التركيز على رواية قالون عن نافع. يساعد المتعلّم على الانتقال من الحروف العربية إلى تلاوة الآيات والسور، من خلال دروس تفاعلية، وتلاوات مرجعية، ومتابعة مباشرة للكلمات أثناء القراءة، وتدريبات على التجويد، وتحديات للحفظ والمراجعة، بواجهة عربية وإنجليزية.

### الوصف متوسط الطول

يجمع رَتِّل بين تعلّم القراءة والتدرّب على التلاوة ومراجعة الحفظ في تجربة تفاعلية واحدة، مع التركيز على رواية قالون عن نافع. يخدم التطبيق المبتدئين الذين يتعلّمون الحروف العربية، وكذلك المتعلّمين الذين يتدرّبون على قراءة الآيات والسور كاملة. ويقدّم مسارًا تعليميًا يبدأ بأصوات الحروف وأشكالها المتصلة والحركات، ثم ينتقل إلى السور القصيرة عبر تمارين الاستماع وأنشطة القراءة الموجّهة.

في استوديو التلاوة، يستطيع المتعلّم القراءة باستخدام الميكروفون ومتابعة ملاحظات تقارن الكلمات التي يتعرّف عليها النموذج بالنص القرآني، وتُبرز تقدّمه والكلمات التي تجاوزها. كما يمكنه الاستماع إلى تلاوات مرجعية والتدرّب آيةً بآية. ويعرض التطبيق أحكام التجويد بألوان توضيحية، ويشرح مخارج الحروف والأحكام الخاصة برواية قالون، ويستخدم نموذج ذكاء اصطناعي مخصّصًا لتقديم ملاحظات حول أنماط التجويد التي يرصدها في الصوت. ويتيح البحث الصوتي العثور على موضع الآية بتلاوة بضع كلمات، بينما تساعد الاختبارات وتحديات استذكار الآيات على الحفظ والمراجعة.

تقنيًا، يعتمد المشروع على واجهة ويب مبنية باستخدام Next.js وReact وTypeScript، وخادم خلفي بلغة Python يعتمد على FastAPI ونماذج Whisper جرى ضبطها لتلاوات رواية قالون. ويضم المستودع أدوات لإعداد البيانات القرآنية، وتدريب النماذج، وتحليل أحكام التجويد، وبناء التحديات التعليمية. وعند تشغيله محليًا، تُعالَج التسجيلات على جهاز المستخدم. وتتمثل فكرة المشروع في تسهيل التدريب الموجّه على قراءة القرآن وتلاوته، من خلال محتوى تعليمي ثنائي اللغة، وملاحظات فورية، ومسار واضح يبدأ بالحروف الأولى ويصل إلى التلاوة المستقلة.

## Repository at a glance | نظرة على هيكل المشروع

| Path | English | العربية |
| --- | --- | --- |
| `web/` | Bilingual web interface, beginner lessons, recitation studio, tajweed pages, search, and challenges. | واجهة الويب ثنائية اللغة، ودروس المبتدئين، واستوديو التلاوة، وصفحات التجويد، والبحث، والتحديات. |
| `src/streaming/` | Speech recognition API, live recitation processing, word matching, and practice feedback. | واجهة التعرّف على الكلام، ومعالجة التلاوة المباشرة، ومطابقة الكلمات، وملاحظات التدريب. |
| `src/tajweed/` | Qālūn tajweed rules, text colouring, model labels, and evaluation tools. | أحكام التجويد لرواية قالون، وتلوين النص، ووسوم النماذج، وأدوات التقييم. |
| `src/learning/` | Learning challenge indexes and ayah similarity tools. | فهارس التحديات التعليمية وأدوات قياس التشابه بين الآيات. |
| `src/dataset_collection/`, `src/preprocess/` | Quran assets, dataset collection, and data preparation. | الموارد القرآنية، وجمع مجموعات البيانات، وتجهيزها. |
| `src/training/`, `src/training_with_gpu/`, `src/training_with_tajweed/` | Model training and evaluation workflows. | مسارات تدريب النماذج وتقييمها. |
| `src/deployment/` | Downloading and publishing model assets through Hugging Face. | تنزيل موارد النماذج ونشرها عبر Hugging Face. |
| `docs/` | Setup guides, design notes, and experiment results. | أدلة الإعداد، وملاحظات التصميم، ونتائج التجارب. |
| `run.py`, `run.sh`, `run.bat` | Automated local setup and application startup. | إعداد البيئة المحلية وتشغيل التطبيق تلقائيًا. |

For installation and usage, see [README.md](README.md). للتثبيت والاستخدام، راجع [README.md](README.md).
