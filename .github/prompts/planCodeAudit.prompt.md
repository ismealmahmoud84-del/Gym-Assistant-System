## Plan: مراجعة كود شاملة (Code Audit)

TL;DR - مراجعة كاملة للمشروع مقارنةً بالـ PDF المرجعي، مع فحص معماري، جودة الكود، الأمن، وتكامل AI. سأشغّل اختبارات ومحللات ثابتة ثم أقدّم تقريرًا مع أهم 5 تحسينات.

**Steps**
1. Discovery: اجمع الملفات/مسارات المفتاحية وامسح كود `fitness_api`, `tools`, `models/`, وملفات الإعداد والاختبارات. *depends on step 2 (PDF access)*
2. Alignment: استخدم ملف الـ PDF المرجعي (المسار المزوّد) لمقارنة المتطلبات المعمارية والـ Appendices. أؤكد نطاقك (المستخدم اختار: كامل المستودع + تشغيل اختبارات وأدوات فحص).
3. Static checks: شغّل `pytest -q`, `ruff`, `black --check`, و`bandit`/`safety` إن أمكن لتجميع أخطاء سريعة وثغرات أمنية.
4. Manual code review: تفحص `models.py`, `views.py`, أي ملف `*service*.py` أو `ai_service.py`, `model_manager.py`, `calorie_predictor.py`, `nutrition_calculator.py`, وملفات التحضير `utils.py` و`scripts` في `tools/`.
5. ML-specific review: تحقق من تحميل الموديلات (ModelRegistry)، استخدام الـ scaler/features، تقييم طرق التحقق (MAE/RMSE/R2) الموجودة، وعمليات الـ sanitization لأخراج الـ AI.
6. Bug hunting & perf: فحص استخدام DataFrame (pandas) وعمليات in-memory التي قد تستهلك ذاكرة، والتأكد من عدم إعادة تحميل الموديل متكررًا.
7. Produce deliverables: (A) قائمة Mismatches مع المراجع للـ PDF، (B) قائمة أعلى 5 توصيات Refactor، (C) أي ثغرات أمنية حرجة، و(D) خطوات تحقق مقترحة + أوامر لتشغيل الفحوصات.

**Relevant files**
- `fitness_api/models.py` — بيانات قاعدة النظام، تأكد التجريد والـ ORM usage.
- `fitness_api/views.py` — منطق العرض وAPI endpoints.
- `fitness_api/ai_service.py` — تكامل Gemini/OpenRouter والتبديل (failover).
- `fitness_api/model_manager.py` — تحميل/كاش الموديلات.
- `fitness_api/calorie_predictor.py`, `fitness_api/nutrition_calculator.py` — منطق التنبؤ والحساب.
- `fitness_api/utils.py` — تحضير المدخلات للـ ML.
- `tools/train_fitness_model.py`, `tools/validate_results.py`, `tools/evaluate_calorie_model.py` — سكربتات التدريب والتقييم.
- `models/` (artifacts) — `calories_model_final.pkl`, `scaler_final.pkl`, `features_list.pkl`, `fitness_level_model.pkl`, `preprocessing_config.pkl`.
- `data/gym_members_exercise_tracking.cv.csv` — عينات التدريب/الاختبار.
- Reference PDF: `docs/Gym Smart Assistant System Book 2.pdf`

**Verification**
1. تشغيل الاختبارات: `pytest -q` ثم مراجعة الفailures.
2. تشغيل السكربت التقييمي: `python tools/evaluate_calorie_model.py` للتحقق من المقاييس (R2/MAE/RMSE).
3. شغل `ruff` و`black --check` للتنسيق والثبات، و`bandit -r fitness_api` لفحص الأمن السريع.
4. ابحث عن مفاتيح/سلاسل حساسة: `git grep -n "KEY\|SECRET\|TOKEN\|API_KEY"`.
5. تحقق من `ModelRegistry.get_model` لوجود caching/singleton، وتحقق أن الموديل لا يُعاد تحميله في كل طلب.
6. قارِن منطق failover في `ai_service.py` مع الوصف في الـ PDF Appendix.

**Decisions / Assumptions**
- سأعتمد الملف الـ PDF كمصدر حقيقي للمعمارية (المستخدم أكد الوصول).
- سأتبع نطاق "كامل المستودع" وتشغيل الاختبارات وأدوات الفحص (المستخدم اختار ذلك).
- لن أعدّل الكود في هذه المرحلة — هذه خطة المراجعة فقط.

**Further Considerations**
1. هل تريد تقريرًا مفصّلًا بصيغة PDF بعد المراجعة أم تلخيصًا نصيًا يكفي؟
2. هل ترغب أن أفتح Pull Requests للتغييرات المقترحة بعد الموافقة؟
