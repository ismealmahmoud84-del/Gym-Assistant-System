import logging
import os

from django.utils.html import strip_tags
from rest_framework.response import Response
from rest_framework.views import APIView

from .ai_service import AIService, get_ai_service
from .models import UserFitnessData, WorkoutPlan
from .workout_plan_parser import parse_workout_plan_from_ai

logger = logging.getLogger(__name__)
GENAI_DEBUG_MODE = os.environ.get("GENAI_DEBUG_MODE", "false").lower() == "true"
WORKOUT_PLAN_FALLBACK = (
    "الخدمة الذكية غير متاحة حالياً. خطة تمارين عامة:\n\n"
    "أسبوع 1-4:\n"
    "يوم1: Full Body (Squat 3x12, Pushup 3x10, Pullup 3x8)\n"
    "يوم2: راحة\n"
    "يوم3: Upper Body\n"
    "يوم4: كارديو خفيف\n"
    "يوم5: Lower Body + Core\n"
    "يوم6: راحة\n"
    "يوم7: Mobility وتمدد"
)


def generate_workout_plan_text(prompt):
    if GENAI_DEBUG_MODE:
        return WORKOUT_PLAN_FALLBACK

    try:
        ai_service = get_ai_service()
        response = ai_service.get_response(prompt)
        if response == AIService.STATIC_FALLBACK_RESPONSE:
            return WORKOUT_PLAN_FALLBACK

        key_info = ai_service.get_last_used_key_info()
        logger.info(
            "Workout plan generated using %s key #%s",
            key_info["provider"],
            key_info["key_index"],
        )
        return response
    except Exception as e:
        logger.warning(
            "Failed to generate AI workout plan, using static fallback: %s", str(e)
        )
        return WORKOUT_PLAN_FALLBACK


class WorkoutPlanView(APIView):
    def post(self, request):
        if not request.user.is_authenticated:
            return Response({"error": "Authentication required."}, status=401)

        try:
            try:
                fitness_data = UserFitnessData.objects.filter(user=request.user).latest(
                    "created_at"
                )
            except UserFitnessData.DoesNotExist:
                return Response(
                    {"error": "No fitness data found. Complete analysis first."},
                    status=400,
                )

            fitness_level = fitness_data.fitness_level
            activity_level = fitness_data.activity_level
            daily_calories = fitness_data.daily_calories
            age = fitness_data.age
            gender = "ذكر" if fitness_data.gender == 0 else "أنثى"

            # Gemini prompt for Arabic workout plan with compact context.
            # NOTE: Do NOT include any JSON block at the end — user prefers a clean
            # Arabic text-only plan.
            prompt = (
                "أنت مدرب جيم محترف. أنشئ خطة تمارين مخصصة لمدة 4 أسابيع "
                "بصياغة عربية واضحة ومنسقة.\n"
                f"بيانات العميل الأساسية:\n"
                f"- العمر: {age}\n"
                f"- الجنس: {gender}\n"
                f"- مستوى اللياقة: {fitness_level}\n"
                f"- مستوى النشاط: {activity_level}\n"
                f"- سعرات يومية مستهدفة: {daily_calories}\n\n"
                "المطلوب باختصار:\n"
                "- 4 إلى 5 أيام تمرين أسبوعياً\n"
                "- تمارين قوة + كارديو + إحماء + تبريد\n"
                "- تقدم أسبوعي بسيط\n"
                "- نصائح أمان واستشفاء\n"
                "- الرد بالعربية فقط، منسقًا بشكل نظيف للقراءة البسيطة\n\n"
                "هيكل الرد المطلوب (بالترتيب):\n"
                "1) عنوان موجز للخطة\n"
                "2) نظرة عامة قصيرة (1-2 جملة)\n"
                "3) جدول أسبوعي منسق بالعربية\n"
                "   (أعد ترتيب الأسابيع والأيام بصيغة قابلة للقراءة)\n"
                "4) قائمة التمارين الرئيسية مع المجموعات والتكرارات لكل تمرين\n"
                "5) نصائح تغذية وراحة قصيرة\n\n"
                "مهم: لا تُدرج أي كتلة JSON أو ملخص JSON في نهاية الرد.\n"
                "لا تُدرج أي علامات برمجية (```json```) أو كتل بيانات قابلة لل\n"
                "البرمجة —\n"
                "فقط نص عربي منسق للعرض.\n"
                "الهدف هو إنتاج نص واضح وجميل يمكن عرضه مباشرة في واجهة "
                "المستخدم\n"
                "دون تعديل إضافي.\n"
            )

            plan_text = generate_workout_plan_text(prompt)
            # Ensure any HTML returned by AI is stripped to avoid XSS
            try:
                plan_text = strip_tags(plan_text)
            except Exception:
                # fallback: keep original text
                pass

            plan_json = parse_workout_plan_from_ai(
                plan_text, fitness_level=fitness_level
            )

            # Save plan
            plan = WorkoutPlan.objects.create(
                user=request.user,
                fitness_data=fitness_data,
                plan_title=f"خطة {fitness_level} مخصصة",
                plan_json=plan_json,
                plan_text=plan_text,
                difficulty_level=fitness_level.lower(),
                equipment_needed="جيم/منزل",
            )

            return Response(
                {
                    "status": "success",
                    "plan_id": plan.id,
                    "workout_plan": plan.plan_text,
                    "plan_json": plan.plan_json,
                    "title": plan.plan_title,
                    "difficulty": plan.difficulty_level,
                },
                content_type="application/json; charset=utf-8",
            )

        except Exception as e:
            logger.error("Workout plan generation failed: %s", str(e), exc_info=True)
            return Response(
                {"error": "Failed to generate workout plan."},
                status=500,
                content_type="application/json; charset=utf-8",
            )
