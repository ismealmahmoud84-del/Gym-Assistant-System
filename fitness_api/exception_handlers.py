"""
Custom DRF exception handler — user-friendly JSON errors with dev-only details.
"""

import logging
import traceback

from django.conf import settings
from django.http import JsonResponse
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import exception_handler

from .ml_exceptions import (
    InputValidationError,
    MLException,
    ModelLoadError,
    PredictionError,
)

logger = logging.getLogger(__name__)

USER_FRIENDLY_MESSAGES = {
    "INVALID_INPUT": "البيانات المُدخلة غير صحيحة. يرجى التحقق والمحاولة مجدداً.",
    "MODEL_LOAD_FAILED": "خدمة التحليل غير متاحة حالياً. يرجى المحاولة لاحقاً.",
    "PREDICTION_FAILED": "فشل التنبؤ. يرجى التحقق من البيانات والمحاولة مجدداً.",
    "ML_ERROR": "حدث خطأ في خدمة التحليل. يرجى المحاولة لاحقاً.",
    "DATA_MISMATCH": "تعارض في البيانات. يرجى إعادة الإدخال.",
    "PREPROCESSING_FAILED": "فشل معالجة البيانات. يرجى المحاولة مجدداً.",
    "AUTH_REQUIRED": "يجب تسجيل الدخول للوصول إلى هذه الخدمة.",
    "INVALID_DATE": "صيغة التاريخ غير صحيحة. استخدم YYYY-MM-DD.",
    "INTERNAL_ERROR": "حدث خطأ غير متوقع. يرجى المحاولة لاحقاً.",
    "NOT_FOUND": "المورد المطلوب غير موجود.",
    "PERMISSION_DENIED": "ليس لديك صلاحية للوصول إلى هذا المورد.",
    "VALIDATION_ERROR": "البيانات المرسلة غير صالحة.",
}


def _build_payload(code, message, details=None, http_status=None):
    payload = {
        "success": False,
        "error": message,
        "code": code,
    }
    if settings.DEBUG and details:
        payload["details"] = details
    return payload


def _ml_exception_status(exc):
    if isinstance(exc, ModelLoadError):
        return status.HTTP_503_SERVICE_UNAVAILABLE
    if isinstance(exc, (InputValidationError, PredictionError)):
        return status.HTTP_400_BAD_REQUEST
    return status.HTTP_500_INTERNAL_SERVER_ERROR


def custom_exception_handler(exc, context):
    """Return structured JSON for all API errors."""
    if isinstance(exc, MLException):
        code = exc.error_code
        message = (
            exc.message
            if settings.DEBUG
            else USER_FRIENDLY_MESSAGES.get(code, exc.message)
        )
        payload = _build_payload(
            code=code,
            message=message,
            details=str(exc.internal_error) if exc.internal_error else None,
        )
        logger.warning("ML exception [%s]: %s", code, exc.message)
        return Response(payload, status=_ml_exception_status(exc))

    response = exception_handler(exc, context)

    if response is not None:
        code = "VALIDATION_ERROR"
        if response.status_code == status.HTTP_401_UNAUTHORIZED:
            code = "AUTH_REQUIRED"
        elif response.status_code == status.HTTP_403_FORBIDDEN:
            code = "PERMISSION_DENIED"
        elif response.status_code == status.HTTP_404_NOT_FOUND:
            code = "NOT_FOUND"

        error_message = USER_FRIENDLY_MESSAGES.get(code, "حدث خطأ في الطلب.")
        details = response.data

        if isinstance(details, dict) and "detail" in details:
            if settings.DEBUG:
                error_message = str(details["detail"])
            details = details["detail"]
        elif settings.DEBUG and details:
            error_message = str(details)

        payload = _build_payload(
            code=code,
            message=error_message,
            details=details if settings.DEBUG else None,
        )
        response.data = payload
        if "success" not in response.data:
            response.data["success"] = False
        return response

    logger.error("Unhandled exception in API: %s", exc, exc_info=True)
    payload = _build_payload(
        code="INTERNAL_ERROR",
        message=(
            USER_FRIENDLY_MESSAGES["INTERNAL_ERROR"] if not settings.DEBUG else str(exc)
        ),
        details=traceback.format_exc() if settings.DEBUG else None,
    )
    return JsonResponse(payload, status=500)
