import logging

import numpy as np

from .model_manager import ModelRegistry
from .utils import prepare_calorie_model_input

logger = logging.getLogger(__name__)
CALORIE_MODEL_VERSION = "calories_model_final_v1"
CONFIDENCE_Z_SCORE = 1.96


def _calculate_calorie_sanity_bounds(weight_kg, duration_min):
    """Return realistic calorie burn bounds for a single workout."""
    duration_hours = float(duration_min) / 60.0
    min_kcal = 50.0
    max_kcal = float(weight_kg) * 15.0 * duration_hours
    return min_kcal, max_kcal


def _validate_inputs(gender, age, height_cm, weight_kg, duration, heart_rate):
    if gender not in [0, 1]:
        raise ValueError("Gender must be 0 (male) or 1 (female)")
    if not (10 <= age <= 100):
        raise ValueError("Age must be between 10 and 100")
    if not (100 <= height_cm <= 220):
        raise ValueError("Height must be between 100 and 220 cm")
    if not (30 <= weight_kg <= 200):
        raise ValueError("Weight must be between 30 and 200 kg")
    if not (1 <= duration <= 300):
        raise ValueError("Duration must be between 1 and 300 minutes")
    if not (40 <= heart_rate <= 220):
        raise ValueError("Heart rate must be between 40 and 220 bpm")

    max_hr = 220 - age
    if heart_rate > max_hr:
        raise ValueError(f"Heart rate {heart_rate} exceeds age-adjusted max {max_hr}")


def _estimate_confidence_range(
    model, input_scaled, prediction, weight_kg, duration_min
):
    """Estimate confidence interval using ensemble variance or physiological bounds."""
    min_calories, max_calories = _calculate_calorie_sanity_bounds(
        weight_kg, duration_min
    )
    method = "physiological_bounds"

    if hasattr(model, "estimators_"):
        tree_predictions = np.array(
            [float(est.predict(input_scaled)[0]) for est in model.estimators_]
        )
        std = float(np.std(tree_predictions))
        low = max(round(prediction - CONFIDENCE_Z_SCORE * std, 1), min_calories)
        high = min(round(prediction + CONFIDENCE_Z_SCORE * std, 1), max_calories)
        method = "ensemble_std"
    else:
        margin = max(prediction * 0.1, (max_calories - min_calories) * 0.15)
        low = max(round(prediction - margin, 1), min_calories)
        high = min(round(prediction + margin, 1), max_calories)

    return {
        "low": low,
        "high": high,
        "confidence_level": 0.95,
        "method": method,
    }


def predict_calories(gender, age, height_cm, weight_kg, duration, heart_rate):
    """
    Predict workout calories using the centralized calorie model pipeline.

    Args:
        gender: 0 = male, 1 = female
        age: age in years
        height_cm: height in centimeters
        weight_kg: weight in kilograms
        duration: workout duration in minutes
        heart_rate: average heart rate in bpm

    Returns:
        float: predicted calories burned
    """
    _validate_inputs(gender, age, height_cm, weight_kg, duration, heart_rate)

    input_data = prepare_calorie_model_input(
        gender=gender,
        age=age,
        height_cm=height_cm,
        weight_kg=weight_kg,
        duration_min=duration,
        heart_rate=heart_rate,
    )

    scaler = ModelRegistry.get_model("calorie_scaler")
    model = ModelRegistry.get_model("calorie")
    features = ModelRegistry.get_model("calorie_features")

    input_scaled = scaler.transform(input_data[features])
    prediction = float(model.predict(input_scaled)[0])

    min_calories, max_calories = _calculate_calorie_sanity_bounds(weight_kg, duration)
    if prediction < min_calories or prediction > max_calories:
        raise ValueError(
            f"النتيجة غير منطقية: {prediction:.0f} kcal لمدة {duration} دقيقة "
            f"(مسموح بين {min_calories:.0f} و {max_calories:.0f} kcal)"
        )

    logger.info(
        "Calorie prediction | version=%s | input=%s | output_kcal=%.1f",
        CALORIE_MODEL_VERSION,
        {
            "gender": gender,
            "age": age,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "duration_min": duration,
            "heart_rate": heart_rate,
        },
        prediction,
    )

    return round(prediction, 1)


def predict_calories_with_confidence(
    gender, age, height_cm, weight_kg, duration, heart_rate
):
    """
    Predict calories and return ML-informed confidence range.
    """
    _validate_inputs(gender, age, height_cm, weight_kg, duration, heart_rate)

    input_data = prepare_calorie_model_input(
        gender=gender,
        age=age,
        height_cm=height_cm,
        weight_kg=weight_kg,
        duration_min=duration,
        heart_rate=heart_rate,
    )

    scaler = ModelRegistry.get_model("calorie_scaler")
    model = ModelRegistry.get_model("calorie")
    features = ModelRegistry.get_model("calorie_features")

    input_scaled = scaler.transform(input_data[features])
    prediction = float(model.predict(input_scaled)[0])

    min_calories, max_calories = _calculate_calorie_sanity_bounds(weight_kg, duration)
    if prediction < min_calories or prediction > max_calories:
        raise ValueError(
            f"النتيجة غير منطقية: {prediction:.0f} kcal لمدة {duration} دقيقة "
            f"(مسموح بين {min_calories:.0f} و {max_calories:.0f} kcal)"
        )

    rounded_prediction = round(prediction, 1)
    confidence_range = _estimate_confidence_range(
        model, input_scaled, rounded_prediction, weight_kg, duration
    )

    logger.info(
        "Calorie prediction with confidence | version=%s | output_kcal=%.1f | range=%s",
        CALORIE_MODEL_VERSION,
        rounded_prediction,
        confidence_range,
    )

    return {
        "prediction": rounded_prediction,
        "confidence_range": confidence_range,
    }
