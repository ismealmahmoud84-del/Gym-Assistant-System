"""
Utility functions for ML model preprocessing and data alignment.
Ensures consistent feature order, units, and data types across all endpoints.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def prepare_calorie_model_input(
    gender, age, height_cm, weight_kg, duration_min, heart_rate
):
    """
    Centralized function to prepare calorie model input for the new 6-feature model.
    **CRITICAL**: Ensures correct feature order expected by scaler/model pipeline.

    Args:
        gender: 0 for male, 1 for female (integer)
        age: age in years (integer)
        height_cm: height in centimeters (float)
        weight_kg: weight in kilograms (float)
        duration_min: duration in MINUTES (float) - NOT seconds!
        heart_rate: heart rate in bpm (integer)

    Returns:
        pd.DataFrame with correct columns for the new calorie model

    Example:
        >>> cal_input = prepare_calorie_model_input(
        ...     gender=0, age=30, height_cm=175, weight_kg=70,
        ...     duration_min=30, heart_rate=130
        ... )
        >>> # Height is in cm, duration is in minutes
        >>> prediction = calorie_model.predict(cal_input)
    """
    try:
        # Validate inputs
        if not isinstance(gender, (int, float)) or gender not in [0, 1]:
            raise ValueError(f"gender must be 0 (male) or 1 (female), got {gender}")

        if not isinstance(age, (int, float)) or age <= 0:
            raise ValueError(f"age must be positive, got {age}")

        if not isinstance(height_cm, (int, float)) or height_cm <= 0:
            raise ValueError(f"height_cm must be positive, got {height_cm}")

        if not isinstance(weight_kg, (int, float)) or weight_kg <= 0:
            raise ValueError(f"weight_kg must be positive, got {weight_kg}")

        if not isinstance(duration_min, (int, float)) or duration_min <= 0:
            raise ValueError(f"duration_min must be positive, got {duration_min}")

        if not isinstance(heart_rate, (int, float)) or heart_rate <= 0:
            raise ValueError(f"heart_rate must be positive, got {heart_rate}")

        # Create DataFrame with new 6-feature order expected by trained pipeline
        df = pd.DataFrame(
            [
                [
                    int(gender),
                    int(age),
                    float(height_cm),
                    float(weight_kg),
                    float(duration_min),
                    float(heart_rate),
                ]
            ],
            columns=["Gender", "Age", "Height", "Weight", "Duration", "Heart_Rate"],
        )

        logger.debug(
            "Calorie input prepared for new model: Height=%.1fcm, Duration=%.1fmin",
            df["Height"].iloc[0],
            df["Duration"].iloc[0],
        )
        return df

    except Exception as e:
        logger.error(f"Failed to prepare calorie model input: {str(e)}")
        raise


def prepare_fitness_model_input(
    age,
    gender,
    weight_kg,
    height_m,
    max_bpm,
    avg_bpm,
    resting_bpm,
    fat_percentage,
    water_intake,
    workout_frequency,
):
    """
    Prepare fitness level classification model input.

    IMPORTANT: The trained model expects these exact FEATURES:
      ['Age', 'Gender', 'Weight (kg)', 'Height (m)', 'BMI',
       'Resting_BPM', 'Workout_Frequency (days/week)',
       'Water_Intake (liters)', 'Fat_Percentage', 'Workout_Type_Code']

    Note:
    - max_bpm and avg_bpm are accepted for backward compatibility but are not part
      of the trained feature set.
    - workout_type_code is derived heuristically from workout_frequency because the
      endpoint provides only frequency.
    """
    try:
        age_i = int(age)
        gender_i = int(gender)
        weight_f = float(weight_kg)
        height_f = float(height_m)
        resting_bpm_i = int(resting_bpm)
        fat_pct_f = float(fat_percentage)
        water_f = float(water_intake)
        workout_freq_i = int(workout_frequency)

        if height_f <= 0:
            raise ValueError(f"height_m must be > 0, got {height_f}")
        if weight_f <= 0:
            raise ValueError(f"weight_kg must be > 0, got {weight_f}")

        # BMI = kg / m^2
        bmi_f = weight_f / (height_f**2)

        # Heuristic mapping to Workout_Type_Code.
        # Matches tools/train_fitness_model.py mapping.
        # WORKOUT_TYPE_MAPPING = {
        #     'Cardio': 0,
        #     'Strength': 1,
        #     'HIIT': 2,
        #     'Yoga': 3,
        #     'Mixed': 4,
        #     'Other': 5,
        # }
        # Use workout frequency days/week as a proxy for workout type.
        if workout_freq_i <= 1:
            workout_type_code = 5  # Other
        elif workout_freq_i <= 3:
            workout_type_code = 1  # Strength
        elif workout_freq_i <= 5:
            workout_type_code = 4  # Mixed
        else:
            workout_type_code = 2  # HIIT

        # Create DataFrame with the exact features and order used during training
        # Order: Age, Gender, Weight, Height, BMI, Resting_BPM,
        # Frequency, Water, Fat, Type
        df = pd.DataFrame(
            [
                [
                    age_i,
                    gender_i,
                    weight_f,
                    height_f,
                    bmi_f,
                    resting_bpm_i,
                    workout_freq_i,
                    water_f,
                    fat_pct_f,
                    workout_type_code,
                ]
            ],
            columns=[
                "Age",
                "Gender",
                "Weight (kg)",
                "Height (m)",
                "BMI",
                "Resting_BPM",
                "Workout_Frequency (days/week)",
                "Water_Intake (liters)",
                "Fat_Percentage",
                "Workout_Type_Code",
            ],
        )

        logger.debug(
            "Fitness input prepared successfully (model feature schema aligned)"
        )
        return df

    except Exception as e:
        logger.error(f"Failed to prepare fitness model input: {str(e)}")
        raise
