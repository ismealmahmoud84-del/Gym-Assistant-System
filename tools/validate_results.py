import os
import sys

import django

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "gym_config.settings")


def _setup_django() -> None:
    django.setup()


def _process_validation() -> None:
    from fitness_api.calorie_predictor import predict_calories
    from fitness_api.model_manager import ModelRegistry
    from fitness_api.nutrition_calculator import NutritionCalculator

    nutrition = NutritionCalculator.calculate_full_nutrition(
        weight_kg=75,
        height_m=1.75,
        age_years=30,
        activity_level="moderate",
        gender=0,
    )

    print("=== Nutrition (Male 75kg 175cm 30y moderate) ===")
    print(f"BMR: {nutrition['bmr']:.0f} kcal")
    print(f"Daily: {nutrition['daily_calories']} kcal")
    print(f"BMI: {nutrition['bmi']} ({nutrition['bmi_category']})")
    print(
        f"Protein: {nutrition['protein_grams']}g | "
        f"Carbs: {nutrition['carbs_grams']}g | "
        f"Fats: {nutrition['fats_grams']}g"
    )
    print(f"Water: {nutrition['water_with_exercise_liters']}L")

    print("\n=== Calorie predictions ===")
    for duration, heart_rate in [(20, 100), (30, 130), (60, 130), (45, 160)]:
        kcal = predict_calories(0, 30, 175, 75, duration, heart_rate)
        print(
            f"{duration}min @ {heart_rate}bpm -> {kcal} kcal "
            f"({kcal / duration:.1f} kcal/min)"
        )

    print("\n=== Fitness labels ===")
    print(ModelRegistry.get_fitness_labels())


if __name__ == "__main__":
    _setup_django()
    _process_validation()
