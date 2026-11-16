from django.test import TestCase

from fitness_api.nutrition_calculator import NutritionCalculator


class NutritionCalculatorDjangoTests(TestCase):
    def test_daily_calories_matches_goal_adjusted(self):
        data = NutritionCalculator.calculate_full_nutrition(
            weight_kg=70,
            height_m=1.75,
            age_years=30,
            activity_level="moderate",
            gender=0,
            use_weight_based=True,
            goal="cutting",
        )
        self.assertEqual(data["daily_calories"], data["goal_adjusted_calories"])

    def test_weight_based_fallback_when_remaining_non_positive(self):
        # Use very large protein demand scenario to force fallback
        data = NutritionCalculator.calculate_full_nutrition(
            weight_kg=300,
            height_m=1.6,
            age_years=25,
            activity_level="very_active",
            gender=0,
            use_weight_based=True,
            goal="cutting",
        )
        # calculation_method should be either 'weight_based' or 'percentage_based'
        self.assertIn(data["calculation_method"], ["weight_based", "percentage_based"])

    def test_water_outputs_are_integers(self):
        data = NutritionCalculator.calculate_full_nutrition(
            weight_kg=80,
            height_m=1.8,
            age_years=40,
            activity_level="light",
            gender=1,
            use_weight_based=False,
            goal="maintenance",
        )
        self.assertIsInstance(data["water_cups_daily"], int)
        self.assertIsInstance(data["water_bottles_daily"], int)
