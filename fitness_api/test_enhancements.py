import json
from unittest.mock import MagicMock, patch

import numpy as np
from django.contrib.auth.models import User
from django.test import Client, TestCase

from .models import UserFitnessData
from .workout_plan_parser import parse_workout_plan_from_ai


class EnhancementTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username="enhanced", password="password123"
        )
        for day in range(1, 16):
            UserFitnessData.objects.create(
                user=self.user,
                gender=0,
                age=20 + day,
                weight=70 + day,
                height=1.75,
                fitness_level="متوسط" if day % 2 == 0 else "مبتدئ",
                activity_level="moderate",
            )

    @patch("fitness_api.views.predict_calories")
    @patch("fitness_api.views.get_fitness_model")
    def test_prediction_includes_bmi_fields(
        self, mock_get_fitness_model, mock_predict_calories
    ):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1])
        mock_get_fitness_model.return_value = mock_model
        mock_predict_calories.return_value = 320.0

        self.client.login(username="enhanced", password="password123")
        response = self.client.post(
            "/api/predict/",
            data=json.dumps(
                {
                    "age": 28,
                    "gender": 0,
                    "weight": 80,
                    "height": 1.8,
                }
            ),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("bmi", data)
        self.assertIn("bmi_category", data)
        self.assertIn("bmi_description", data)

    def test_history_pagination_and_filters(self):
        self.client.login(username="enhanced", password="password123")
        response = self.client.get(
            "/api/history/?page=1&page_size=5&fitness_level=متوسط"
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(len(data["history"]), 5)
        self.assertIn("pagination", data)
        self.assertEqual(data["pagination"]["page"], 1)
        self.assertEqual(data["pagination"]["page_size"], 5)
        self.assertGreaterEqual(data["pagination"]["total_count"], 5)

    @patch("fitness_api.views.get_ai_service")
    def test_chat_streaming_response(self, mock_get_ai_service):
        mock_service = MagicMock()

        def fake_stream(_prompt):
            yield "مرحباً "
            yield "بك"

        mock_service.stream_response.side_effect = fake_stream
        mock_service.get_last_used_key_info.return_value = {
            "provider": "gemini",
            "key_index": 1,
        }
        mock_get_ai_service.return_value = mock_service

        self.client.login(username="enhanced", password="password123")
        response = self.client.post(
            "/api/chat/",
            data=json.dumps({"message": "مرحبا", "stream": True}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response["Content-Type"])
        content = b"".join(response.streaming_content).decode("utf-8")
        self.assertIn("مرحباً", content)
        self.assertIn('"done": true', content)

    @patch("fitness_api.views.get_ai_service")
    def test_chat_response_utf8_json(self, mock_get_ai_service):
        mock_service = MagicMock()
        mock_service.get_response.return_value = "مرحباً بك"
        mock_get_ai_service.return_value = mock_service

        self.client.login(username="enhanced", password="password123")
        response = self.client.post(
            "/api/chat/",
            data=json.dumps({"message": "مرحبا", "stream": False}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("application/json", response["Content-Type"])
        self.assertIn("charset=utf-8", response["Content-Type"])
        self.assertEqual(response.json().get("assistant_response"), "مرحباً بك")

    def test_workout_plan_parser_extracts_json_and_exercises(self):
        text = """
        خطة 4 أسابيع
        يوم1: Squat 3x12
        يوم2: راحة
        ---JSON---
        {"weeks": 4, "days_per_week": 5, "exercises": [{"name": "Squat", "sets": 3, "reps": 12, "rest_seconds": 60}]}
        """
        plan = parse_workout_plan_from_ai(text, fitness_level="متوسط")
        self.assertEqual(plan["weeks"], 4)
        self.assertGreaterEqual(len(plan["exercises"]), 1)
        self.assertEqual(plan["exercises"][0]["name"], "Squat")
