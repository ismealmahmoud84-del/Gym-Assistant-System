from django.urls import path

from .views import (
    DashboardChartDataView,
    FitnessHistoryView,
    FitnessPredictionView,
    GymChatbotView,
    WorkoutCaloriePredictorView,
    user_dashboard,
)
from .views_workoutplan import WorkoutPlanView

urlpatterns = [
    path("predict/", FitnessPredictionView.as_view(), name="fitness_predict"),
    path("chat/", GymChatbotView.as_view(), name="gym_chat"),
    path("dashboard/", user_dashboard, name="user_dashboard"),
    path("dashboard-data/", DashboardChartDataView.as_view(), name="dashboard_data"),
    path("workout-calories/", WorkoutCaloriePredictorView, name="workout_calories"),
    path("history/", FitnessHistoryView.as_view(), name="fitness_history"),
    path("workout-plan/", WorkoutPlanView.as_view(), name="workout_plan"),
]
