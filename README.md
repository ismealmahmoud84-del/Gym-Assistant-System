# Gym Smart Assistant System (Django)

## Project Overview
This is a Django-based web application for a fitness assistant. It includes:
- Authentication (Login/Signup)
- Dashboard with charts and history
- Fitness prediction (ML)
- Nutrition calculation
- AI chatbot (with automatic provider fallback)

---

## System Requirements
- Python 3.10+ (project uses `py314` settings; Python 3.10/3.11/3.12/3.13+ should work)
- Git (optional)
- Windows recommended (tested for local run)

---

## Setup (Local Development)
### 1) Create and activate a virtual environment
From the project root (where `manage.py` is located):

```bat
cd "FINAL Gym_Assistant_System\\FINAL Gym_Assistant_System\\Gym_Assistant_System"
python -m venv .venv
.venv\Scripts\activate
```

### 2) Install dependencies
```bat
pip install -r requirements.txt
```

### 3) Configure environment variables
A `.env` file is loaded automatically from the project folder.

Copy the example:
```bat
copy .env.example .env
```

See `env_note.txt` for reminders about required variables.


Open `.env` and set values you need. At minimum:
- `DJANGO_SECRET_KEY` (required for production; for local you can keep the default)
- `DJANGO_DEBUG` (default: True)
- `DJANGO_ALLOWED_HOSTS`
- `CSRF_TRUSTED_ORIGINS`

> Note: If your AI provider needs an API key (used by `fitness_api/ai_service.py`), ensure it is configured in `.env`.

---

## Database / Migrations
SQLite is used by default (`db.sqlite3`).

Run migrations:
```bat
python manage.py makemigrations
python manage.py migrate
```

(Optional) Create an admin user:
```bat
python manage.py createsuperuser
```

---

## Run the Development Server
```bat
python manage.py runserver 8000
```

Then open:
- Application: http://localhost:8000/
- Admin: http://localhost:8000/admin/

---

## API Endpoints (Quick List)
Base path: `/api/`

- `POST /api/predict/` (Fitness + calories + nutrition)
- `POST /api/chat/` (Gym chatbot)
- `GET /api/dashboard-data/`
- `GET /api/history/`
- `GET /api/workout-plan/`

---

## Nutrition calculator improvements

- The `nutrition_calculator.py` module is the single source of truth for BMR, TDEE,
	and macronutrient calculations.
- Recent improvements ensure macros are computed from the goal-adjusted calorie
	target (so `cutting` or `bulking` goals alter protein/carbs/fats correctly).
- The weight-based macro calculator now falls back to a safe percentage-based
	distribution if remaining calories after protein are insufficient.
- Water intake outputs (`cups`, `bottles_500ml`) are returned as integers.

Use `fitness_api.nutrition_calculator.NutritionCalculator.calculate_full_nutrition(...)`
to retrieve the full profile in other parts of the app.

---

## Running Tests
Tests can be executed via:
```bat
python -m pytest
```

A helper script exists:
- `run_tests.ps1`

---

## How to Deliver (Submission Checklist)
Before handing it in, ensure:
- Source code is included (no missing files)
- `requirements.txt` is present
- `.env.example` is present and instructions mention `.env`
- The app starts with the steps above

---

## Notes
- Templates are loaded from app folders (no custom `TEMPLATES[DIRS]`).
- Static files are served in development mode (`DEBUG=True`).

