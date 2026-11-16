"""
Parse structured workout plan data from AI-generated Arabic text.
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

EXERCISE_PATTERN = re.compile(
    r"([A-Za-z\u0600-\u06FF][A-Za-z\u0600-\u06FF\s\-]+?)\s*[:\-]?\s*(\d+)\s*[x×X]\s*(\d+)",
    re.UNICODE,
)
REST_PATTERN = re.compile(r"راحة|استراحة|rest", re.IGNORECASE)
WEEK_PATTERN = re.compile(r"(?:أسبوع|week)\s*(\d+)", re.IGNORECASE)
DAY_PATTERN = re.compile(r"(?:يوم|day)\s*(\d+)", re.IGNORECASE)
JSON_BLOCK_PATTERN = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE
)
INLINE_JSON_PATTERN = re.compile(
    r"---JSON---\s*(\{.*?\})\s*(?:---|$)", re.DOTALL | re.IGNORECASE
)


def _default_plan_json(fitness_level="beginner"):
    return {
        "weeks": 4,
        "days_per_week": 5,
        "difficulty": fitness_level.lower(),
        "weekly_schedule": [],
        "exercises": [],
    }


def _normalize_exercise(name, sets, reps, rest_seconds=60):
    return {
        "name": name.strip(),
        "sets": int(sets),
        "reps": int(reps),
        "rest_seconds": rest_seconds,
    }


def _extract_json_block(text):
    for pattern in (JSON_BLOCK_PATTERN, INLINE_JSON_PATTERN):
        match = pattern.search(text)
        if not match:
            continue
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse workout plan JSON block: %s", exc)
    return None


def _parse_schedule_lines(text):
    schedule = []
    current_week = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        week_match = WEEK_PATTERN.search(line)
        if week_match:
            current_week = int(week_match.group(1))
            continue

        day_match = DAY_PATTERN.search(line)
        exercise_match = EXERCISE_PATTERN.search(line)

        if day_match or exercise_match or REST_PATTERN.search(line):
            entry = {
                "week": current_week or 1,
                "day": int(day_match.group(1)) if day_match else len(schedule) + 1,
                "label": line,
                "is_rest": bool(REST_PATTERN.search(line)),
            }
            if exercise_match:
                entry["exercise"] = _normalize_exercise(
                    exercise_match.group(1),
                    exercise_match.group(2),
                    exercise_match.group(3),
                )
            schedule.append(entry)

    return schedule


def parse_workout_plan_from_ai(text, fitness_level="beginner"):
    """
    Build structured plan JSON from AI text + optional embedded JSON block.
    """
    plan_json = _default_plan_json(fitness_level)
    if not text:
        return plan_json

    parsed_json = _extract_json_block(text)
    if isinstance(parsed_json, dict):
        plan_json.update({k: v for k, v in parsed_json.items() if v is not None})
        if isinstance(parsed_json.get("exercises"), list):
            plan_json["exercises"] = parsed_json["exercises"]
        if isinstance(parsed_json.get("weekly_schedule"), list):
            plan_json["weekly_schedule"] = parsed_json["weekly_schedule"]
            return plan_json

    exercises = []
    for match in EXERCISE_PATTERN.finditer(text):
        exercises.append(
            _normalize_exercise(match.group(1), match.group(2), match.group(3))
        )

    schedule = _parse_schedule_lines(text)

    if exercises:
        plan_json["exercises"] = exercises
    if schedule:
        plan_json["weekly_schedule"] = schedule
        plan_json["days_per_week"] = max(
            plan_json.get("days_per_week", 5),
            len([item for item in schedule if not item.get("is_rest")]),
        )

    weeks_found = {item.get("week", 1) for item in schedule}
    if weeks_found:
        plan_json["weeks"] = max(plan_json.get("weeks", 4), max(weeks_found))

    return plan_json
