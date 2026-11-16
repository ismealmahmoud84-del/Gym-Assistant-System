@echo off
setlocal

REM Gym Smart Assistant System - Tests Runner (Windows)

if not exist .venv\Scripts\activate (
  echo [ERROR] .venv not found. Create it first: python -m venv .venv
  exit /b 1
)

call .venv\Scripts\activate

echo [INFO] Running pytest...
python -m pytest

