@echo off
setlocal

REM Gym Smart Assistant System (Django) - Local Run Script (Windows)

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PARENT_DIR=%%~fI"
set "LOCAL_VENV=%SCRIPT_DIR%.venv\Scripts\activate.bat"
set "PARENT_VENV=%PARENT_DIR%\.venv\Scripts\activate.bat"

pushd "%SCRIPT_DIR%"

REM 1) Activate venv if one is not already active
if defined VIRTUAL_ENV (
  echo [INFO] Using active virtual environment: %VIRTUAL_ENV%
) else if exist "%LOCAL_VENV%" (
  call "%LOCAL_VENV%"
) else if exist "%PARENT_VENV%" (
  call "%PARENT_VENV%"
) else (
  echo [ERROR] No virtual environment found.
  echo [ERROR] Create one in this folder or the parent folder with: python -m venv .venv
  popd
  exit /b 1
)

REM 2) Install dependencies if needed (best-effort)
python -m pip show django >nul 2>nul
if errorlevel 1 (
  echo [INFO] Installing requirements...
  python -m pip install -r requirements.txt
)

REM 3) Ensure DB schema
python manage.py makemigrations --noinput
python manage.py migrate --noinput

REM 4) Start server
echo.
echo [INFO] Starting server on http://localhost:8000/
python manage.py runserver 8000

popd

