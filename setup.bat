@echo off
rem Whiskers V5: create the virtual environment, install the dependencies and run the self-check.
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Install Python 3.12 from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
    goto :fail
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating the virtual environment in .venv ...
    python -m venv .venv
    if errorlevel 1 goto :fail
)

echo Installing the dependencies (the CPU build of PyTorch is about 200 MB) ...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :fail

echo.
echo Self-check:
".venv\Scripts\python.exe" src\bot.py --selfcheck
if errorlevel 1 goto :fail

echo.
echo Done. In the RLBot launcher, add src\bot.toml and put Whiskers in a match.
pause
exit /b 0

:fail
echo.
echo Setup did not finish. See the message above.
pause
exit /b 1
