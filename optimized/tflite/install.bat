@echo off
rem SA3 TFLite installer for Windows — the .bat twin of install.sh.
rem
rem Mirrors install.sh's steps with plain stdlib tooling (no uv required):
rem   1. create a project-local .venv\ (python -m venv)
rem   2. pip install -r requirements.txt into it
rem   3. hand off to scripts\install.py for the weight-download prompt
rem
rem Extra args are forwarded to install.py, e.g.:
rem   install.bat --download sm-music
rem
rem Note: ai-edge-litert ships win_amd64 wheels for Python 3.10-3.13 —
rem use a Python in that range (3.11 recommended).
setlocal
set "SCRIPT_DIR=%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo error: python not found on PATH. 1>&2
    echo   Install Python 3.10-3.13 from https://www.python.org/downloads/ 1>&2
    echo   ^(check "Add python.exe to PATH" in the installer^) and re-run install.bat. 1>&2
    exit /b 1
)

if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
    echo Reusing existing .venv\
) else (
    echo Creating virtual environment at .venv\ ...
    python -m venv "%SCRIPT_DIR%.venv"
    if errorlevel 1 exit /b 1
)

echo Installing dependencies ^(pip install -r requirements.txt^) ...
"%SCRIPT_DIR%.venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
"%SCRIPT_DIR%.venv\Scripts\python.exe" -m pip install -r "%SCRIPT_DIR%requirements.txt"
if errorlevel 1 exit /b 1

rem install.py's pip step is skipped — deps were just installed above
rem (same contract install.sh uses).
set "INSTALL_SKIP_PIP=1"
"%SCRIPT_DIR%.venv\Scripts\python.exe" "%SCRIPT_DIR%scripts\install.py" %*
exit /b %ERRORLEVEL%
