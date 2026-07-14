@echo off
rem Windows twin of the `sa3` bash wrapper — runs scripts\sa3_tflite.py via the
rem project-local .venv (created by install.bat) so no activation is needed.
rem Falls back to whatever `python` is on PATH if .venv\ doesn't exist yet.
rem
rem Usage: sa3.bat --prompt "lofi house" --dit sm-music --decoder same-s --out a.wav
setlocal
set "SCRIPT_DIR=%~dp0"

if not exist "%SCRIPT_DIR%scripts\sa3_tflite.py" (
    echo error: scripts\sa3_tflite.py not found - repo files are missing or moved. 1>&2
    exit /b 1
)

set "PY=%SCRIPT_DIR%.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo warning: .venv\ not found - run install.bat for one-time setup. Trying `python` from PATH. 1>&2
    set "PY=python"
)

"%PY%" "%SCRIPT_DIR%scripts\sa3_tflite.py" %*
exit /b %ERRORLEVEL%
