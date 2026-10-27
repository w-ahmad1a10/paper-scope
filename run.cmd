@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHON=%CD%\.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo.
    echo The local Python environment is missing.
    echo Follow the setup section in README.md, then run this file again.
    echo.
    pause
    exit /b 1
)

"%PYTHON%" -m paper_scope %*
set "CODE=%ERRORLEVEL%"

if not "%~1"=="" exit /b %CODE%

echo.
pause
exit /b %CODE%
