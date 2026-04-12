@echo off
setlocal
cd /d "%~dp0"

where powershell >nul 2>nul
if errorlevel 1 (
  echo [ERROR] PowerShell not found on PATH.
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_all.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
  echo.
  echo [ERROR] run_all.ps1 failed with exit code %EXIT_CODE%.
)

exit /b %EXIT_CODE%
