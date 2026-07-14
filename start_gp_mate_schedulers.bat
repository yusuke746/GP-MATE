@echo off
setlocal

set "PS1=%~dp0start_gp_mate_schedulers.ps1"

if not exist "%PS1%" (
  echo Script not found: %PS1%
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1%"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo Failed with exit code %RC%
  exit /b %RC%
)

echo Done.
endlocal
