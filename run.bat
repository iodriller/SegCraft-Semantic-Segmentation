@echo off
title SegCraft
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
if errorlevel 1 (
  echo.
  echo SegCraft did not start. Review the error above.
  pause
)
