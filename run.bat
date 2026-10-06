@echo off
rem Rattil: set up and run with one command on Windows. See README.md.
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" run.py %*
  goto done
)
for %%v in (3.12 3.11 3.13 3.10 3.14) do (
  py -%%v -c "pass" >nul 2>&1
  if not errorlevel 1 (
    py -%%v run.py %*
    goto done
  )
)
python -c "import sys; sys.exit(not (3, 10) <= sys.version_info[:2] <= (3, 14))" >nul 2>&1
if not errorlevel 1 (
  python run.py %*
  goto done
)
echo Rattil needs Python 3.10-3.13 (3.12 recommended): https://www.python.org/downloads/
echo In the installer, tick "Add python.exe to PATH", then run this again.
if not defined CI pause
exit /b 1
:done
set "code=%errorlevel%"
rem Keep the window open on errors when double-clicked (not in CI).
if not "%code%"=="0" if not defined CI pause
exit /b %code%
