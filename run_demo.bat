@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 goto use_python
py -3.11 run_demo.py %*
goto finished
:use_python
python run_demo.py %*
:finished
set "DEMO_EXIT=%ERRORLEVEL%"
if not "%DEMO_EXIT%"=="0" echo Demo failed. See the error above and README.md.
if /I not "%DEMO_NO_PAUSE%"=="1" pause
exit /b %DEMO_EXIT%
