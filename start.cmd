@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 goto use_python
py -3 bootstrap.py run %*
goto finished

:use_python
where python >nul 2>nul
if errorlevel 1 goto no_python
python bootstrap.py run %*
goto finished

:no_python
echo Python 3.11 or newer was not found.
echo Install Python from https://www.python.org/downloads/ and try again.
set "RUN_RESULT=2"
goto pause_and_exit

:finished
set "RUN_RESULT=%ERRORLEVEL%"
if "%RUN_RESULT%"=="0" goto exit_now

:pause_and_exit
echo.
pause

:exit_now
exit /b %RUN_RESULT%
