@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto launch
echo First setup: Python 3.11 and internet are required.
py -3.11 -m venv .venv
if errorlevel 1 goto failed
:launch
if exist ".venv\demo-setup-complete" goto run
echo Installing dependencies. This can take several minutes and download several GB.
".venv\Scripts\python.exe" -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo ready>".venv\demo-setup-complete"
:run
echo Once the server is ready, open http://127.0.0.1:7860 in your browser.
echo Keep this window open. Press Ctrl+C to stop.
".venv\Scripts\python.exe" -u run_demo.py %*
if errorlevel 1 goto failed
exit /b 0
:failed
echo.
echo Setup or startup failed. Read the error above and TEST_AND_SHARE.md.
pause
exit /b 1
