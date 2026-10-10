@echo off
cd /d "%~dp0"
echo 正在启动智能旅行助手...
echo 启动后请在浏览器打开 http://127.0.0.1:8000
.venv\Scripts\python.exe run.py
pause
