@echo off
echo Starting Athena...
echo.
echo Open http://localhost:8000 in Chrome
echo.
cd /d "%~dp0athena"
C:/Users/kko8/AppData/Local/Programs/Python/Python311/python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
