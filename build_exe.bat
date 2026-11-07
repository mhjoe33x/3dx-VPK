@echo off
REM Optional: one vpk_tool.exe that needs no Python (run on Windows).
cd /d "%~dp0"
py -3 -m pip install -r requirements.txt pyinstaller
py -3 -m PyInstaller --onefile --windowed --name vpk_tool vpk_tool.py
echo.
echo Built: dist\vpk_tool.exe
pause
