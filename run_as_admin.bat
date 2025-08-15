@echo off
echo Running NeedAimbot with Administrator privileges...
echo.
echo Make sure Arduino is connected to USB port
echo Close Arduino IDE if it's open
echo.
pause

cd /d "%~dp0"
powershell -Command "Start-Process -FilePath '.\x64\Release\needaimbot.exe' -Verb RunAs"