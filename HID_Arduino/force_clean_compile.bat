@echo off
chcp 65001 >nul
echo ========================================
echo Arduino 완전 클린 빌드
echo ========================================
echo.

echo [1/6] Arduino IDE 종료 중...
taskkill /F /IM arduino.exe 2>nul
taskkill /F /IM arduino-cli.exe 2>nul
taskkill /F /IM ArduinoIDE.exe 2>nul
timeout /t 2 >nul

echo [2/6] 빌드 캐시 삭제 중...
rmdir /S /Q "%LOCALAPPDATA%\Arduino15\.cache" 2>nul
rmdir /S /Q "%LOCALAPPDATA%\Temp\arduino" 2>nul
del /F /Q "%TEMP%\arduino*" 2>nul
for /d %%D in ("%TEMP%\arduino*") do rmdir /S /Q "%%D" 2>nul
for /d %%D in ("%TEMP%\arduino_build_*") do rmdir /S /Q "%%D" 2>nul
for /d %%D in ("%TEMP%\arduino_cache_*") do rmdir /S /Q "%%D" 2>nul

echo [3/6] 사용자 임시 파일 삭제 중...
rmdir /S /Q "%USERPROFILE%\AppData\Local\Temp\arduino_build_*" 2>nul
rmdir /S /Q "%USERPROFILE%\AppData\Local\Temp\arduino_cache_*" 2>nul

echo [4/6] Arduino 프로젝트 빌드 폴더 삭제 중...
rmdir /S /Q "%LOCALAPPDATA%\arduino\sketches" 2>nul
if exist "C:\Users\th072\Desktop\needaimbot\HID_Arduino\build" (
    rmdir /S /Q "C:\Users\th072\Desktop\needaimbot\HID_Arduino\build"
)

echo [5/6] 컴파일된 core 삭제 중...
rmdir /S /Q "%LOCALAPPDATA%\arduino\cores" 2>nul
echo [OK] Core 캐시 삭제 완료

echo [6/6] boards.txt 설정 확인 중...
findstr /C:"leonardo.build.vid=0x046D" "C:\Users\th072\AppData\Local\Arduino15\packages\arduino\hardware\avr\1.8.6\boards.txt" >nul
if %errorLevel% equ 0 (
    echo [OK] VID 설정: 0x046D
) else (
    echo [WARNING] VID 설정 확인 필요
)

findstr /C:"leonardo.build.pid=0xC07D" "C:\Users\th072\AppData\Local\Arduino15\packages\arduino\hardware\avr\1.8.6\boards.txt" >nul
if %errorLevel% equ 0 (
    echo [OK] PID 설정: 0xC07D
) else (
    echo [WARNING] PID 설정 확인 필요
)

findstr /C:"leonardo.build.usb_manufacturer" "C:\Users\th072\AppData\Local\Arduino15\packages\arduino\hardware\avr\1.8.6\boards.txt" >nul
if %errorLevel% equ 0 (
    echo [OK] Manufacturer 설정됨
) else (
    echo [WARNING] Manufacturer 설정 필요
)

echo.
echo ========================================
echo 클린 완료!
echo ========================================
echo.
echo 다음 단계:
echo.
echo 1. Arduino IDE 재시작
echo.
echo 2. Tools - Board - Arduino Leonardo 선택
echo.
echo 3. Sketch - Verify/Compile (처음부터 컴파일)
echo.
echo 4. Upload
echo.
echo 5. Arduino USB 재연결
echo.
echo 6. check_mouse.bat 실행
echo.
echo ========================================
pause
