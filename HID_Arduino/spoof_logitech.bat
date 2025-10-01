@echo off
chcp 65001 >nul
echo ========================================
echo Arduino USB Descriptor Spoofer
echo Logitech G502 HERO 위장 패치
echo ========================================
echo.

:: 관리자 권한 확인
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] 관리자 권한이 필요합니다!
    echo 이 배치파일을 우클릭 후 "관리자 권한으로 실행"을 선택하세요.
    pause
    exit /b 1
)

:: Arduino 경로 찾기
set "ARDUINO_PATH="
set "USBCORE_PATH="

:: 일반적인 Arduino 경로들 검색
echo [INFO] Arduino 설치 경로를 찾는 중...

:: 방법 1: AppData (Arduino IDE 1.6+)
for /d %%D in ("%LOCALAPPDATA%\Arduino15\packages\arduino\hardware\avr\*") do (
    if exist "%%D\cores\arduino\USBCore.h" (
        set "USBCORE_PATH=%%D\cores\arduino\USBCore.h"
        set "ARDUINO_PATH=%%D"
        goto :found
    )
)

:: 방법 2: Program Files
if exist "C:\Program Files (x86)\Arduino\hardware\arduino\avr\cores\arduino\USBCore.h" (
    set "USBCORE_PATH=C:\Program Files (x86)\Arduino\hardware\arduino\avr\cores\arduino\USBCore.h"
    set "ARDUINO_PATH=C:\Program Files (x86)\Arduino\hardware\arduino\avr"
    goto :found
)

if exist "C:\Program Files\Arduino\hardware\arduino\avr\cores\arduino\USBCore.h" (
    set "USBCORE_PATH=C:\Program Files\Arduino\hardware\arduino\avr\cores\arduino\USBCore.h"
    set "ARDUINO_PATH=C:\Program Files\Arduino\hardware\arduino\avr"
    goto :found
)

:: 찾지 못함
echo [ERROR] USBCore.h 파일을 찾을 수 없습니다!
echo.
echo 수동으로 경로를 입력하세요:
echo 예: C:\Program Files (x86)\Arduino\hardware\arduino\avr\cores\arduino\USBCore.h
echo.
set /p "USBCORE_PATH=USBCore.h 전체 경로: "

if not exist "%USBCORE_PATH%" (
    echo [ERROR] 파일이 존재하지 않습니다: %USBCORE_PATH%
    pause
    exit /b 1
)

:found
echo [OK] USBCore.h 찾음: %USBCORE_PATH%
echo.

:: 백업 생성
set "BACKUP_PATH=%USBCORE_PATH%.backup"
if not exist "%BACKUP_PATH%" (
    echo [INFO] 원본 백업 중...
    copy "%USBCORE_PATH%" "%BACKUP_PATH%" >nul
    echo [OK] 백업 완료: %BACKUP_PATH%
) else (
    echo [INFO] 백업 파일이 이미 존재합니다.
)
echo.

:: 임시 파일 생성
set "TEMP_FILE=%TEMP%\USBCore_patched.h"

echo [INFO] USB Descriptor 패치 중...
echo.
echo 변경 사항:
echo   VID: 0x046D (Logitech)
echo   PID: 0xC07D (G502 HERO)
echo   Manufacturer: Logitech
echo   Product: G502 HERO Gaming Mouse
echo.

:: PowerShell로 파일 내용 수정
powershell -Command ^
    "$content = Get-Content '%USBCORE_PATH%' -Encoding UTF8; " ^
    "$changed = $false; " ^
    "for ($i = 0; $i -lt $content.Count; $i++) { " ^
    "  if ($content[$i] -match '^\s*#\s*define\s+USB_VID\s+') { " ^
    "    $content[$i] = '#define USB_VID 0x046D  // Logitech (SPOOFED)'; " ^
    "    $changed = $true; " ^
    "  } " ^
    "  elseif ($content[$i] -match '^\s*#\s*define\s+USB_PID\s+') { " ^
    "    $content[$i] = '#define USB_PID 0xC07D  // G502 HERO (SPOOFED)'; " ^
    "    $changed = $true; " ^
    "  } " ^
    "  elseif ($content[$i] -match '^\s*#\s*define\s+USB_MANUFACTURER\s+') { " ^
    "    $content[$i] = '#define USB_MANUFACTURER \"\"Logitech\"\"  // (SPOOFED)'; " ^
    "    $changed = $true; " ^
    "  } " ^
    "  elseif ($content[$i] -match '^\s*#\s*define\s+USB_PRODUCT\s+') { " ^
    "    $content[$i] = '#define USB_PRODUCT \"\"G502 HERO Gaming Mouse\"\"  // (SPOOFED)'; " ^
    "    $changed = $true; " ^
    "  } " ^
    "} " ^
    "if ($changed) { " ^
    "  $content | Out-File -FilePath '%TEMP_FILE%' -Encoding UTF8; " ^
    "  exit 0; " ^
    "} else { " ^
    "  exit 1; " ^
    "}"

if %errorLevel% equ 0 (
    :: 패치된 파일로 교체
    copy /Y "%TEMP_FILE%" "%USBCORE_PATH%" >nul
    del "%TEMP_FILE%" >nul
    echo [OK] 패치 완료!
) else (
    echo [WARNING] #define 문을 찾을 수 없습니다.
    echo USBCore.h 파일 형식이 예상과 다를 수 있습니다.
    echo.
    echo 수동 패치가 필요합니다:
    echo 1. 파일 열기: %USBCORE_PATH%
    echo 2. 다음 값들을 찾아서 수정:
    echo    #define USB_VID 0x046D
    echo    #define USB_PID 0xC07D
    echo    #define USB_MANUFACTURER "Logitech"
    echo    #define USB_PRODUCT "G502 HERO Gaming Mouse"
)

echo.
echo ========================================
echo 패치 완료!
echo ========================================
echo.
echo [중요] 다음 단계:
echo 1. Arduino IDE를 재시작하세요
echo 2. HID_Arduino.ino를 다시 업로드하세요
echo 3. Arduino를 USB에서 뽑았다가 다시 꽂으세요
echo 4. Windows 장치 관리자에서 "Logitech G502 HERO"로 인식되는지 확인
echo.
echo [복원] 원래대로 되돌리려면:
echo    %BACKUP_PATH%
echo    위 파일을 USBCore.h로 복사하세요
echo.
pause
