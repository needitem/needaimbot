@echo off
chcp 65001 >nul
echo ========================================
echo Arduino USB Descriptor Spoofer v2
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

:: boards.txt 경로
set "BOARDS_TXT=C:\Users\th072\AppData\Local\Arduino15\packages\arduino\hardware\avr\1.8.6\boards.txt"

if not exist "%BOARDS_TXT%" (
    echo [ERROR] boards.txt를 찾을 수 없습니다!
    echo 경로: %BOARDS_TXT%
    pause
    exit /b 1
)

echo [OK] boards.txt 찾음: %BOARDS_TXT%
echo.

:: 백업 생성
set "BACKUP_PATH=%BOARDS_TXT%.backup"
if not exist "%BACKUP_PATH%" (
    echo [INFO] 원본 백업 중...
    copy "%BOARDS_TXT%" "%BACKUP_PATH%" >nul
    echo [OK] 백업 완료: %BACKUP_PATH%
) else (
    echo [INFO] 백업 파일이 이미 존재합니다.
)
echo.

echo [INFO] USB Descriptor 패치 중...
echo.
echo 변경 사항:
echo   VID: 0x046D (Logitech)
echo   PID: 0xC07D (G502 HERO)
echo   Product: "G502 HERO Gaming Mouse"
echo.

:: PowerShell로 파일 내용 수정
powershell -Command ^
    "$content = Get-Content '%BOARDS_TXT%' -Encoding UTF8; " ^
    "$changed = $false; " ^
    "for ($i = 0; $i -lt $content.Count; $i++) { " ^
    "  if ($content[$i] -match '^leonardo\.build\.vid=') { " ^
    "    $content[$i] = 'leonardo.build.vid=0x046D'; " ^
    "    Write-Host '[PATCH] Line ' ($i+1) ': leonardo.build.vid=0x046D'; " ^
    "    $changed = $true; " ^
    "  } " ^
    "  elseif ($content[$i] -match '^leonardo\.build\.pid=') { " ^
    "    $content[$i] = 'leonardo.build.pid=0xC07D'; " ^
    "    Write-Host '[PATCH] Line ' ($i+1) ': leonardo.build.pid=0xC07D'; " ^
    "    $changed = $true; " ^
    "  } " ^
    "  elseif ($content[$i] -match '^leonardo\.build\.usb_product=') { " ^
    "    $content[$i] = 'leonardo.build.usb_product=\"\"G502 HERO Gaming Mouse\"\"'; " ^
    "    Write-Host '[PATCH] Line ' ($i+1) ': leonardo.build.usb_product'; " ^
    "    $changed = $true; " ^
    "  } " ^
    "} " ^
    "if ($changed) { " ^
    "  $content | Out-File -FilePath '%BOARDS_TXT%' -Encoding UTF8; " ^
    "  Write-Host ''; " ^
    "  Write-Host '[OK] 패치 완료!' -ForegroundColor Green; " ^
    "  exit 0; " ^
    "} else { " ^
    "  Write-Host '[ERROR] leonardo 설정을 찾을 수 없습니다!' -ForegroundColor Red; " ^
    "  exit 1; " ^
    "}"

if %errorLevel% equ 0 (
    echo.
    echo ========================================
    echo 패치 성공!
    echo ========================================
    echo.
    echo [중요] 다음 단계:
    echo 1. Arduino IDE를 재시작하세요
    echo 2. Tools ^> Board ^> Arduino Leonardo 선택
    echo 3. HID_Arduino.ino를 다시 업로드하세요
    echo 4. Arduino를 USB에서 뽑았다가 다시 꽂으세요
    echo.
    echo [확인 방법]
    echo - Windows 장치 관리자에서 "G502 HERO Gaming Mouse" 확인
    echo - PowerShell: Get-PnpDevice -Class Mouse
    echo.
    echo [복원 방법]
    echo    copy /Y "%BACKUP_PATH%" "%BOARDS_TXT%"
    echo.
) else (
    echo.
    echo [ERROR] 패치 실패! 수동으로 수정하세요:
    echo 파일: %BOARDS_TXT%
    echo.
    echo 다음 줄을 찾아서 수정:
    echo   leonardo.build.vid=0x046D
    echo   leonardo.build.pid=0xC07D
    echo   leonardo.build.usb_product="G502 HERO Gaming Mouse"
    echo.
)

pause
