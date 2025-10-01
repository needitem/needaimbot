@echo off
chcp 65001 >nul
echo ========================================
echo Arduino USB VID/PID Spoofer
echo Logitech G502 HERO 완전 위장
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
set "USBCORE_CPP=C:\Users\th072\AppData\Local\Arduino15\packages\arduino\hardware\avr\1.8.6\cores\arduino\USBCore.cpp"

if not exist "%BOARDS_TXT%" (
    echo [ERROR] boards.txt를 찾을 수 없습니다!
    echo 경로: %BOARDS_TXT%
    pause
    exit /b 1
)

if not exist "%USBCORE_CPP%" (
    echo [ERROR] USBCore.cpp를 찾을 수 없습니다!
    echo 경로: %USBCORE_CPP%
    pause
    exit /b 1
)

echo [OK] Arduino 파일 찾음
echo.

:: 백업 생성
set "BOARDS_BACKUP=%BOARDS_TXT%.backup"
set "USBCORE_BACKUP=%USBCORE_CPP%.backup"

if not exist "%BOARDS_BACKUP%" (
    echo [INFO] boards.txt 백업 중...
    copy "%BOARDS_TXT%" "%BOARDS_BACKUP%" >nul
    echo [OK] 백업 완료
)

if not exist "%USBCORE_BACKUP%" (
    echo [INFO] USBCore.cpp 백업 중...
    copy "%USBCORE_CPP%" "%USBCORE_BACKUP%" >nul
    echo [OK] 백업 완료
)
echo.

echo [INFO] USB Descriptor 패치 중...
echo.
echo 변경 사항:
echo   VID: 0x046D (Logitech)
echo   PID: 0xC07D (G502 HERO)
echo   Upload VID/PID: 0x046D / 0xC07C, 0xC07D
echo   Manufacturer: "Logitech"
echo   Product: "G502 HERO Gaming Mouse"
echo.

:: boards.txt 패치
powershell -Command ^
    "$content = Get-Content '%BOARDS_TXT%' -Encoding UTF8; " ^
    "$changed = $false; " ^
    "$inLeonardo = $false; " ^
    "for ($i = 0; $i -lt $content.Count; $i++) { " ^
    "  if ($content[$i] -match '^leonardo\.name=') { $inLeonardo = $true; } " ^
    "  if ($inLeonardo -and $content[$i] -match '^[a-z]+\.name=') { $inLeonardo = $true; } " ^
    "  if ($inLeonardo) { " ^
    "    if ($content[$i] -match '^leonardo\.vid\.\d+=') { " ^
    "      $content[$i] = $content[$i] -replace '0x[0-9A-Fa-f]{4}', '0x046D'; " ^
    "      $changed = $true; " ^
    "    } " ^
    "    elseif ($content[$i] -match '^leonardo\.pid\.\d+=') { " ^
    "      if ($content[$i] -match '0x0036') { " ^
    "        $content[$i] = $content[$i] -replace '0x0036', '0xC07C'; " ^
    "      } elseif ($content[$i] -match '0x8036') { " ^
    "        $content[$i] = $content[$i] -replace '0x8036', '0xC07D'; " ^
    "      } " ^
    "      $changed = $true; " ^
    "    } " ^
    "    elseif ($content[$i] -match '^leonardo\.upload_port\.\d+\.vid=') { " ^
    "      $content[$i] = $content[$i] -replace '0x[0-9A-Fa-f]{4}', '0x046D'; " ^
    "      $changed = $true; " ^
    "    } " ^
    "    elseif ($content[$i] -match '^leonardo\.upload_port\.\d+\.pid=') { " ^
    "      if ($content[$i] -match '0x0036') { " ^
    "        $content[$i] = $content[$i] -replace '0x0036', '0xC07C'; " ^
    "      } elseif ($content[$i] -match '0x8036') { " ^
    "        $content[$i] = $content[$i] -replace '0x8036', '0xC07D'; " ^
    "      } " ^
    "      $changed = $true; " ^
    "    } " ^
    "    elseif ($content[$i] -match '^leonardo\.build\.vid=') { " ^
    "      $content[$i] = 'leonardo.build.vid=0x046D'; " ^
    "      $changed = $true; " ^
    "    } " ^
    "    elseif ($content[$i] -match '^leonardo\.build\.pid=') { " ^
    "      $content[$i] = 'leonardo.build.pid=0xC07D'; " ^
    "      $changed = $true; " ^
    "    } " ^
    "    elseif ($content[$i] -match '^leonardo\.build\.usb_manufacturer=') { " ^
    "      $content[$i] = 'leonardo.build.usb_manufacturer=\"\"Logitech\"\"'; " ^
    "      $changed = $true; " ^
    "    } " ^
    "    elseif ($content[$i] -match '^leonardo\.build\.usb_product=') { " ^
    "      $content[$i] = 'leonardo.build.usb_product=\"\"G502 HERO Gaming Mouse\"\"'; " ^
    "      $changed = $true; " ^
    "    } " ^
    "    elseif ($content[$i] -match '^leonardo\.build\.board=' -and -not ($content[$i] -match 'usb_manufacturer')) { " ^
    "      $content[$i] = 'leonardo.build.usb_manufacturer=\"\"Logitech\"\"' + [Environment]::NewLine + $content[$i]; " ^
    "      $changed = $true; " ^
    "    } " ^
    "  } " ^
    "} " ^
    "if ($changed) { " ^
    "  $content | Out-File -FilePath '%BOARDS_TXT%' -Encoding UTF8; " ^
    "  Write-Host '[OK] boards.txt 패치 완료!' -ForegroundColor Green; " ^
    "  exit 0; " ^
    "} else { " ^
    "  Write-Host '[ERROR] leonardo 설정을 찾을 수 없습니다!' -ForegroundColor Red; " ^
    "  exit 1; " ^
    "}"

if %errorLevel% neq 0 (
    echo [ERROR] boards.txt 패치 실패!
    pause
    exit /b 1
)

:: USBCore.cpp 패치
powershell -Command ^
    "$content = Get-Content '%USBCORE_CPP%' -Encoding UTF8; " ^
    "$changed = $false; " ^
    "$found046D = $false; " ^
    "for ($i = 0; $i -lt $content.Count; $i++) { " ^
    "  if ($content[$i] -match '#if\s+USB_VID\s*==\s*0x046D') { " ^
    "    $found046D = $true; " ^
    "    break; " ^
    "  } " ^
    "  if ($content[$i] -match '#if\s+USB_VID\s*==\s*0x2341') { " ^
    "    $newLines = @(); " ^
    "    $newLines += '#if USB_VID == 0x046D'; " ^
    "    $newLines += '#  if defined(USB_MANUFACTURER)'; " ^
    "    $newLines += '#    undef USB_MANUFACTURER'; " ^
    "    $newLines += '#  endif'; " ^
    "    $newLines += '#  define USB_MANUFACTURER \"\"Logitech\"\"'; " ^
    "    $newLines += '#elif USB_VID == 0x2341'; " ^
    "    $content[$i] = ($newLines -join [Environment]::NewLine); " ^
    "    $changed = $true; " ^
    "    break; " ^
    "  } " ^
    "} " ^
    "if ($found046D) { " ^
    "  Write-Host '[OK] USBCore.cpp 이미 패치됨' -ForegroundColor Green; " ^
    "  exit 0; " ^
    "} " ^
    "if ($changed) { " ^
    "  $content | Out-File -FilePath '%USBCORE_CPP%' -Encoding UTF8; " ^
    "  Write-Host '[OK] USBCore.cpp 패치 완료!' -ForegroundColor Green; " ^
    "  exit 0; " ^
    "} else { " ^
    "  Write-Host '[ERROR] USB_VID 정의를 찾을 수 없습니다!' -ForegroundColor Red; " ^
    "  exit 1; " ^
    "}"

if %errorLevel% neq 0 (
    echo [ERROR] USBCore.cpp 패치 실패!
    pause
    exit /b 1
)

echo.
echo ========================================
echo 패치 완료!
echo ========================================
echo.
echo [중요] 다음 단계:
echo.
echo 1. Arduino IDE 완전히 종료
echo.
echo 2. force_clean_compile.bat 실행 (캐시 삭제)
echo.
echo 3. Arduino IDE 재시작
echo.
echo 4. Tools - Board - Arduino Leonardo 선택
echo.
echo 5. Sketch - Verify/Compile (처음부터 컴파일)
echo.
echo 6. Upload
echo.
echo 7. Arduino USB 재연결
echo.
echo 8. check_mouse.bat 실행하여 확인
echo.
echo ========================================
echo.
echo [복원 방법]
echo    copy /Y "%BOARDS_BACKUP%" "%BOARDS_TXT%"
echo    copy /Y "%USBCORE_BACKUP%" "%USBCORE_CPP%"
echo.
pause
