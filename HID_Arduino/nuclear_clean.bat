@echo off
chcp 65001 >nul
echo ========================================
echo Arduino 흔적 완전 제거 (Nuclear Option)
echo ========================================
echo.

net session >nul 2>&1
if %errorLevel% neq 0 (
    echo 관리자 권한으로 실행해주세요!
    pause
    exit /b 1
)

echo 경고: 이 작업은 모든 Arduino 관련 드라이버와 레지스트리를 삭제합니다.
echo.
pause

echo.
echo [1/7] Arduino 확인...
powershell -Command "Get-PnpDevice | Where-Object { $_.InstanceId -like '*2341*8036*' -or $_.InstanceId -like '*2A03*' } | Format-Table FriendlyName,InstanceId,Status"

echo.
echo Arduino가 연결되어 있지 않은지 확인하세요!
pause

echo.
echo [2/7] SetupAPI 캐시 삭제...
del /F /Q "%SystemRoot%\inf\setupapi.dev.log" 2>nul
del /F /Q "%SystemRoot%\inf\setupapi.app.log" 2>nul

echo.
echo [3/7] USB 드라이버 정보 삭제 (VID_2341)...
pnputil /delete-driver oem*.inf /uninstall /force 2>nul | findstr /C:"2341"

echo.
echo [4/7] 레지스트리 완전 삭제...

REM USB 열거
reg delete "HKLM\SYSTEM\CurrentControlSet\Enum\USB\VID_2341&PID_8036" /f >nul 2>&1
reg delete "HKLM\SYSTEM\CurrentControlSet\Enum\USB\VID_2A03&PID_0043" /f >nul 2>&1

REM HID 열거
reg delete "HKLM\SYSTEM\CurrentControlSet\Enum\HID\VID_2341&PID_8036" /f >nul 2>&1

REM 장치 클래스
reg delete "HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e978-e325-11ce-bfc1-08002be10318}" /v "UpperFilters" /f >nul 2>&1

REM COM 포트 매핑
reg delete "HKLM\HARDWARE\DEVICEMAP\SERIALCOMM" /v "\Device\VCP0" /f >nul 2>&1

echo.
echo [5/7] PowerShell로 깊은 레지스트리 정리...
powershell -Command ^
    "$paths = @('HKLM:\SYSTEM\CurrentControlSet\Enum\USB', 'HKLM:\SYSTEM\CurrentControlSet\Enum\HID'); " ^
    "foreach ($path in $paths) { " ^
    "  Get-ChildItem $path -Recurse -ErrorAction SilentlyContinue | " ^
    "  Where-Object { $_.PSPath -match '2341|2A03' } | " ^
    "  Remove-Item -Recurse -Force -ErrorAction SilentlyContinue; " ^
    "}"

echo.
echo [6/7] DevNode 캐시 제거...
pnputil /scan-devices >nul 2>&1

echo.
echo [7/7] 완료!

echo.
echo ========================================
echo 검증
echo ========================================
echo.
powershell -Command "Get-PnpDevice | Where-Object { $_.InstanceId -like '*2341*' -or $_.InstanceId -like '*2A03*' } | Measure-Object | Select-Object -ExpandProperty Count" > "%TEMP%\arduino_count.txt"
set /p count=<"%TEMP%\arduino_count.txt"
del "%TEMP%\arduino_count.txt"

if "%count%"=="0" (
    echo [OK] Arduino 장치가 완전히 제거되었습니다!
) else (
    echo [WARNING] 아직 %count%개의 Arduino 관련 항목이 남아있습니다.
    echo 재부팅 후 다시 확인하세요.
)

echo.
echo 반드시 재부팅하세요!
echo.
set /p reboot="지금 재부팅? (Y/N): "
if /i "%reboot%"=="Y" shutdown /r /t 5
pause
