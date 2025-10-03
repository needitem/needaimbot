@echo off
chcp 65001 >nul
echo ========================================
echo Arduino 레지스트리 강제 정리
echo ========================================
echo.

net session >nul 2>&1
if %errorLevel% neq 0 (
    echo 관리자 권한이 필요합니다.
    echo 이 파일을 우클릭 - "관리자 권한으로 실행"하세요.
    pause
    exit /b 1
)

echo [1/5] Arduino USB 모두 뽑았는지 확인...
echo.
echo 계속하려면 Arduino를 뽑고 아무 키나 누르세요...
pause >nul

echo.
echo [2/5] USB 장치 캐시 삭제 중...
pnputil /enum-devices /class USB /connected | findstr /C:"VID_2341" >nul
if %errorLevel% equ 0 (
    echo 경고: Arduino가 여전히 연결되어 있습니다!
    echo Arduino를 뽑고 다시 시도하세요.
    pause
    exit /b 1
)

echo.
echo [3/5] 레지스트리 백업 중...
reg export "HKLM\SYSTEM\CurrentControlSet\Enum\USB" "%TEMP%\usb_backup.reg" /y >nul 2>&1
reg export "HKLM\SYSTEM\CurrentControlSet\Enum\HID" "%TEMP%\hid_backup.reg" /y >nul 2>&1
echo 백업 완료: %TEMP%\usb_backup.reg

echo.
echo [4/5] Arduino 장치 정보 완전 삭제 중...

REM Method 1: Registry deletion
powershell -Command "Get-ChildItem 'HKLM:\SYSTEM\CurrentControlSet\Enum\USB' -Recurse | Where-Object { $_.Name -like '*VID_2341*PID_8036*' } | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
powershell -Command "Get-ChildItem 'HKLM:\SYSTEM\CurrentControlSet\Enum\HID' -Recurse | Where-Object { $_.Name -like '*VID_2341*PID_8036*' } | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"

REM Method 2: DevCon approach (fallback)
echo DevCon 방식으로 정리 중...
pnputil /remove-device "USB\VID_2341&PID_8036" /subtree /force >nul 2>&1
pnputil /remove-device "HID\VID_2341&PID_8036" /subtree /force >nul 2>&1

echo.
echo [5/5] 장치 관리자 캐시 초기화...
rundll32.exe setupapi,InstallHinfSection DefaultUninstall 132 %windir%\inf\usbstor.inf >nul 2>&1

echo.
echo ========================================
echo 정리 완료!
echo ========================================
echo.
echo 중요: 반드시 재부팅하세요!
echo.
echo 재부팅 후:
echo 1. 스푸핑된 Arduino만 연결
echo 2. check_arduino_interfaces.bat 실행
echo 3. VID_2341이 더 이상 나타나지 않는지 확인
echo.
set /p reboot="지금 재부팅하시겠습니까? (Y/N): "
if /i "%reboot%"=="Y" shutdown /r /t 10 /c "Arduino 레지스트리 정리 후 재부팅"
pause
