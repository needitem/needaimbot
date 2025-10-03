@echo off
chcp 65001 >nul
echo ========================================
echo Arduino 레지스트리 정리 (자동)
echo ========================================
echo.
echo 경고: 이 작업은 모든 Arduino 장치 정보를 삭제합니다.
echo       현재 연결된 Arduino도 재연결해야 합니다.
echo.
pause

echo.
echo [1/3] Arduino USB 뽑으세요...
pause

echo.
echo [2/3] Arduino 관련 레지스트리 항목 삭제 중...

REM USB 장치 정보 삭제
for /f "tokens=*" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Enum\USB" /s /f "VID_2341&PID_8036" ^| findstr "HKEY"') do (
    echo 삭제 중: %%a
    reg delete "%%a" /f >nul 2>&1
)

REM HID 장치 정보 삭제
for /f "tokens=*" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Enum\HID" /s /f "VID_2341&PID_8036" ^| findstr "HKEY"') do (
    echo 삭제 중: %%a
    reg delete "%%a" /f >nul 2>&1
)

echo.
echo [3/3] 완료!
echo.
echo 다음 단계:
echo 1. PC 재부팅 (권장)
echo 2. 스푸핑된 Arduino만 연결 (COM6)
echo 3. check_mouse.bat으로 확인
echo.
echo 이제 VID_2341^&PID_8036 장치는 나타나지 않아야 합니다.
echo.
pause
