@echo off
chcp 65001 >nul
echo ========================================
echo 유령 장치 제거 (GhostBuster)
echo ========================================
echo.

net session >nul 2>&1
if %errorLevel% neq 0 (
    echo 관리자 권한으로 실행해주세요!
    pause
    exit /b 1
)

echo [방법 1] 환경변수 설정 후 장치 관리자로 수동 제거
echo.
echo 1단계: 환경변수 설정
setx DEVMGR_SHOW_NONPRESENT_DEVICES 1 /M
set DEVMGR_SHOW_NONPRESENT_DEVICES=1

echo.
echo 2단계: 장치 관리자 열기...
echo.
echo ========================================
echo 다음 작업을 수행하세요:
echo ========================================
echo.
echo 1. 장치 관리자에서 "보기" → "숨겨진 장치 표시" 체크
echo.
echo 2. "마우스 및 기타 포인팅 장치" 확장
echo    - 회색/반투명으로 표시된 "HID 규격 마우스" 항목들 찾기
echo    - 각 항목 우클릭 → "속성" → "세부 정보" 탭
echo    - "하드웨어 ID"에서 VID_2341^&PID_8036 확인
echo    - 해당 항목 우클릭 → "제거" → "이 장치의 드라이버 삭제" 체크 → 확인
echo.
echo 3. "범용 직렬 버스 컨트롤러" 확장
echo    - 회색으로 표시된 "USB 복합 장치" 중 VID_2341 항목 모두 제거
echo.
echo 4. "포트 (COM ^& LPT)" 확장
echo    - 회색으로 표시된 "Arduino Leonardo (COM3)" 제거
echo    - 회색으로 표시된 "Arduino Leonardo (COM5)" 제거
echo.
echo ========================================
pause

start devmgmt.msc

echo.
echo 장치 관리자 작업을 완료한 후 아무 키나 누르세요...
pause >nul

echo.
echo [방법 2] PowerShell로 강제 제거 시도...
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$devicesToRemove = Get-PnpDevice | Where-Object { " ^
    "  ($_.InstanceId -like '*VID_2341*PID_8036*' -or " ^
    "   $_.InstanceId -like '*VID_046D*PID_C07D*') -and " ^
    "  $_.Status -eq 'Unknown' " ^
    "}; " ^
    "foreach ($device in $devicesToRemove) { " ^
    "  Write-Host 'Removing:' $device.FriendlyName '(' $device.InstanceId ')'; " ^
    "  $device | Disable-PnpDevice -Confirm:$false -ErrorAction SilentlyContinue; " ^
    "  Start-Sleep -Milliseconds 500; " ^
    "  $device | Uninstall-PnpDevice -Confirm:$false -ErrorAction SilentlyContinue; " ^
    "}"

echo.
echo [방법 3] 레지스트리 직접 삭제...
echo.

for /f "tokens=*" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Enum\HID" /s /f "VID_2341&PID_8036" ^| findstr "HKEY"') do (
    echo Deleting: %%a
    reg delete "%%a" /f >nul 2>&1
)

for /f "tokens=*" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Enum\HID" /s /f "VID_046D&PID_C07D" ^| findstr "HKEY"') do (
    echo Deleting: %%a
    reg delete "%%a" /f >nul 2>&1
)

echo.
echo [검증]
echo.
powershell -Command "Get-PnpDevice -Class Mouse | Format-Table Status,FriendlyName,InstanceId -AutoSize"

echo.
echo ========================================
echo 완료!
echo ========================================
echo.
echo 여전히 나타나면:
echo 1. PC 완전 종료 (재시작 아님!)
echo 2. 전원 끄기
echo 3. 30초 대기
echo 4. 전원 켜기
echo.
pause
