@echo off
chcp 65001 >nul
echo ========================================
echo Arduino 모든 USB 인터페이스 확인
echo ========================================
echo.

echo [모든 Arduino 관련 장치 (VID_046D^&PID_C07D)]
powershell -Command "Get-PnpDevice | Where-Object { $_.InstanceId -like '*VID_046D*PID_C07D*' } | Format-List FriendlyName,InstanceId,Status,Class"

echo.
echo ========================================
echo [모든 Arduino 관련 장치 (VID_2341^&PID_8036)]
powershell -Command "Get-PnpDevice | Where-Object { $_.InstanceId -like '*VID_2341*PID_8036*' } | Format-List FriendlyName,InstanceId,Status,Class"

echo.
pause
