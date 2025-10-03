@echo off
chcp 65001 >nul
echo ========================================
echo Arduino 마우스 VID/PID 확인
echo ========================================
echo.

echo [연결된 모든 마우스 장치]
echo.
powershell -Command "Get-PnpDevice -Class Mouse | Format-Table -AutoSize"

echo.
echo ========================================
echo [상세 정보 - VID/PID 포함]
echo ========================================
echo.

powershell -Command ^
    "Get-PnpDevice -Class Mouse | ForEach-Object { " ^
    "  $device = $_; " ^
    "  $hardwareId = (Get-PnpDeviceProperty -InstanceId $device.InstanceId -KeyName 'DEVPKEY_Device_HardwareIds').Data[0]; " ^
    "  Write-Host '장치명:' $device.FriendlyName; " ^
    "  Write-Host 'HardwareID:' $hardwareId; " ^
    "  if ($hardwareId -match 'VID_([0-9A-F]{4})') { Write-Host 'VID:' $matches[1] }; " ^
    "  if ($hardwareId -match 'PID_([0-9A-F]{4})') { Write-Host 'PID:' $matches[1] }; " ^
    "  Write-Host ''; " ^
    "}"

echo.
echo ========================================
echo 확인 방법:
echo ========================================
echo.
echo Arduino가 Logitech으로 위장되었다면:
echo   VID: 046D (Logitech)
echo   PID: C07D (G502 HERO)
echo.
echo 위장되지 않았다면:
echo   VID: 2341 또는 2A03 (Arduino)
echo   PID: 8036 또는 0036
echo.
pause
