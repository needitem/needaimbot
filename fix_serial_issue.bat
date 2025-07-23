@echo off
echo 시리얼 포트 문제 해결 중...

echo 1. 실행 중인 needaimbot 프로세스 강제 종료...
taskkill /f /im ai.exe 2>nul
taskkill /f /im needaimbot.exe 2>nul

echo 2. COM 포트 상태 확인...
for /f "tokens=1" %%i in ('wmic path Win32_SerialPort get DeviceID /value ^| findstr "COM"') do (
    echo Found: %%i
)

echo 3. 디바이스 관리자에서 시리얼 포트 재시작...
echo - Windows + R 키를 누르고 devmgmt.msc 입력
echo - 포트(COM 및 LPT) 확장
echo - 해당 COM 포트 우클릭 > 사용 안 함 > 사용
echo.

echo 4. 시스템 USB 재시작 (권장)...
echo - 아두이노/시리얼 디바이스를 뽑았다가 다시 꽂으세요

echo 5. 임시 해결책 - 시스템 재시작
echo - 완전한 해결을 위해서는 재시작을 권장합니다

echo.
echo 예방 방법:
echo - 프로그램 종료시 반드시 정상 종료(X 버튼 클릭)
echo - Ctrl+C나 강제 종료 금지
echo - 시리얼 장치 연결 상태에서 프로그램 실행

pause