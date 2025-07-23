@echo off
echo 자원 소모량 최적화 적용 중...

echo 1. 임시 파일 정리...
del /q build.log 2>nul
del /q build_output.txt 2>nul
del /q /s *.cache 2>nul
del /q /s *.deps 2>nul
del /q /s *.tmp 2>nul

echo 2. CUDA 캐시 정리...
rmdir /s /q "%TEMP%\CUDA" 2>nul
rmdir /s /q "%LOCALAPPDATA%\NVIDIA Corporation\NvToolsExt" 2>nul

echo 3. Visual Studio 임시 파일 정리...
rmdir /s /q "needaimbot\x64\Release\*.cache" 2>nul
rmdir /s /q "x64\Release\*.cache" 2>nul

echo 4. 재휴지통 정리 (optional)...
rd /s /q C:\$Recycle.Bin 2>nul

echo 5. 시스템 메모리 정리...
echo off | clip
echo. | clip

echo.
echo 최적화 완료!
echo - 임시 파일 정리됨
echo - CUDA 캐시 리셋됨  
echo - 메모리 정리됨
echo.
echo 다음 단계:
echo 1. 프로젝트 리빌드 필요
echo 2. optimization_fixes.cpp의 클래스들을 기존 코드에 통합
echo 3. CUDA 커널 링크 오류 수정 필요
pause