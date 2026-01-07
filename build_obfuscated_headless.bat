@echo off
setlocal

:: =============================================================================
:: 심볼 난독화 + HEADLESS 빌드 스크립트
:: 오버레이/GUI 없이 난독화된 바이너리 생성
:: =============================================================================

:: CUDA 환경변수 설정
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
set CUDAToolkit_ROOT=%CUDA_PATH%
set PATH=%CUDA_PATH%\bin;%LOCALAPPDATA%\Microsoft\WinGet\Links;%PATH%

:: 디렉토리 설정
set SOURCE_DIR=needaimbot
set OBF_SOURCE_DIR=needaimbot_obf
set BUILD_DIR=build_obf_headless

:: VS Build Tools 환경 설정
if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
) else (
    echo ERROR: Visual Studio Build Tools not found!
    pause
    exit /b 1
)

:: Python 확인
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python not found! Please install Python 3.x
    pause
    exit /b 1
)

:: =============================================================================
:: Step 1: 심볼 난독화
:: =============================================================================
echo.
echo [1/3] Obfuscating strings...
echo ============================================

:: 기존 난독화 소스 삭제
if exist %OBF_SOURCE_DIR% (
    rmdir /s /q %OBF_SOURCE_DIR%
)

:: Python 스크립트로 문자열 난독화 (안전한 버전)
python scripts/obfuscate_strings.py --source %SOURCE_DIR% --output %OBF_SOURCE_DIR%

if %ERRORLEVEL% neq 0 (
    echo Symbol obfuscation failed!
    pause
    exit /b 1
)

:: =============================================================================
:: Step 2: CMake 구성 (난독화 소스 + HEADLESS)
:: =============================================================================
echo.
echo [2/3] Configuring CMake for obfuscated HEADLESS build...
echo ============================================

:: 기존 빌드 디렉토리 삭제
if exist %BUILD_DIR% (
    rmdir /s /q %BUILD_DIR%
)

:: CMake 구성 (SRC_DIR로 난독화 소스 지정)
cmake -B %BUILD_DIR% -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DSRC_DIR=needaimbot_obf ^
    -DHEADLESS_BUILD=ON ^
    -DENABLE_OBFUSCATION=ON ^
    -DENABLE_ANTI_DEBUG=ON ^
    -DSTRIP_SYMBOLS=ON

if %ERRORLEVEL% neq 0 (
    echo CMake configuration failed!
    pause
    exit /b 1
)

:: =============================================================================
:: Step 3: 빌드
:: =============================================================================
echo.
echo [3/3] Building obfuscated HEADLESS binary...
echo ============================================

cmake --build %BUILD_DIR% --parallel

if %ERRORLEVEL% neq 0 (
    echo Build failed!
    pause
    exit /b 1
)

echo.
echo ============================================
echo OBFUSCATED HEADLESS BUILD SUCCESSFUL!
echo ============================================
echo.
echo Output: %BUILD_DIR%\bin\Release\NVDisplayContainer.exe
echo.
echo Applied:
echo   [x] HEADLESS mode (no overlay/GUI)
echo   [x] Sensitive strings replaced (YOLO, detection, aim, etc.)
echo   [x] Debug symbols stripped
echo   [x] Anti-debugging enabled
echo.
echo String map: %OBF_SOURCE_DIR%\_string_map.json
echo (Delete before distribution!)
echo.
pause
