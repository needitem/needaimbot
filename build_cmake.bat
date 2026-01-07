@echo off
setlocal

:: CUDA 환경변수 설정
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
set CUDAToolkit_ROOT=%CUDA_PATH%
set PATH=%CUDA_PATH%\bin;%LOCALAPPDATA%\Microsoft\WinGet\Links;%PATH%

:: 빌드 디렉토리
set BUILD_DIR=build

:: VS Build Tools 환경 설정 (컴파일러만 사용, IDE 불필요)
if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
) else (
    echo ERROR: Visual Studio Build Tools not found!
    echo Install from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
    pause
    exit /b 1
)

:: 빌드 디렉토리 없으면 CMake 구성
if not exist %BUILD_DIR%\build.ninja (
    echo [1/2] Configuring with CMake + Ninja...
    cmake -B %BUILD_DIR% -G Ninja -DCMAKE_BUILD_TYPE=Release
    if %ERRORLEVEL% neq 0 (
        echo CMake configuration failed!
        pause
        exit /b 1
    )
)

:: 빌드
echo Building...
cmake --build %BUILD_DIR% --parallel

if %ERRORLEVEL% neq 0 (
    echo Build failed!
    pause
    exit /b 1
)

echo.
echo Build successful!
echo Output: %BUILD_DIR%\GamePerformanceAnalyzer.exe
pause
