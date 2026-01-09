@echo off
setlocal

echo ========================================
echo   NVDisplayContainer Build
echo ========================================
echo.

if not exist "build" mkdir build

cmake -B build -DCMAKE_BUILD_TYPE=Release
if %errorlevel% neq 0 (
    echo CMake configuration failed!
    exit /b 1
)

cmake --build build --config Release -j%NUMBER_OF_PROCESSORS%
if %errorlevel% neq 0 (
    echo Build failed!
    exit /b 1
)

echo.
echo ========================================
echo   Build Complete!
echo ========================================
echo Output: build\bin\Release\NVDisplayContainer.exe
echo.

endlocal
