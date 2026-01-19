@echo off
REM MakcuRelay Windows Build Script

setlocal

set BUILD_DIR=build
set BUILD_TYPE=Release

if not "%1"=="" set BUILD_TYPE=%1

echo Building MakcuRelay for Windows...
echo Build type: %BUILD_TYPE%

REM Create build directory
if not exist %BUILD_DIR% mkdir %BUILD_DIR%
cd %BUILD_DIR%

REM Configure with CMake
cmake -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ..

REM Build
cmake --build . --config %BUILD_TYPE%

echo.
echo Build complete!
echo Executable: %BUILD_DIR%\%BUILD_TYPE%\MakcuRelay.exe
echo.
echo Usage: MakcuRelay.exe [SERIAL_PORT] [UDP_PORT]
echo Example: MakcuRelay.exe COM3 5005

cd ..
