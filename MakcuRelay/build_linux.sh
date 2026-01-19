#!/bin/bash

# MakcuRelay Linux Build Script

set -e

BUILD_DIR="build_linux"
BUILD_TYPE="${1:-Release}"

echo "Building MakcuRelay for Linux..."
echo "Build type: $BUILD_TYPE"

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" ..

# Build
cmake --build . --config "$BUILD_TYPE" -j$(nproc)

# Copy executable to root
cp MakcuRelay ..

echo ""
echo "Build complete!"
echo "Executable: ./MakcuRelay"
echo ""
echo "Usage: ./MakcuRelay [SERIAL_PORT] [UDP_PORT]"
echo "Example: ./MakcuRelay /dev/ttyUSB0 5005"
