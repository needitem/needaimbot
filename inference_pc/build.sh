#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$SCRIPT_DIR/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc)}"
CUDA_ARCHS="${CUDA_ARCHS:-72;87}"
NATIVE_OPT="${NATIVE_OPT:-ON}"
LTO="${LTO:-ON}"
GENERATOR="${GENERATOR:-}"
CLEAN=0

usage() {
    cat <<'EOF'
Usage: ./build.sh [options]

Options:
  --clean                  Remove build directory before configure
  --debug                  Build type Debug
  --release                Build type Release (default)
  --jobs N                 Parallel build jobs (default: nproc)
  --build-dir DIR          Build directory (default: ./build)
  --arch "A;B"             CUDA architectures (default: "72;87")
  --native ON|OFF          Enable host -march=native/-mtune=native (default: ON)
  --lto ON|OFF             Enable IPO/LTO when supported (default: ON)
  --generator NAME         Pass CMake generator (e.g. Ninja)
  -h, --help               Show this help

Environment variables (same meaning): BUILD_DIR, BUILD_TYPE, JOBS, CUDA_ARCHS,
NATIVE_OPT, LTO, GENERATOR
EOF
}

normalize_on_off() {
    local value="${1^^}"
    case "$value" in
        ON|OFF) echo "$value" ;;
        *)
            echo "Invalid value: $1 (expected ON or OFF)" >&2
            exit 1
            ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)
            CLEAN=1
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --release)
            BUILD_TYPE="Release"
            shift
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --arch)
            CUDA_ARCHS="$2"
            shift 2
            ;;
        --native)
            NATIVE_OPT="$(normalize_on_off "$2")"
            shift 2
            ;;
        --lto)
            LTO="$(normalize_on_off "$2")"
            shift 2
            ;;
        --generator)
            GENERATOR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

NATIVE_OPT="$(normalize_on_off "$NATIVE_OPT")"
LTO="$(normalize_on_off "$LTO")"

if [[ "$CLEAN" -eq 1 ]]; then
    rm -rf "$BUILD_DIR"
fi

cmake_args=(
    -S "$SCRIPT_DIR"
    -B "$BUILD_DIR"
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    "-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHS"
    "-DSIMPLE_ENABLE_NATIVE_OPT=$NATIVE_OPT"
    "-DSIMPLE_ENABLE_LTO=$LTO"
)

if [[ -n "$GENERATOR" ]]; then
    cmake_args+=(-G "$GENERATOR")
fi

echo "[build.sh] Configure"
echo "  BUILD_DIR=$BUILD_DIR"
echo "  BUILD_TYPE=$BUILD_TYPE"
echo "  JOBS=$JOBS"
echo "  CUDA_ARCHS=$CUDA_ARCHS"
echo "  CUDA_FAST_MATH=ON (fixed)"
echo "  NATIVE_OPT=$NATIVE_OPT"
echo "  LTO=$LTO"
if [[ -n "$GENERATOR" ]]; then
    echo "  GENERATOR=$GENERATOR"
fi

cmake "${cmake_args[@]}"
cmake --build "$BUILD_DIR" -j "$JOBS"

BIN_PATH="$BUILD_DIR/bin/Release/simple_inference"
echo "[build.sh] Done"
echo "[build.sh] Binary: $BIN_PATH"
