cmake_minimum_required(VERSION 3.16)

# =============================================================================
# Platform Detection and CUDA Configuration
# =============================================================================
# Linux/Jetson: Use system CUDA
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
set(CUDAToolkit_ROOT "${CUDA_TOOLKIT_ROOT_DIR}")

project(SimpleInference LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Build options
option(SIMPLE_ENABLE_NATIVE_OPT "Enable -march=native for host compiler" OFF)
option(SIMPLE_ENABLE_LTO "Enable interprocedural optimization (IPO/LTO)" OFF)

# =============================================================================
# Path Configuration (Linux/Jetson)
# =============================================================================
set(CUDA_PATH "/usr/local/cuda")
set(TENSORRT_INCLUDE_DIR "/usr/include/aarch64-linux-gnu")
set(TENSORRT_LIB_DIR "/usr/lib/aarch64-linux-gnu")

# =============================================================================
# CUDA Configuration
# =============================================================================
find_package(CUDAToolkit REQUIRED)

# Jetson Xavier NX / AGX: SM 7.2, Jetson Orin: SM 8.7
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 72 87)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --diag-suppress=611,221,20091,20054")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCUB_IGNORE_DEPRECATED_CPP_DIALECT")

# =============================================================================
# Simple Aimbot (2PC: UDP capture + TensorRT + Makcu)
# =============================================================================
add_executable(simple_inference
    simple_main.cpp
    needaimbot/cuda/simple_inference.cu
    needaimbot/cuda/simple_postprocess.cu
    needaimbot/capture/udp_capture.cpp
    needaimbot/mouse/input_drivers/MakcuConnection.cpp
)

target_include_directories(simple_inference PRIVATE
    ${CMAKE_SOURCE_DIR}/needaimbot
    ${CMAKE_SOURCE_DIR}/needaimbot/capture
    ${CMAKE_SOURCE_DIR}/needaimbot/cuda
    ${CMAKE_SOURCE_DIR}/needaimbot/mouse/input_drivers
    ${CMAKE_SOURCE_DIR}/needaimbot/modules
    ${CUDA_PATH}/include
    ${TENSORRT_INCLUDE_DIR}
)

target_link_directories(simple_inference PRIVATE
    ${TENSORRT_LIB_DIR}
    ${CUDA_PATH}/lib64
)

target_link_libraries(simple_inference PRIVATE
    nvinfer
    CUDA::cudart
    CUDA::cuda_driver
    pthread
)

target_compile_definitions(simple_inference PRIVATE
    NOMINMAX
)

target_compile_options(simple_inference PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-fomit-frame-pointer>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-math-errno>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-trapping-math>
    $<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-fdata-sections>
)
if(SIMPLE_ENABLE_NATIVE_OPT)
    target_compile_options(simple_inference PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-march=native>
        $<$<COMPILE_LANGUAGE:CXX>:-mtune=native>
    )
endif()
target_compile_options(simple_inference PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-O3,-fomit-frame-pointer,-fno-math-errno,-fno-trapping-math,-ffunction-sections,-fdata-sections>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O3,-dlcm=ca>
    $<$<COMPILE_LANGUAGE:CUDA>:--extra-device-vectorization>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)
target_link_options(simple_inference PRIVATE
    -Wl,--gc-sections
)

set_target_properties(simple_inference PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Release"
)

# =============================================================================
# Seed the runtime config on a clean build.
# =============================================================================
# simple_config.default.json (tracked) is the canonical default. The app reads
# <exeDir>/simple_config.json and rewrites it at runtime, so we copy the default
# ONLY when the destination is absent - i.e. on a fresh/clean build where
# build/ was wiped. Incremental builds leave an existing (possibly runtime-
# tuned) simple_config.json untouched. To change the default, edit
# simple_config.default.json and commit it.
set(_simple_config_dest "${CMAKE_BINARY_DIR}/bin/Release/simple_config.json")
if(NOT EXISTS "${_simple_config_dest}")
    configure_file(
        "${CMAKE_SOURCE_DIR}/simple_config.default.json"
        "${_simple_config_dest}"
        COPYONLY
    )
    message(STATUS "Seeded default simple_config.json (clean build)")
endif()

if(SIMPLE_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT simple_ipo_supported OUTPUT simple_ipo_error)
    if(simple_ipo_supported)
        set_property(TARGET simple_inference PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "SIMPLE_ENABLE_LTO is ON but IPO is not supported: ${simple_ipo_error}")
    endif()
endif()
