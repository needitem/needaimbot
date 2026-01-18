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
set(CMAKE_CUDA_ARCHITECTURES 72 87)

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
    $<$<COMPILE_LANGUAGE:CXX>:-ffast-math>
)
target_compile_options(simple_inference PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-O3>
)

set_target_properties(simple_inference PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Release"
)

# =============================================================================
# Test Inference (standalone test)
# =============================================================================
add_executable(test_inference
    test_inference.cpp
    needaimbot/cuda/simple_inference.cu
    needaimbot/cuda/simple_postprocess.cu
)

target_include_directories(test_inference PRIVATE
    ${CMAKE_SOURCE_DIR}/needaimbot
    ${CMAKE_SOURCE_DIR}/needaimbot/cuda
    ${CUDA_PATH}/include
    ${TENSORRT_INCLUDE_DIR}
)

target_link_directories(test_inference PRIVATE
    ${TENSORRT_LIB_DIR}
    ${CUDA_PATH}/lib64
)

target_link_libraries(test_inference PRIVATE
    nvinfer
    CUDA::cudart
    CUDA::cuda_driver
)

target_compile_options(test_inference PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3>
)

set_target_properties(test_inference PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Release"
)

# =============================================================================
# EngineExport Subproject (optional)
# =============================================================================
option(BUILD_ENGINE_EXPORT "Build EngineExport tool" OFF)
if(BUILD_ENGINE_EXPORT)
    add_subdirectory(engine_export)
endif()
