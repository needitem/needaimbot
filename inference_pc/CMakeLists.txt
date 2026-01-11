cmake_minimum_required(VERSION 3.16)

# =============================================================================
# Platform Detection and CUDA Configuration
# =============================================================================
if(WIN32)
    # Windows CUDA path configuration
    set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1")
    set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc.exe")
    set(CUDAToolkit_ROOT "${CUDA_TOOLKIT_ROOT_DIR}")
    set(ENV{CUDA_PATH} "${CUDA_TOOLKIT_ROOT_DIR}")
else()
    # Linux/Jetson: Use system CUDA
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
    set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
    set(CUDAToolkit_ROOT "${CUDA_TOOLKIT_ROOT_DIR}")
endif()

project(InferencePC LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# =============================================================================
# Path Configuration (Platform-specific)
# =============================================================================
if(WIN32)
    set(CUDA_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1")
    set(TENSORRT_PATH "${CMAKE_SOURCE_DIR}/needaimbot/modules/TensorRT-10.14.1.48")
    set(CUDNN_PATH "C:/Program Files/NVIDIA/CUDNN/v9.17")
    set(CUDNN_VERSION "13.1")
    set(TENSORRT_LIB_NAMES nvinfer_10 nvonnxparser_10)
else()
    # Linux/Jetson: Use system-installed TensorRT
    set(CUDA_PATH "/usr/local/cuda")
    set(TENSORRT_INCLUDE_DIR "/usr/include/aarch64-linux-gnu")
    set(TENSORRT_LIB_DIR "/usr/lib/aarch64-linux-gnu")
    set(TENSORRT_LIB_NAMES nvinfer nvonnxparser)
endif()

# =============================================================================
# CUDA Configuration
# =============================================================================
find_package(CUDAToolkit REQUIRED)

if(WIN32)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
else()
    # Jetson Xavier NX / AGX: SM 7.2, Jetson Orin: SM 8.7
    set(CMAKE_CUDA_ARCHITECTURES 72 87)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --diag-suppress=611,221,20091,20054")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCUB_IGNORE_DEPRECATED_CPP_DIALECT")
if(WIN32)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING")
endif()

# =============================================================================
# Source Files - 2PC Architecture (Makcu/KMBox only, UDP capture only)
# =============================================================================
set(CPP_SOURCES
    needaimbot/needaimbot.cpp
    needaimbot/config/config.cpp
    needaimbot/capture/udp_capture.cpp
    needaimbot/capture/lz4.c
    needaimbot/keyboard/keyboard_listener.cpp
    needaimbot/keyboard/keycodes.cpp
    needaimbot/mouse/mouse.cpp
    # Input drivers - only Makcu and KMBox
    needaimbot/mouse/input_drivers/MakcuConnection.cpp
    needaimbot/mouse/input_drivers/kmboxNet.cpp
    needaimbot/mouse/input_drivers/my_enc.cpp
    needaimbot/utils/image_io.cpp
    needaimbot/utils/input_state.cpp
    needaimbot/scr/other_tools.cpp
    needaimbot/cuda/unified_graph_pipeline_cleanup.cpp
    needaimbot/cuda/cuda_resource_manager.cpp
)

set(CUDA_SOURCES
    needaimbot/cuda/unified_graph_pipeline.cu
    needaimbot/cuda/preprocessing.cu
    needaimbot/cuda/detection/postProcessGpu.cu
    needaimbot/cuda/detection/cuda_float_processing.cu
)

# =============================================================================
# Executable
# =============================================================================
add_executable(${PROJECT_NAME} ${CPP_SOURCES} ${CUDA_SOURCES})

# =============================================================================
# Include Directories
# =============================================================================
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/needaimbot
    ${CMAKE_SOURCE_DIR}/needaimbot/include
    ${CMAKE_SOURCE_DIR}/needaimbot/keyboard
    ${CMAKE_SOURCE_DIR}/needaimbot/config
    ${CMAKE_SOURCE_DIR}/needaimbot/capture
    ${CMAKE_SOURCE_DIR}/needaimbot/postprocess
    ${CMAKE_SOURCE_DIR}/needaimbot/mouse
    ${CMAKE_SOURCE_DIR}/needaimbot/modules/stb
    ${CMAKE_SOURCE_DIR}/needaimbot/modules
    ${CUDA_PATH}/include
)

if(WIN32)
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${TENSORRT_PATH}/include
        ${CUDNN_PATH}/include/${CUDNN_VERSION}
    )
else()
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${TENSORRT_INCLUDE_DIR}
    )
endif()

# =============================================================================
# Library Directories
# =============================================================================
if(WIN32)
    target_link_directories(${PROJECT_NAME} PRIVATE
        ${TENSORRT_PATH}/lib
        ${CUDA_PATH}/lib/x64
        ${CUDNN_PATH}/lib/${CUDNN_VERSION}
    )
else()
    target_link_directories(${PROJECT_NAME} PRIVATE
        ${TENSORRT_LIB_DIR}
        ${CUDA_PATH}/lib64
    )
endif()

# =============================================================================
# Link Libraries
# =============================================================================
target_link_libraries(${PROJECT_NAME} PRIVATE
    # TensorRT (platform-specific names)
    ${TENSORRT_LIB_NAMES}
    # CUDA
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas
)

if(WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        # Network
        ws2_32
    )
else()
    target_link_libraries(${PROJECT_NAME} PRIVATE
        # Linux network
        pthread
    )
endif()

# =============================================================================
# Compile Definitions
# =============================================================================
target_compile_definitions(${PROJECT_NAME} PRIVATE
    NDEBUG
    _CONSOLE
    _CRT_SECURE_NO_WARNINGS
    UNICODE
    _UNICODE
    # 2PC Architecture flags
    USE_2PC_ARCHITECTURE
    HEADLESS_BUILD  # No overlay on inference PC
)

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W3>
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4819>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4244>
        $<$<COMPILE_LANGUAGE:CXX>:/openmp>
        $<$<COMPILE_LANGUAGE:CXX>:/O2>
        $<$<COMPILE_LANGUAGE:CXX>:/Oi>
        $<$<COMPILE_LANGUAGE:CXX>:/Oy>
        $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>
        $<$<COMPILE_LANGUAGE:CXX>:/fp:fast>
    )

    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W3,/utf-8,/wd4819,/wd4244,/O2,/Oi>
    )

    target_link_options(${PROJECT_NAME} PRIVATE
        /LTCG
    )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
        $<$<COMPILE_LANGUAGE:CXX>:-ffast-math>
        $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>
    )
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-O3>
    )
endif()

# =============================================================================
# Copy DLLs (post-build) - Windows only
# =============================================================================
if(WIN32)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${TENSORRT_PATH}/bin/nvinfer_10.dll"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${TENSORRT_PATH}/bin/nvonnxparser_10.dll"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${TENSORRT_PATH}/bin/nvinfer_plugin_10.dll"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CUDNN_PATH}/bin/${CUDNN_VERSION}/cudnn64_9.dll"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
        COMMENT "Copying runtime DLLs..."
    )
endif()

# =============================================================================
# Output Directory
# =============================================================================
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/Debug"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/Release"
)

# =============================================================================
# Test Executable: UDP Inference Test
# =============================================================================
add_executable(test_udp_inference
    test_udp_inference.cpp
    preprocessing_simple.cu
    needaimbot/capture/udp_capture.cpp
    needaimbot/capture/lz4.c
)

target_include_directories(test_udp_inference PRIVATE
    ${CMAKE_SOURCE_DIR}/needaimbot
    ${CMAKE_SOURCE_DIR}/needaimbot/include
    ${CMAKE_SOURCE_DIR}/needaimbot/capture
    ${CMAKE_SOURCE_DIR}/needaimbot/modules
    ${CMAKE_SOURCE_DIR}/needaimbot/modules/glfw/include
    ${CUDA_PATH}/include
)

if(WIN32)
    target_include_directories(test_udp_inference PRIVATE
        ${TENSORRT_PATH}/include
    )
    target_link_directories(test_udp_inference PRIVATE
        ${TENSORRT_PATH}/lib
        ${CUDA_PATH}/lib/x64
    )
    target_link_libraries(test_udp_inference PRIVATE
        nvinfer_10
        CUDA::cudart
        CUDA::cuda_driver
        ws2_32
    )
else()
    target_include_directories(test_udp_inference PRIVATE
        ${TENSORRT_INCLUDE_DIR}
    )
    target_link_directories(test_udp_inference PRIVATE
        ${TENSORRT_LIB_DIR}
        ${CUDA_PATH}/lib64
    )
    target_link_libraries(test_udp_inference PRIVATE
        nvinfer
        CUDA::cudart
        CUDA::cuda_driver
        pthread
    )
endif()

target_compile_definitions(test_udp_inference PRIVATE
    NOMINMAX
)
if(WIN32)
    target_compile_definitions(test_udp_inference PRIVATE
        _CRT_SECURE_NO_WARNINGS
        WIN32_LEAN_AND_MEAN
    )
endif()

set_target_properties(test_udp_inference PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

if(MSVC)
    target_compile_options(test_udp_inference PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W3>
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CXX>:/O2>
    )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(test_udp_inference PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
    )
endif()

set_target_properties(test_udp_inference PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/Debug"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/Release"
)

# Copy DLLs for test executable (Windows only)
if(WIN32)
    add_custom_command(TARGET test_udp_inference POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${TENSORRT_PATH}/bin/nvinfer_10.dll"
            "$<TARGET_FILE_DIR:test_udp_inference>"
        COMMENT "Copying TensorRT DLLs for test_udp_inference..."
    )
endif()

# =============================================================================
# Test Executable: Makcu Mouse Test
# =============================================================================
add_executable(test_makcu_mouse
    test_makcu_mouse.cpp
    needaimbot/mouse/input_drivers/MakcuConnection.cpp
)

target_include_directories(test_makcu_mouse PRIVATE
    ${CMAKE_SOURCE_DIR}/needaimbot
    ${CMAKE_SOURCE_DIR}/needaimbot/include
)

target_link_libraries(test_makcu_mouse PRIVATE
    pthread
)

target_compile_definitions(test_makcu_mouse PRIVATE
    NOMINMAX
)

set_target_properties(test_makcu_mouse PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/Debug"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/Release"
)

# =============================================================================
# EngineExport Subproject - ONNX to TensorRT Engine Converter
# =============================================================================
option(BUILD_ENGINE_EXPORT "Build EngineExport GUI tool (requires GLFW)" ON)
if(BUILD_ENGINE_EXPORT)
    # Check if GLFW is available on Linux
    if(NOT WIN32)
        find_package(glfw3 QUIET)
        find_library(GLFW_LIB_CHECK glfw HINTS /usr/lib /usr/lib/aarch64-linux-gnu)
        if(NOT glfw3_FOUND AND NOT GLFW_LIB_CHECK)
            message(WARNING "GLFW not found. EngineExport will not be built. Install with: sudo apt-get install libglfw3-dev")
            set(BUILD_ENGINE_EXPORT OFF)
        endif()
    endif()
    if(BUILD_ENGINE_EXPORT)
        add_subdirectory(engine_export)
    endif()
endif()
