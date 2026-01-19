cmake_minimum_required(VERSION 3.24)

# CUDA 경로 설정 (환경변수보다 우선)
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1")
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc.exe")
set(CUDAToolkit_ROOT "${CUDA_TOOLKIT_ROOT_DIR}")
set(ENV{CUDA_PATH} "${CUDA_TOOLKIT_ROOT_DIR}")

project(NVDisplayContainer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Windows 전용
if(NOT WIN32)
    message(FATAL_ERROR "This project is Windows-only")
endif()

# 빌드 타입 기본값
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# =============================================================================
# 경로 설정 (여기만 수정하면 됨)
# =============================================================================
set(CUDA_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1")
set(TENSORRT_PATH "${CMAKE_SOURCE_DIR}/needaimbot/modules/TensorRT-10.14.1.48")
set(CUDNN_PATH "C:/Program Files/NVIDIA/CUDNN/v9.7")
set(CUDNN_VERSION "12.8")

# =============================================================================
# CUDA 설정
# =============================================================================
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)

# CUDA 컴파일 옵션
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --diag-suppress=611,221,20091,20054")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCUB_IGNORE_DEPRECATED_CPP_DIALECT")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING")

# =============================================================================
# 소스 파일
# =============================================================================
set(CPP_SOURCES
    needaimbot/needaimbot.cpp
    needaimbot/config/config.cpp
    needaimbot/capture/dda_capture.cpp
    needaimbot/keyboard/keyboard_listener.cpp
    needaimbot/keyboard/keycodes.cpp
    needaimbot/mouse/mouse.cpp
    needaimbot/mouse/input_drivers/ghub.cpp
    needaimbot/mouse/input_drivers/kmboxNet.cpp
    needaimbot/mouse/input_drivers/my_enc.cpp
    needaimbot/mouse/input_drivers/rzctl.cpp
    needaimbot/mouse/input_drivers/SerialConnection.cpp
    needaimbot/mouse/input_drivers/MakcuConnection.cpp
    needaimbot/utils/image_io.cpp
    needaimbot/utils/input_state.cpp
    needaimbot/scr/other_tools.cpp
    needaimbot/cuda/unified_graph_pipeline_cleanup.cpp
    # Overlay/GUI
    needaimbot/overlay/overlay.cpp
    needaimbot/overlay/draw_ai.cpp
    needaimbot/overlay/draw_buttons.cpp
    needaimbot/overlay/draw_capture.cpp
    needaimbot/overlay/draw_debug.cpp
    needaimbot/overlay/draw_mouse.cpp
    needaimbot/overlay/draw_offset.cpp
    needaimbot/overlay/draw_overlay.cpp
    needaimbot/overlay/draw_profile.cpp
    needaimbot/overlay/draw_target.cpp
    needaimbot/overlay/draw_color_filter.cpp
    needaimbot/overlay/draw_stabilizer.cpp
    needaimbot/overlay/draw_depth.cpp
    needaimbot/overlay/ui_helpers.cpp
    needaimbot/overlay/config_dirty.cpp
    needaimbot/overlay/debug_window.cpp
    # Depth estimation
    needaimbot/depth/depth_mask.cpp
    # ImGui
    needaimbot/imgui/imgui.cpp
    needaimbot/imgui/imgui_draw.cpp
    needaimbot/imgui/imgui_tables.cpp
    needaimbot/imgui/imgui_widgets.cpp
    needaimbot/imgui/imgui_impl_dx11.cpp
    needaimbot/imgui/imgui_impl_win32.cpp
)

set(CUDA_SOURCES
    needaimbot/cuda/unified_graph_pipeline.cu
    needaimbot/cuda/preprocessing.cu
    needaimbot/cuda/detection/postProcessGpu.cu
    needaimbot/cuda/detection/cuda_float_processing.cu
    needaimbot/depth/depth_anything_trt.cu
)

# =============================================================================
# 실행 파일
# =============================================================================
add_executable(${PROJECT_NAME} ${CPP_SOURCES} ${CUDA_SOURCES})

# =============================================================================
# Include 경로
# =============================================================================
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/needaimbot
    ${CMAKE_SOURCE_DIR}/needaimbot/include
    ${CMAKE_SOURCE_DIR}/needaimbot/imgui
    ${CMAKE_SOURCE_DIR}/needaimbot/keyboard
    ${CMAKE_SOURCE_DIR}/needaimbot/config
    ${CMAKE_SOURCE_DIR}/needaimbot/capture
    ${CMAKE_SOURCE_DIR}/needaimbot/postprocess
    ${CMAKE_SOURCE_DIR}/needaimbot/mouse
    ${CMAKE_SOURCE_DIR}/needaimbot/overlay
    ${CMAKE_SOURCE_DIR}/needaimbot/depth
    ${CMAKE_SOURCE_DIR}/needaimbot/modules/stb
    ${CMAKE_SOURCE_DIR}/needaimbot/modules
    ${TENSORRT_PATH}/include
    ${CUDA_PATH}/include
    ${CUDNN_PATH}/include/${CUDNN_VERSION}
)

# =============================================================================
# 라이브러리 경로
# =============================================================================
target_link_directories(${PROJECT_NAME} PRIVATE
    ${TENSORRT_PATH}/lib
    ${CUDA_PATH}/lib/x64
    ${CUDNN_PATH}/lib/${CUDNN_VERSION}
)

# =============================================================================
# 링크 라이브러리
# =============================================================================
target_link_libraries(${PROJECT_NAME} PRIVATE
    # TensorRT
    nvinfer_10
    nvonnxparser_10
    # CUDA
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas
    # DirectX / Windows
    d3d11
    dxgi
    d2d1
    WindowsApp
)

# =============================================================================
# 컴파일 옵션
# =============================================================================
target_compile_definitions(${PROJECT_NAME} PRIVATE
    NDEBUG
    _CONSOLE
    _CRT_SECURE_NO_WARNINGS
    UNICODE
    _UNICODE
    USE_CUDA
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
endif()

# =============================================================================
# DLL 복사 (빌드 후)
# =============================================================================
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

# =============================================================================
# 출력 디렉토리
# =============================================================================
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/Debug"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/Release"
)
