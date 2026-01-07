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
# 난독화 옵션 (성능 영향 없음)
# =============================================================================
option(ENABLE_OBFUSCATION "Enable compile-time obfuscation" OFF)
option(ENABLE_ANTI_DEBUG "Enable anti-debugging checks" OFF)
option(STRIP_SYMBOLS "Strip debug symbols from binary" OFF)

# 소스 디렉토리 설정 (난독화 빌드 시 needaimbot_obf 사용)
set(SRC_DIR "needaimbot" CACHE STRING "Source directory name")

# =============================================================================
# Headless 빌드 옵션 (오버레이/GUI 완전 제거)
# =============================================================================
option(HEADLESS_BUILD "Build without overlay/GUI (no DirectX window)" OFF)

# =============================================================================
# 경로 설정 (여기만 수정하면 됨)
# =============================================================================
set(CUDA_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1")
set(TENSORRT_PATH "${CMAKE_SOURCE_DIR}/${SRC_DIR}/modules/TensorRT-10.14.1.48")
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
set(CPP_SOURCES_CORE
    ${SRC_DIR}/needaimbot.cpp
    ${SRC_DIR}/config/config.cpp
    ${SRC_DIR}/capture/dda_capture.cpp
    ${SRC_DIR}/keyboard/keyboard_listener.cpp
    ${SRC_DIR}/keyboard/keycodes.cpp
    ${SRC_DIR}/mouse/mouse.cpp
    ${SRC_DIR}/mouse/input_drivers/ghub.cpp
    ${SRC_DIR}/mouse/input_drivers/kmboxNet.cpp
    ${SRC_DIR}/mouse/input_drivers/my_enc.cpp
    ${SRC_DIR}/mouse/input_drivers/rzctl.cpp
    ${SRC_DIR}/mouse/input_drivers/SerialConnection.cpp
    ${SRC_DIR}/mouse/input_drivers/MakcuConnection.cpp
    ${SRC_DIR}/utils/image_io.cpp
    ${SRC_DIR}/utils/input_state.cpp
    ${SRC_DIR}/scr/other_tools.cpp
    ${SRC_DIR}/cuda/unified_graph_pipeline_cleanup.cpp
)

# 오버레이/GUI 소스 (HEADLESS_BUILD=OFF 일 때만 포함)
set(CPP_SOURCES_OVERLAY
    ${SRC_DIR}/overlay/overlay.cpp
    ${SRC_DIR}/overlay/draw_ai.cpp
    ${SRC_DIR}/overlay/draw_buttons.cpp
    ${SRC_DIR}/overlay/draw_capture.cpp
    ${SRC_DIR}/overlay/draw_debug.cpp
    ${SRC_DIR}/overlay/draw_mouse.cpp
    ${SRC_DIR}/overlay/draw_offset.cpp
    ${SRC_DIR}/overlay/draw_overlay.cpp
    ${SRC_DIR}/overlay/draw_profile.cpp
    ${SRC_DIR}/overlay/draw_target.cpp
    ${SRC_DIR}/overlay/draw_color_filter.cpp
    ${SRC_DIR}/overlay/draw_stabilizer.cpp
    ${SRC_DIR}/overlay/ui_helpers.cpp
    # ImGui
    ${SRC_DIR}/imgui/imgui.cpp
    ${SRC_DIR}/imgui/imgui_draw.cpp
    ${SRC_DIR}/imgui/imgui_tables.cpp
    ${SRC_DIR}/imgui/imgui_widgets.cpp
    ${SRC_DIR}/imgui/imgui_impl_dx11.cpp
    ${SRC_DIR}/imgui/imgui_impl_win32.cpp
)

# Headless 빌드 여부에 따라 소스 선택
if(HEADLESS_BUILD)
    set(CPP_SOURCES ${CPP_SOURCES_CORE})
    message(STATUS "HEADLESS_BUILD enabled - Overlay/GUI excluded from build")
else()
    set(CPP_SOURCES ${CPP_SOURCES_CORE} ${CPP_SOURCES_OVERLAY})
    message(STATUS "Full build with Overlay/GUI")
endif()

set(CUDA_SOURCES
    ${SRC_DIR}/cuda/unified_graph_pipeline.cu
    ${SRC_DIR}/cuda/preprocessing.cu
    ${SRC_DIR}/cuda/detection/postProcessGpu.cu
    ${SRC_DIR}/cuda/detection/cuda_float_processing.cu
)

# =============================================================================
# 실행 파일
# =============================================================================
add_executable(${PROJECT_NAME} ${CPP_SOURCES} ${CUDA_SOURCES})

# =============================================================================
# Include 경로
# =============================================================================
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/${SRC_DIR}
    ${CMAKE_SOURCE_DIR}/${SRC_DIR}/include
    ${CMAKE_SOURCE_DIR}/${SRC_DIR}/imgui
    ${CMAKE_SOURCE_DIR}/${SRC_DIR}/keyboard
    ${CMAKE_SOURCE_DIR}/${SRC_DIR}/config
    ${CMAKE_SOURCE_DIR}/${SRC_DIR}/capture
    ${CMAKE_SOURCE_DIR}/${SRC_DIR}/postprocess
    ${CMAKE_SOURCE_DIR}/${SRC_DIR}/mouse
    ${CMAKE_SOURCE_DIR}/${SRC_DIR}/overlay
    ${CMAKE_SOURCE_DIR}/${SRC_DIR}/modules/stb
    ${CMAKE_SOURCE_DIR}/${SRC_DIR}/modules
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
)

# 난독화 정의 추가
if(ENABLE_OBFUSCATION)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_OBFUSCATION)
endif()

if(ENABLE_ANTI_DEBUG)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_ANTI_DEBUG)
endif()

# Headless 빌드 정의 추가
if(HEADLESS_BUILD)
    target_compile_definitions(${PROJECT_NAME} PRIVATE HEADLESS_BUILD)
endif()

if(MSVC)
    # C++ 컴파일 옵션 (CUDA 제외)
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
        $<$<COMPILE_LANGUAGE:CXX>:/guard:cf>
        $<$<COMPILE_LANGUAGE:CXX>:/Qspectre>
    )
    
    # CUDA 컴파일 옵션
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W3,/utf-8,/wd4819,/wd4244,/O2,/Oi>
    )
    
    target_link_options(${PROJECT_NAME} PRIVATE
        /LTCG
    )
    
    # 심볼 스트립 (난독화 빌드용)
    if(STRIP_SYMBOLS)
        target_link_options(${PROJECT_NAME} PRIVATE
            /DEBUG:NONE
            /OPT:REF
            /OPT:ICF
        )
        target_compile_options(${PROJECT_NAME} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/Gw>
            $<$<COMPILE_LANGUAGE:CXX>:/GS->
        )
    endif()
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
