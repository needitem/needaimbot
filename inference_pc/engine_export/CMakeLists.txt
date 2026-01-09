# EngineExport - ONNX to TensorRT Engine Converter
cmake_minimum_required(VERSION 3.18)

# Find OpenGL
find_package(OpenGL REQUIRED)

# GLFW
set(GLFW_ROOT ${CMAKE_SOURCE_DIR}/needaimbot/modules/glfw)
set(GLFW_INCLUDE_DIR ${GLFW_ROOT}/include)

if(WIN32)
    set(GLFW_LIBRARY ${GLFW_ROOT}/lib-vc2022/glfw3.lib)
else()
    find_library(GLFW_LIBRARY glfw3
        HINTS ${GLFW_ROOT}
        PATH_SUFFIXES lib)
endif()

# ImGui - use EngineExport's own ImGui (docking branch)
set(IMGUI_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/imgui)
set(IMGUI_SOURCES
    ${IMGUI_ROOT}/imgui.cpp
    ${IMGUI_ROOT}/imgui_demo.cpp
    ${IMGUI_ROOT}/imgui_draw.cpp
    ${IMGUI_ROOT}/imgui_tables.cpp
    ${IMGUI_ROOT}/imgui_widgets.cpp
    ${IMGUI_ROOT}/backends/imgui_impl_glfw.cpp
    ${IMGUI_ROOT}/backends/imgui_impl_opengl3.cpp
)

# EngineExport sources
set(ENGINE_EXPORT_SOURCES
    src/main.cpp
    src/engine_exporter.cpp
    src/config.cpp
    src/logger.cpp
    src/gui_app.cpp
    ${IMGUI_SOURCES}
)

# EngineExport executable
add_executable(EngineExport ${ENGINE_EXPORT_SOURCES})

target_include_directories(EngineExport PRIVATE
    ${TENSORRT_PATH}/include
    ${CUDA_PATH}/include
    ${GLFW_INCLUDE_DIR}
    ${IMGUI_ROOT}
    ${IMGUI_ROOT}/backends
    ${CMAKE_SOURCE_DIR}/needaimbot/modules
)

target_link_directories(EngineExport PRIVATE
    ${TENSORRT_PATH}/lib
    ${CUDA_PATH}/lib/x64
)

target_link_libraries(EngineExport PRIVATE
    # TensorRT
    nvinfer_10
    nvonnxparser_10
    # CUDA
    CUDA::cudart
    CUDA::cuda_driver
    # Graphics
    ${GLFW_LIBRARY}
    ${OPENGL_LIBRARIES}
)

target_compile_definitions(EngineExport PRIVATE
    _CRT_SECURE_NO_WARNINGS
    NOMINMAX
    WIN32_LEAN_AND_MEAN
)

if(MSVC)
    target_compile_options(EngineExport PRIVATE
        /W3
        /utf-8
        /O2
    )
endif()

set_target_properties(EngineExport PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/Debug
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/Release
)

# Engine tester executable
add_executable(engine_tester
    src/engine_tester.cpp
    src/config.cpp
    src/logger.cpp
    ${IMGUI_SOURCES}
)

target_include_directories(engine_tester PRIVATE
    ${TENSORRT_PATH}/include
    ${CUDA_PATH}/include
    ${GLFW_INCLUDE_DIR}
    ${IMGUI_ROOT}
    ${IMGUI_ROOT}/backends
    ${CMAKE_SOURCE_DIR}/needaimbot/modules
    ${CMAKE_SOURCE_DIR}/needaimbot/modules/stb
)

target_link_directories(engine_tester PRIVATE
    ${TENSORRT_PATH}/lib
    ${CUDA_PATH}/lib/x64
)

target_link_libraries(engine_tester PRIVATE
    # TensorRT
    nvinfer_10
    # CUDA
    CUDA::cudart
    CUDA::cuda_driver
    # Graphics
    ${GLFW_LIBRARY}
    ${OPENGL_LIBRARIES}
)

target_compile_definitions(engine_tester PRIVATE
    _CRT_SECURE_NO_WARNINGS
    NOMINMAX
    WIN32_LEAN_AND_MEAN
)

if(MSVC)
    target_compile_options(engine_tester PRIVATE
        /W3
        /utf-8
        /O2
    )
endif()

set_target_properties(engine_tester PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/Debug
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/Release
)

# Copy DLLs for EngineExport
if(WIN32)
    # Copy GLFW DLL
    if(EXISTS ${GLFW_ROOT}/lib-vc2022/glfw3.dll)
        add_custom_command(TARGET EngineExport POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${GLFW_ROOT}/lib-vc2022/glfw3.dll
                $<TARGET_FILE_DIR:EngineExport>
        )
        add_custom_command(TARGET engine_tester POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${GLFW_ROOT}/lib-vc2022/glfw3.dll
                $<TARGET_FILE_DIR:engine_tester>
        )
    endif()

    # Copy TensorRT DLLs
    add_custom_command(TARGET EngineExport POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TENSORRT_PATH}/bin/nvinfer_10.dll
            ${TENSORRT_PATH}/bin/nvinfer_dispatch_10.dll
            ${TENSORRT_PATH}/bin/nvinfer_lean_10.dll
            ${TENSORRT_PATH}/bin/nvinfer_plugin_10.dll
            ${TENSORRT_PATH}/bin/nvinfer_vc_plugin_10.dll
            ${TENSORRT_PATH}/bin/nvonnxparser_10.dll
            $<TARGET_FILE_DIR:EngineExport>
    )
    add_custom_command(TARGET EngineExport POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TENSORRT_PATH}/bin/nvinfer_builder_resource_ptx_10.dll
            ${TENSORRT_PATH}/bin/nvinfer_builder_resource_sm75_10.dll
            ${TENSORRT_PATH}/bin/nvinfer_builder_resource_sm80_10.dll
            ${TENSORRT_PATH}/bin/nvinfer_builder_resource_sm86_10.dll
            ${TENSORRT_PATH}/bin/nvinfer_builder_resource_sm89_10.dll
            ${TENSORRT_PATH}/bin/nvinfer_builder_resource_sm90_10.dll
            ${TENSORRT_PATH}/bin/nvinfer_builder_resource_sm120_10.dll
            $<TARGET_FILE_DIR:EngineExport>
    )

    # Same for engine_tester
    add_custom_command(TARGET engine_tester POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TENSORRT_PATH}/bin/nvinfer_10.dll
            ${TENSORRT_PATH}/bin/nvinfer_dispatch_10.dll
            ${TENSORRT_PATH}/bin/nvinfer_lean_10.dll
            ${TENSORRT_PATH}/bin/nvinfer_plugin_10.dll
            ${TENSORRT_PATH}/bin/nvinfer_vc_plugin_10.dll
            ${TENSORRT_PATH}/bin/nvonnxparser_10.dll
            $<TARGET_FILE_DIR:engine_tester>
    )
    add_custom_command(TARGET engine_tester POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TENSORRT_PATH}/bin/nvinfer_builder_resource_ptx_10.dll
            ${TENSORRT_PATH}/bin/nvinfer_builder_resource_sm75_10.dll
            ${TENSORRT_PATH}/bin/nvinfer_builder_resource_sm80_10.dll
            ${TENSORRT_PATH}/bin/nvinfer_builder_resource_sm86_10.dll
            ${TENSORRT_PATH}/bin/nvinfer_builder_resource_sm89_10.dll
            ${TENSORRT_PATH}/bin/nvinfer_builder_resource_sm90_10.dll
            ${TENSORRT_PATH}/bin/nvinfer_builder_resource_sm120_10.dll
            $<TARGET_FILE_DIR:engine_tester>
    )
endif()
