cmake_minimum_required(VERSION 3.18)

project(GamePC_Streamer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT WIN32)
    message(FATAL_ERROR "GamePC Streamer is Windows-only")
endif()

add_executable(${PROJECT_NAME} WIN32
    src/main.cpp
    src/lz4.c
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    NOMINMAX
    WIN32_LEAN_AND_MEAN
    _CRT_SECURE_NO_WARNINGS
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ws2_32
    d3d11
    dxgi
)

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        /W3 /O2 /utf-8
    )
    # Console subsystem for debugging
    set_target_properties(${PROJECT_NAME} PROPERTIES
        LINK_FLAGS "/SUBSYSTEM:CONSOLE"
    )
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/Release"
)
