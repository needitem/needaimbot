cmake_minimum_required(VERSION 3.10)
project(MakcuRelay LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    # Statically link the MSVC runtime so the second PC
    # does not need a separate VC++ Redistributable installed.
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

if (WIN32)
    add_executable(MakcuRelay WIN32
        src/main.cpp
        src/MakcuConnection.cpp
    )
else()
    add_executable(MakcuRelay
        src/main.cpp
        src/MakcuConnection.cpp
    )
endif()

target_include_directories(MakcuRelay PRIVATE src)

if (WIN32)
    target_link_libraries(MakcuRelay PRIVATE ws2_32)
else()
    # Linux: link pthread for std::thread
    target_link_libraries(MakcuRelay PRIVATE pthread)
endif()
