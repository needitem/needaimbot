cmake_minimum_required(VERSION 3.24)

project(needaimbot_2PC)

# Option to select which component to build
option(BUILD_GAME_PC "Build GamePC streamer component" OFF)
option(BUILD_INFERENCE_PC "Build InferencePC component" ON)

if(BUILD_GAME_PC)
    add_subdirectory(game_pc)
endif()

if(BUILD_INFERENCE_PC)
    add_subdirectory(inference_pc)
endif()

# If neither is selected, show help
if(NOT BUILD_GAME_PC AND NOT BUILD_INFERENCE_PC)
    message(STATUS "")
    message(STATUS "========================================")
    message(STATUS "needaimbot 2PC Architecture Build System")
    message(STATUS "========================================")
    message(STATUS "")
    message(STATUS "Select which component to build:")
    message(STATUS "")
    message(STATUS "  cmake -DBUILD_GAME_PC=ON ..      # GamePC streamer (Windows, no CUDA)")
    message(STATUS "  cmake -DBUILD_INFERENCE_PC=ON .. # InferencePC (Windows + CUDA + TensorRT)")
    message(STATUS "")
    message(STATUS "Or build from component directory directly:")
    message(STATUS "")
    message(STATUS "  cd game_pc && cmake -B build && cmake --build build")
    message(STATUS "  cd inference_pc && cmake -B build && cmake --build build")
    message(STATUS "")
endif()
