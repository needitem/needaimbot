# Project Overview

## Project Purpose
needaimbot is a high-performance C++ AI-powered aiming assistance tool that utilizes deep learning models and advanced computer vision techniques for real-time target detection and tracking. Built with cutting-edge technologies including TensorRT for GPU acceleration, CUDA for parallel computing, and enhanced Kalman filtering with frame-based prediction.

## Tech Stack
- **Language**: C++17
- **Graphics**: CUDA 12.8, TensorRT 10.8
- **UI**: Dear ImGui with DirectX 11
- **Build System**: Visual Studio 2022, MSBuild
- **Libraries**: 
  - NVIDIA CUDA Toolkit
  - NVIDIA TensorRT
  - Dear ImGui
  - Eigen (math library)
  - GLFW (window management)
  - STB Libraries (image processing)
  - SimpleIni (configuration)
  - cuDNN 9.7.1

## Main Components
- **AppContext**: God Object containing all global state (needs refactoring)
- **Detector**: AI inference engine using TensorRT
- **Capture**: Screen capture with multiple methods (Desktop Duplication API, etc.)
- **Mouse**: Input handling for various devices (Logitech G-Hub, Razer, KMBox, Serial)
- **Overlay**: Real-time UI using ImGui
- **CUDA Kernels**: GPU-accelerated processing
- **Tracking**: Kalman filter-based target tracking

## Performance Requirements
- Inference Time: <5ms (RTX 3080)
- End-to-End Latency: <10ms total
- VRAM Usage: ~2GB
- RAM Usage: ~500MB
- CPU Usage: <10%
- Max Frame Rate: 240 FPS

## System Requirements
- Windows 10/11 (x64)
- NVIDIA RTX 2060 or better (Turing architecture or newer)
- 8GB RAM minimum, 16GB recommended
- CUDA 12.8
- 2GB free storage