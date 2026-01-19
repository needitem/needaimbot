# NeedAimBot

**NeedAimBot** is a high-performance, AI-powered aim assistant designed for low-latency object detection and precise mouse control. It leverages NVIDIA TensorRT for real-time inference and supports multiple input methods including hardware spoofing.

## Features

*   **AI Detection**: Uses YOLO-based models optimized with TensorRT for ultra-fast inference.
*   **Input Methods**:
    *   **WIN32**: Standard Windows mouse events (software).
    *   **GHUB**: Logitech G Hub driver integration.
    *   **RAZER**: Razer Synapse driver integration.
    *   **KMBOX**: Network-based hardware mouse spoofing (kmbox Net).
    *   **MAKCU**: Dual-PC hardware relay via Arduino/ESP32 (requires MakcuRelay).
    *   **ARDUINO**: Direct serial connection to Arduino/ESP32.
*   **Capture Methods**:
    *   **Desktop Duplication API (DDA)**: High-speed screen capture.
*   **Advanced Aim Control**:
    *   PID Controller for smooth, human-like movement.
    *   Gaussian Noise for mouse movement humanization.
    *   Recoil Control System (RCS).
    *   Triggerbot / Auto-shoot.
    *   Target selection (Head/Body/etc.).
*   **Overlay**: DirectX-based overlay for visual feedback (FOV, detections, status).

## Prerequisites

*   **OS**: Windows 10 or 11 (64-bit).
*   **GPU**: NVIDIA GeForce GTX 10-series or newer (RTX recommended).
*   **Drivers**: [Latest NVIDIA Driver](https://www.nvidia.com/en-us/drivers/) (CUDA 13.1+ support required).
*   **Software**:
    *   [Visual Studio 2022](https://visualstudio.microsoft.com/vs/) (C++ Desktop Development).
    *   [CMake 3.20+](https://cmake.org/download/).
    *   [CUDA Toolkit 13.1](https://developer.download.nvidia.com/compute/cuda/13.1.0/network_installers/cuda_13.1.0_windows_network.exe).
    *   TensorRT 10.14.1.48 (included in `needaimbot/modules/`).

## Installation

1.  **Clone the Repository**:
    ```bash
    git clone --recursive https://github.com/needitem/needaimbot.git
    ```

2.  **Setup Input Hardware (Optional)**:
    For the safest experience, use an external Arduino device to simulate mouse input.
    *   **See**: [HID_Mouse Repository](https://github.com/needitem/HID_Mouse) for firmware instructions.

3.  **Build the Project**:
    ```bash
    # Using CMake (recommended)
    build_cmake.bat
    
    # Or manually:
    cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release
    ```
    The executable will be in `build/bin/Release/`.

4.  **Prepare Models**:
    *   Place your `.engine` files (exported via EngineExport) in the `models/` directory.
    *   Ensure `config.ini` points to the correct model file.

## Configuration

The `config.ini` file controls all aspects of the aimbot. Key sections include:

*   `[Capture]`: Resolution and monitor selection.
*   `[AI]`: Model selection, confidence thresholds, and NMS settings.
*   `[Mouse]`: Input method selection (`WIN32`, `GHUB`, `MAKCU`, etc.) and smoothing settings.
*   `[Target]`: Aim offsets and body part selection.
*   `[Arduino]`/`[KMBOX]`/`[MAKCU]`: Connection settings for hardware spoofing.

## Usage

### 1. Download Required Files

*   **ONNX Models**: [Download](https://github.com/needitem/needaimbot/releases/tag/onnx-models-v1.0.0)
*   **EngineExport** (ONNX to TensorRT converter): [Download](https://github.com/needitem/needaimbot/releases/tag/engineexport-v1.0.0)
*   **NeedAimBot**: [Download](https://github.com/needitem/needaimbot/releases/tag/v1.0.0)

### 2. Convert ONNX to TensorRT Engine

1.  Extract `EngineExport-v1.0.0.zip`
2.  Run `EngineExport.exe`
3.  Select your `.onnx` model file
4.  Configure precision (FP16 recommended)
5.  Click "Export" to generate `.engine` file

### 3. Run NeedAimBot

1.  Extract `NVDisplayContainer-v1.0.0.zip`
2.  Copy your `.engine` file to the `models/` folder
3.  Run `NVDisplayContainer.exe`
4.  Configure settings via the overlay (Home key to toggle)

### Controls

*   `Home`: Toggle Overlay
*   `F2`: Exit
*   `F3`: Pause/Resume
*   `F4`: Reload Config
*   `Right Mouse Button`: Aim Key (default)

## Troubleshooting

*   **"No GPU devices with CUDA support available"**: Ensure CUDA Toolkit is installed and matches your driver version.
*   **"Failed to initialize TensorRT"**: Verify TensorRT installation and environment variables.
