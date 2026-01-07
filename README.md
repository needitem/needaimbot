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
    *   **OBS Hook**: Capture via OBS graphics hook (requires OBS Studio).
*   **Advanced Aim Control**:
    *   PID Controller for smooth, human-like movement.
    *   Recoil Control System (RCS).
    *   Triggerbot / Auto-shoot.
    *   Target selection (Head/Body/etc.).
*   **Overlay**: DirectX-based overlay for visual feedback (FOV, detections, status).

## Prerequisites

*   **OS**: Windows 10 or 11 (64-bit).
*   **GPU**: NVIDIA GeForce GTX 10-series or newer (RTX recommended).
*   **Drivers**: Latest NVIDIA Game Ready Driver.
*   **Software**:
    *   [Visual Studio 2022](https://visualstudio.microsoft.com/vs/) (C++ Desktop Development).
    *   [CMake 3.20+](https://cmake.org/download/).
    *   [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads).
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

1.  **Run the Application**:
    Execute `needaimbot.exe` from the build directory.
    *   *Note: Run as Administrator if using certain input methods or capture hooks.*

2.  **Controls**:
    *   `Home`: Toggle Overlay.
    *   `F2`: Exit.
    *   `F3`: Pause/Resume.
    *   `F4`: Reload Config.
    *   `Right Mouse Button`: Aim Key (default).

## Troubleshooting

*   **"No GPU devices with CUDA support available"**: Ensure CUDA Toolkit is installed and matches your driver version.
*   **"Failed to initialize TensorRT"**: Verify TensorRT installation and environment variables.
*   **Capture Issues**: Try switching between DDA and OBS Hook in `config.ini`.
