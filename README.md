# NeedAimBot - 2PC Architecture

**NeedAimBot** is a high-performance, AI-powered aim assistant designed for low-latency object detection and precise mouse control. This branch implements a **dual-PC architecture** for maximum performance and security by separating game execution from AI inference.

## Architecture Overview

This 2PC setup divides responsibilities between two computers connected via network:

```
┌─────────────────┐    UDP Network    ┌──────────────────┐
│    Game PC      │ ←───────────────→ │  Inference PC    │
│                 │  Screen Data       │                  │
│ • Run Game      │ ─────────────────→ │ • AI Inference   │
│ • Capture       │  Detection Results │ • Mouse Control  │
│   Screen        │ ←───────────────── │   (Makcu/Kmbox)  │
└─────────────────┘                    └──────────────────┘
```

### Game PC
*   Runs the game at maximum performance
*   Captures screen region (256x256 default) using Desktop Duplication API
*   Sends captured frames to Inference PC via UDP
*   Lightweight processing only

### Inference PC
*   Receives screen data from Game PC
*   Runs YOLO-based TensorRT inference
*   Controls mouse via hardware (Makcu or Kmbox)
*   Handles all AI computations

## Key Advantages

*   **Reduced Detection Risk**: Game PC only runs lightweight screen capture code with no AI or cheat-related components, significantly minimizing the risk of anti-cheat detection
*   **Maximum Game Performance**: Game PC resources are fully dedicated to running the game
*   **GPU Isolation**: AI inference doesn't compete with game rendering for GPU resources
*   **Flexible Hardware**: Use dedicated inference hardware (Jetson, server GPU) separate from gaming setup

## Prerequisites

### Game PC
*   **OS**: Windows 10 or 11 (64-bit)
*   **Software**:
    *   [Visual Studio 2022](https://visualstudio.microsoft.com/vs/) (C++ Desktop Development)
    *   [CMake 3.20+](https://cmake.org/download/)
*   **Network**: Gigabit Ethernet recommended for low latency

### Inference PC
*   **OS**: Windows 10 or 11 (64-bit) / Linux (Ubuntu 20.04+) / NVIDIA Jetson
*   **GPU**: NVIDIA GeForce GTX 10-series or newer (RTX recommended)
*   **Drivers**: [Latest NVIDIA Driver](https://www.nvidia.com/en-us/drivers/) with CUDA 13.1+ support
*   **Software**:
    *   [Visual Studio 2022](https://visualstudio.microsoft.com/vs/) (C++ Desktop Development, Windows only)
    *   [CMake 3.20+](https://cmake.org/download/)
    *   [CUDA Toolkit 13.1](https://developer.download.nvidia.com/compute/cuda/13.1.0/network_installers/cuda_13.1.0_windows_network.exe)
    *   TensorRT 10.14.1.48 (included in `inference_pc/needaimbot/modules/`)
*   **Hardware**: Makcu relay or Kmbox device connected to Game PC
*   **Network**: Gigabit Ethernet recommended

## Installation

### 1. Clone the Repository
```bash
git clone --recursive -b 2pc https://github.com/needitem/needaimbot.git
cd needaimbot
```

### 2. Setup Hardware Mouse Device
Connect one of the following to your Game PC:
*   **Makcu Relay**: Arduino/ESP32-based USB relay
    *   See: [HID_Mouse Repository](https://github.com/needitem/HID_Mouse) for firmware
    *   Connect serially to Inference PC (direct or via USB-over-IP)
*   **Kmbox Net**: Network-based hardware mouse emulator
    *   Connect to same network as Inference PC

### 3. Build Game PC Application
On your Game PC:
```bash
cd game_pc
build_cmake.bat

# Or manually:
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```
Executable will be in `game_pc/build/bin/Release/`.

### 4. Build Inference PC Application
On your Inference PC:

**Windows:**
```bash
cd inference_pc
build_cmake.bat

# Or manually:
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

**Linux/Jetson:**
```bash
cd inference_pc
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Executable will be in `inference_pc/build/bin/Release/` (Windows) or `inference_pc/build/bin/` (Linux).

### 5. Prepare AI Model
*   Export your YOLO model to TensorRT `.engine` format using `inference_pc/engine_export/`
*   Place the `.engine` file in `inference_pc/` directory
*   Update `inference_pc/config.ini` with the model filename

## Configuration

### Game PC Configuration (`game_pc/config.ini`)

```ini
[Network]
InferenceIP=192.168.1.100    # IP address of Inference PC
SendPort=5007                 # UDP port to send frames

[Capture]
CaptureX=0                    # Top-left X coordinate of capture region
CaptureY=0                    # Top-left Y coordinate of capture region
CaptureWidth=256              # Capture width (must match model input)
CaptureHeight=256             # Capture height (must match model input)

[Performance]
TargetFPS=90                  # Target capture/send FPS
```

### Inference PC Configuration (`inference_pc/config.ini`)

```ini
[Inference]
ModelPath=your_model.engine   # TensorRT engine file name
Port=5007                      # UDP port to receive frames (must match Game PC)
GamePcIP=                      # Optional: Set to Game PC IP for validation

[Mouse]
# Mouse control device selection
InputMethod=MAKCU              # Options: MAKCU, KMBOX

[MAKCU]
# For Makcu relay device
SerialPort=COM3                # Serial port or device path (/dev/ttyUSB0 on Linux)
BaudRate=115200

[KMBOX]
# For Kmbox device
IP=192.168.1.200               # Kmbox device IP
Port=1408                      # Kmbox port
UUID=YOUR_UUID                 # Kmbox UUID (from device)
```

## Usage

### 1. Network Setup
*   Connect both PCs to the same network (direct Ethernet cable or via switch/router)
*   Ensure firewall allows UDP traffic on the configured port (default: 5007)
*   Note the Inference PC's IP address

### 2. Start Inference PC First
On the Inference PC:
```bash
cd inference_pc
./needaimbot.exe  # Windows
./needaimbot      # Linux
```
The application will wait for incoming frames.

### 3. Start Game PC
On the Game PC:
```bash
cd game_pc
./game_pc.exe
```
*   Run as Administrator for screen capture permissions
*   The application will start capturing and sending frames

### 4. Controls (Game PC)
*   `F2`: Exit application
*   `F3`: Pause/Resume capture
*   `F4`: Reload configuration

## Network Performance Tips

*   Use **gigabit Ethernet** for best latency (avoid Wi-Fi if possible)
*   Use a **direct cable connection** between PCs for lowest latency
*   Set both network adapters to maximum performance in Windows power settings
*   Disable energy-saving features on network adapters
*   Use `TargetFPS` to balance between latency and network load

## Troubleshooting

### Game PC Issues
*   **"Failed to initialize capture"**: Run as Administrator
*   **"Cannot connect to Inference PC"**: Check network configuration and firewall
*   **Low FPS**: Reduce `CaptureWidth/Height` or `TargetFPS`

### Inference PC Issues
*   **"No GPU devices with CUDA support"**: Install CUDA Toolkit and update drivers
*   **"Failed to load model"**: Verify model path and TensorRT version compatibility
*   **"Serial port not found"**: Check Makcu connection and COM port in config
*   **"Kmbox connection failed"**: Verify Kmbox IP, port, and UUID

### Network Issues
*   **"Timeout receiving frames"**: Check network connection and firewall settings
*   **High latency**: Switch to wired connection, check network bandwidth usage
*   **Packet loss**: Reduce `TargetFPS` or check network quality

## Advanced Features

*   **Packet Fragmentation**: Automatically handles large frames exceeding UDP limits
*   **Display Window**: Inference PC can show received frames and detection results (Linux/Jetson)
*   **Cross-Platform**: Supports Windows Game PC → Linux/Jetson Inference PC

## Planned Features

*   **Capture Card Support**: Frame data transmission via hardware capture card (HDMI/DisplayPort) for complete software isolation between Game PC and Inference PC
