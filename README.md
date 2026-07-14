# NeedAimBot - 2PC Architecture

**NeedAimBot** is a high-performance, AI-powered aim assistant designed for low-latency object detection and precise mouse control. This branch implements a **dual-PC architecture** for maximum performance and security by separating game execution from AI inference.

## Architecture Overview

This 2PC setup divides responsibilities between two computers connected via network:

```
┌─────────────────┐                    ┌──────────────────┐
│    Game PC      │    UDP Network     │  Inference PC    │
│                 │  Screen Data       │                  │
│ • Run Game      │ ─────────────────→ │ • AI Inference   │
│ • Capture       │                    │ • Mouse Control  │
│   Screen        │                    └────────┬─────────┘
└────────┬────────┘                             │
         │              Hardware                │
         │         (Makcu/Kmbox USB)            │
         └──────────────←───────────────────────┘
                    Direct Mouse Input
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

## Makcu USB Setup (Linux/Jetson)

For reliable Makcu connection on Linux, you need to set up udev rules for USB reset permissions:

```bash
# Create udev rule for Makcu USB access
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="1a86", MODE="0666"' | sudo tee /etc/udev/rules.d/99-makcu.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

This allows the application to perform USB device resets automatically, eliminating the need to physically unplug/replug the device between runs.

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

## Acquisition Flick: Warped Human-Trajectory Replay

When a target is freshly acquired, the aim snaps to it with a **flick** before the
PD controller takes over for steady tracking. Instead of synthesizing that flick
from a hand-tuned motor formula, this build **replays a real recorded human
stroke**, rigid-transformed onto the aim vector.

**Why:** against a strong mouse-dynamics classifier, a *synthesized* flick
(formula- or model-based) is highly detectable, because it can't reproduce the
full joint micro-structure of real motion. A real stroke rotated/scaled onto the
aim direction keeps that structure exactly, so it reads as human. In the companion
`mouse-bot-detector` study (and matching the SCRAP result, ACM AISec 2020), the
formula generator was detected at ~0.99 accuracy while warped replay sat at ~0.50
(chance) — indistinguishable from a real player.

**How it works** (`needaimbot/mouse/warped_replay.hpp`):
1. On fresh acquire, `generate()` gets the reach vector (distance `D`, direction `θ`).
2. It picks a recorded stroke whose distance is within `flick_distance_tolerance`
   of `D` (so the scale warp stays near 1× and doesn't distort speed/tremor).
3. Rotates it to `θ`, scales it to exactly `D`, adds small per-point jitter, and
   keeps the stroke's **real (irregular) timestamps**.
4. `FlickPlayback` replays it as incremental mouse deltas by wall-clock time.

Generation runs on the CPU (~1.6 µs/flick, off the per-frame path); the old GPU
flick kernel is gone. The DB is loaded once at startup (warmup), so the first
flick doesn't pay the parse cost.

### Trajectory database

The stroke DB is **compiled into the binary** (`needaimbot/mouse/flick_db_embedded.cpp`),
so **no `flick_trajectories.json` file is required** — the aimbot ships as a single
executable. It holds straight, low-lateral-deviation human strokes, canonicalized
to the origin→+x axis with their real distance and timestamps.

To **override** the embedded DB without recompiling, drop a `flick_trajectories.json`
next to the executable (resolved via the config path, then the exe directory);
`load_db` prefers a file when present and otherwise uses the embedded DB.

Regenerating the DB is a two-step, two-repo process:

1. In the **separate companion repo `mouse-bot-detector`** (not part of this repo),
   run `scripts/export_flick_db.py` to produce a `flick_trajectories.json` (filters
   to `path_efficiency ≥ 0.9`, lateral deviation ≤ 0.1× distance).
2. In **this** repo, embed it:
   `python needaimbot/mouse/gen_flick_db_embedded.py path/to/flick_trajectories.json`
   — this rewrites `needaimbot/mouse/flick_db_embedded.{cpp,hpp}`; rebuild to bake it in.

**For real deployment, build the DB from your *own* recorded strokes** — the bundled
DB is derived from a public dataset, which a defender doing near-duplicate / residual
matching can hold and match against.

### Config keys (`simple_config.json`)

```json
"flick_enabled": true,
"flick_replay_db_path": "flick_trajectories.json",
"flick_distance_tolerance": 0.15,
"flick_elastic_amp": 0.03,
"flick_elastic_modes": 3,
"flick_var_amp": 0.06,
"flick_min_reach": 5.0
```

> Each warped stroke gets a per-flick RANDOMIZED hybrid perturbation so a finite
> DB yields unlimited non-repeating flicks with no fixed signature:
> `flick_elastic_amp` is a smooth low-frequency bend (breaks near-duplicates
> without adding jerk) and `flick_var_amp` a step along a natural human-variation
> direction `mag*(A-B)` between two random DB strokes (makes the residual look
> like natural variation, so a residual-spectrum detector can't key on the bend).
> Both amplitudes are drawn uniformly in `[0, amp]` per flick. 0/0 = pure replay.
> Replaces the old `flick_position_jitter` (white noise, added jerk). Measured:
> single-move ~0.68, near-duplicates broken, residual detector evaded on a private
> pool (see `mouse-bot-detector/hybrid_replay.py`).

> Note: the previous PD-controller steady-state motor noise (`aim_sdn_k`,
> `aim_tremor_*`) has been removed — it added jitter to live tracking precision
> and is redundant now that the acquisition flick is genuine human motion.

## Planned Features

*   **Capture Card Support**: Frame data transmission via hardware capture card (HDMI/DisplayPort) for complete software isolation between Game PC and Inference PC
