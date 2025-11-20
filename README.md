# NeedAimBot

NeedAimBot is a high-performance, low-latency AI targeting assistant designed for Windows. It leverages NVIDIA's TensorRT for real-time object detection and utilizes a direct DirectX-to-CUDA capture pipeline to minimize input lag, ensuring the entire capture-inference-control loop remains on the GPU as much as possible.

## Ecosystem

This project is part of a larger suite of tools designed to work together for a robust and undetectable targeting solution:

*   **NeedAimBot**: The core application that handles screen capture, inference, and input injection.
*   **EngineExport**: A utility to convert ONNX AI models into optimized TensorRT engines (`.engine`) specifically tuned for your GPU architecture.
*   **MakcuFlasher**: A tool for flashing custom firmware onto Arduino-based input devices (e.g., Leonardo, Micro) to spoof hardware IDs and bypass anti-cheat detection.
*   **MakcuRelay**: A network relay application that allows NeedAimBot to run on one PC while sending mouse commands to a second PC via a dedicated hardware device (Dual-PC Setup).

---

## Prerequisites

Before installing, ensure your system meets the following requirements:

*   **OS**: Windows 10 or 11 (64-bit).
*   **GPU**: NVIDIA RTX 20 series or higher recommended (RTX 30/40 series preferred for lower latency).
*   **Drivers**: Latest NVIDIA Game Ready Driver.
*   **Software**:
    *   CUDA Toolkit 12.x
    *   cuDNN 9.x
    *   TensorRT 10.x
    *   Visual Studio 2022 (Desktop C++ Workload)

---

## Installation & Setup Guide

### Step 1: Prepare the AI Model (EngineExport)

NeedAimBot requires a TensorRT engine file (`.engine`) specific to your GPU. You cannot simply download a generic engine; it must be built on your machine to ensure compatibility and maximum performance.

1.  Navigate to the **EngineExport** repository.
2.  Install the required Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Place your trained ONNX model (e.g., `yolov8.onnx`) in the `models` folder.
4.  Run the export script to generate the engine:
    ```bash
    python export.py --model your_model.onnx --precision fp16
    ```
    *   *Note: Use `--precision int8` if you have a calibration dataset for even faster inference.*
5.  Once complete, copy the generated `.engine` file to the `needaimbot/models/` directory.

### Step 2: Setup Input Hardware (MakcuFlasher)

For the safest experience, use an external Arduino device to simulate mouse input.

1.  Connect your Arduino Leonardo or Micro to your PC.
2.  Navigate to the **MakcuFlasher** directory.
3.  Run the flashing script:
    ```bash
    python makcu_flash.py
    ```
4.  Follow the prompts to select your device's **COM Port** and flash the firmware.
5.  Once finished, note the COM port; you will need it for the configuration step.

### Step 3: Dual PC Setup (MakcuRelay) - Optional

If you are using a two-PC setup (one for gaming, one for AI) to completely isolate the cheat execution:

1.  On the **Gaming PC**, connect the Arduino device.
2.  Navigate to the **MakcuRelay** directory.
3.  Run the relay script:
    ```bash
    python MakcuRelay.py
    ```
4.  On the **AI PC** (running NeedAimBot), you will configure `config.ini` to point to the Gaming PC's IP address.

### Step 4: Build and Run NeedAimBot

1.  Open `needaimbot.sln` in Visual Studio 2022.
2.  Select **Release** configuration and **x64** platform.
3.  Build the solution (Ctrl+Shift+B).
4.  Navigate to the output directory (`x64/Release`).
5.  Ensure your `.engine` file is in the `models` subdirectory.
6.  Run `needaimbot.exe` as **Administrator** (required for input injection and screen capture).

---

## Configuration Manual

The application is controlled via `config.ini`, which is generated on the first run. Below is a detailed explanation of every setting.

### [Capture]
Settings related to screen capture performance and area.

*   `detection_resolution`: The resolution to resize the captured frame to before inference (e.g., `320`, `416`, `640`). Lower values improve FPS but reduce long-range accuracy.
*   `monitor_idx`: The index of the monitor to capture (0 = primary).
*   `capture_borders`: If `true`, captures the entire screen including borders.
*   `capture_cursor`: If `true`, includes the mouse cursor in the capture (useful for debugging).

### [Target]
Settings for aiming logic and offsets.

*   `body_y_offset`: Vertical offset (0.0 - 1.0) to aim at the body relative to the detection box height.
*   `head_y_offset`: Vertical offset to aim at the head.
*   `offset_step`: How much to adjust the offset when using runtime hotkeys.
*   `enable_target_lock`: If `true`, the aimbot will stick to the current target until it disappears or the key is released.
*   `ignore_third_person`: Prevents aiming at your own character in third-person games (requires specific class training).
*   `auto_aim`: Enables aiming automatically without holding a key (Use with caution).
*   `auto_shoot`: Automatically fires when the crosshair is over a target.
*   `enable_rapidfire`: Toggles rapid-fire mode for semi-auto weapons.
*   `rapidfire_cps`: Clicks per second for rapid fire.

### [Mouse]
Input injection and humanization settings.

*   `input_method`: The driver to use for mouse movement.
    *   `WIN32`: Standard Windows API (detected by most anti-cheats).
    *   `ARDUINO`: Uses a connected Arduino device via serial.
    *   `GHUB`: Uses Logitech G-Hub driver (requires G-Hub installed).
    *   `KMBOX`: Uses KMBox hardware.
    *   `MAKCU`: Uses MakcuRelay for dual-PC setups.
*   `norecoil_ms`: Duration of recoil compensation in milliseconds.
*   `min_movement_threshold`: Minimum pixel distance to move. Prevents micro-jitter when the crosshair is very close to the target.
*   `easynorecoil`: Simple vertical recoil compensation.
*   `easynorecoilstrength`: Strength of the simple recoil compensation.

### [PDController]
PID controller settings for smooth mouse movement.

*   `pd_kp_x`, `pd_kp_y`: Proportional gain. Higher values make the aim snap faster but may overshoot.
*   `pid_ki_x`, `pid_ki_y`: Integral gain. Helps correct small steady-state errors.
*   `pid_kd_x`, `pid_kd_y`: Derivative gain. Dampens the movement to prevent oscillation.

### [Arduino] / [MAKCU] / [KMBOX]
Hardware-specific connection settings.

*   `arduino_port`: COM port for the Arduino (e.g., `COM3`).
*   `arduino_baudrate`: Serial speed (default: `115200` or `2000000`).
*   `makcu_remote_ip`: IP address of the PC running MakcuRelay (for Dual-PC).
*   `makcu_remote_port`: UDP port for MakcuRelay.
*   `kmbox_ip`, `kmbox_port`, `kmbox_mac`: Connection details for KMBox devices.

### [AI]
Inference engine settings.

*   `ai_model`: Filename of the TensorRT engine to load (e.g., `sunxds_0.5.6.engine`).
*   `confidence_threshold`: Minimum confidence (0.0 - 1.0) to consider a detection valid.
*   `max_detections`: Maximum number of targets to process per frame.
*   `postprocess`: Post-processing method (e.g., `yolo12`).

### [Buttons]
Hotkey bindings. You can use standard key names (e.g., `F1`, `RightMouseButton`, `LeftAlt`, `Home`).

*   `button_targeting`: Key to hold for aimbot activation.
*   `button_exit`: Panic key to close the application immediately.
*   `button_pause`: Toggles the aimbot on/off.
*   `button_reload_config`: Reloads `config.ini` without restarting.
*   `button_open_overlay`: Toggles the visual overlay.
*   `button_auto_shoot`: Key to toggle auto-shoot.

### [Overlay]
Visual settings for the debug overlay.

*   `show_window`: Enables the overlay window.
*   `show_fps`: Displays current FPS.
*   `overlay_opacity`: Opacity of the overlay background (0-255).
*   `always_on_top`: Keeps the overlay above the game window.

---

## Troubleshooting

*   **FPS Drops**: Ensure `detection_resolution` is not too high (recommended: 320-480). Check if `input_method` is causing blocking calls.
*   **No Detections**: Verify the `.engine` file matches your TensorRT version and GPU. Check `confidence_threshold`.
*   **Mouse Not Moving**: Run as Administrator. Verify COM port for hardware methods.
*   **Overlay Not Visible**: Ensure the game is in "Borderless Windowed" mode, or enable `always_on_top`.

---

## Disclaimer

This software is for educational and research purposes only. Using this software in multiplayer games may violate their Terms of Service and result in account bans. The authors are not responsible for any consequences resulting from the use of this software. Use at your own risk.
