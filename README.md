# needaimbot

An ultra-low latency AI targeting assistant for Windows that keeps the entire capture → inference → mouse control loop on the GPU. This README focuses on how to install, configure, and safely operate the project.

## 📦 What You Get
- Desktop Duplication (DDA) capture streamed directly into CUDA without CPU copies.
- TensorRT-based inference with configurable models and precision profiles.
- Multiple mouse injection backends (Win32, Arduino, Logitech G-Hub, KMBox, MakCU, Razer) with automatic fallback.
- Overlay and keyboard listener for runtime control plus INI-driven profiles stored next to the executable.

## 🖥️ Requirements
| Component | Minimum | Notes |
|-----------|---------|-------|
| OS | Windows 10/11 64-bit | Desktop Duplication is required for capture. |
| GPU | NVIDIA RTX-class with CUDA 12.x drivers | TensorRT engines must match the installed CUDA/CuDNN toolchain. |
| SDKs | CUDA Toolkit 12.8, cuDNN 9.7.1, TensorRT 10.8.0.43, Visual Studio 2022 (MSVC v143, Desktop C++ workload) | `build.bat` uses MSBuild from Visual Studio 2022. |
| Runtime | Microsoft Visual C++ Redistributable (x64) | Required for MSVC-built binaries. |

## ⚠️ Install the NVIDIA inference stack (one-time)
1. **CUDA Toolkit 12.8.0** – Download the Windows installer from the [CUDA 12.8.0 archive](https://developer.nvidia.com/cuda-12-8-0-download-archive) and install the *Toolkit* + *Driver* components. Accept the default location (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`) so the project file can find the headers and libraries automatically.
2. **cuDNN 9.7.1 for CUDA 12.8** – Grab the Windows ZIP from the [cuDNN download portal](https://developer.nvidia.com/cudnn-downloads) (requires NVIDIA developer login). Extract it under `C:\Program Files\NVIDIA\CUDNN\v9.7\` so that the directory layout matches what the MSBuild project expects:
   ```text
   C:\Program Files\NVIDIA\CUDNN\v9.7\
       bin\12.8\cudnn64_9.dll
       include\12.8\cudnn*.h
       lib\12.8\cudnn.lib
   ```
3. **TensorRT 10.8.0.43** – Download the `TensorRT-10.8.0.43.Windows10.win10.cuda-12.6.zip` package from the [TensorRT 10.x downloads](https://developer.nvidia.com/tensorrt/download/10x). Unzip it into `needaimbot/modules/TensorRT-10.8.0.43/` so the repository keeps all required headers and libraries under version control:
   ```text
   needaimbot/
     modules/
       TensorRT-10.8.0.43/
         bin/       # contains nvinfer_10.dll, nvonnxparser_10.dll, etc.
         include/   # headers used during compilation
         lib/       # .lib import libraries referenced by the project
         samples/   # optional, but keep for compatibility
   ```
   After building, copy the DLLs from `modules/TensorRT-10.8.0.43/bin/` (and `CUDNN\v9.7\bin\12.8\`) next to `needaimbot.exe` or add them to your `PATH` so the runtime can locate them.

## ⬇️ Download & Build
1. **Clone the repository**
   ```powershell
   git clone https://github.com/needitem/needaimbot.git
   cd needaimbot
   ```
2. **Install prerequisites** (CUDA 12.8, cuDNN 9.7.1, TensorRT 10.8, Visual Studio 2022).
3. **Place your TensorRT engine** in `needaimbot/models/`. The default configuration expects `sunxds_0.5.6.engine` but any `.engine` file can be selected through the config.
4. **Build** using the supplied script or Visual Studio:
   ```powershell
   build.bat  # invokes MSBuild in Release x64
   ```
   or open `needaimbot.sln`, choose *Release | x64*, and build the `needaimbot` project.

## ▶️ First Launch
1. Copy the built `needaimbot.exe`, accompanying DLLs, and your `.engine` model into the same directory.
2. Run `needaimbot.exe` as Administrator so the selected input driver can inject mouse events.
3. On first launch the app writes `config.ini` (and `Default.ini`) next to the executable with sane defaults.
4. Use the default hotkeys below to verify that capture and mouse control work.

### Default Hotkeys
| Action | Default Binding |
|--------|-----------------|
| Toggle targeting | Right Mouse Button | 
| Pause/Resume | F3 |
| Reload config | F4 |
| Toggle overlay | Home |
| Emergency stop / Exit | F2 |
| Auto shoot | Left Mouse Button |
| Disable upward aim | None (unbind) |
These bindings are editable in the `[Buttons]` section of any profile INI file.

## ⚙️ Configuration & Profiles
- **Location** – All `.ini` files live beside the executable. `config.ini` stores the active profile name; individual profiles use `<ProfileName>.ini`.
- **Profiles** – Swap profiles via the overlay or edit `config.ini` to set `active_profile`. Profiles can be listed, saved, and deleted programmatically through `Config` helpers.
- **Auto-generation** – Missing files are regenerated with defaults, so you can delete an `.ini` to reset settings.

### Frequently Tuned Parameters
| Section | Key | Purpose & Tips |
|---------|-----|----------------|
| `[Capture]` | `detection_resolution` | Lower values (e.g., 256) increase FPS at the cost of precision. |
| `[Target]` | `body_y_offset`, `head_y_offset` | Vertical aim offsets for body vs. head tracking; tweak per game sensitivity. |
| `[PDController]` | `pd_kp_x`, `pd_kp_y` | Proportional controller gains that control mouse aggressiveness; start at 0.4 and adjust slowly. |
| `[Mouse]` | `input_method` | Choose `WIN32`, `ARDUINO`, `GHUB`, `KMBOX`, `MAKCU`, or `RAZER`. The app falls back to Win32 if the requested driver fails. |
| `[AI]` | `ai_model`, `confidence_threshold`, `max_detections` | Controls which TensorRT engine runs and how many detections are considered per frame. |
| `[WeaponProfiles]` | Weapon-specific recoil tables | Maintain per-weapon recoil compensation with scope multipliers and timing offsets. |

### Parameter Workflow Tips
1. Edit the active profile (`Default.ini` by default) while the app is paused.
2. Press `F4` (reload config) to apply changes instantly without restarting.
3. Use separate profiles per game or weapon set; switching profiles rewrites `config.ini` automatically.

## 🖱️ Input Driver Notes
- **Win32** – No extra hardware; works for most setups but is easiest to detect.
- **Arduino / MakCU** – Requires a serial device flashed with the companion firmware. Configure COM port and baud rate in `[Arduino]` or `[MAKCU]` blocks.
- **KMBox** – Needs LAN connectivity; supply IP/port/MAC. Initialization errors fall back to Win32 with a log message.
- **Logitech G-Hub** – Keep G-Hub running and select `GHUB` input method.
- **Razer** – Uses the bundled `rzctl.dll`; Administrator rights required.

## 🧪 Verifying Performance
- The overlay (Home key) shows FPS and capture status when `show_window` and `show_fps` are enabled.
- CUDA Graph execution keeps latency under 3 ms when the GPU is not saturated; profile with Nsight Systems or CUDA events if spikes appear.

## ❗ Safety & Troubleshooting
- **Respect Game TOS** – Automated aiming may violate game or platform rules. Use on private environments at your own risk.
- **Capture Fails** – Ensure you are on a physical monitor (no remote desktop) and Desktop Duplication is supported.
- **Model Not Found** – The console logs when the requested `.engine` is missing and falls back to the first file under `models/`.
- **Driver Fallbacks** – Check the console for `[Mouse]` warnings if a specialized input driver failed; the app will still run with Win32 control.
- **Reset Config** – Delete the profile `.ini` or call “Reset to defaults” in the overlay; the app regenerates defaults on next launch.

## 🤝 Contributing
1. Fork the repository and create feature branches locally.
2. Follow RAII memory patterns and keep GPU-critical paths free of blocking calls (see existing CUDA modules for guidance).
3. Add or update profile defaults when introducing new config knobs so fresh installs stay functional.
4. Submit a PR with profiling evidence for performance-sensitive changes.

## 📄 License
needaimbot is released under the MIT License. See [LICENSE](LICENSE) for details.
