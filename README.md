<div align="center">

# needaimbot C++

</div>

## 🎯 Overview

**needaimbot** is a high-performance C++ AI-powered aiming assistance tool that utilizes deep learning models and advanced computer vision techniques for real-time target detection and tracking. Built with cutting-edge technologies including TensorRT for GPU acceleration, OpenCV for image processing, and CUDA for parallel computing.

### Key Highlights
- **Real-time AI target detection** with TensorRT optimization
- **Advanced predictive tracking** using Kalman filters and PID controllers
- **Multiple capture methods** including optical flow and hardware acceleration
- **Extensive input device support** (Logitech G-Hub, Razer, KMBox, Serial)
- **Customizable overlay interface** with real-time configuration
- **GPU-accelerated processing** for minimal latency

- **This project is actively being developed thanks to the people who support on [Boosty](https://boosty.to/sunone) and [Patreon](https://www.patreon.com/sunone). By providing active support, you receive enhanced AI models.**

> **⚠️ WARNING:** TensorRT version 10 does not support the Pascal architecture (10 series graphics cards). Use only with GPUs of at least the 20 series.

## How to Use
1. **Download CUDA**
	- Download and install [CUDA 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive).

2. **Download the Latest Release**  
	- Download from [Mega.nz](https://mega.nz/file/0PVxDRLL#b62nQhHkjm4iOIf0i8_yLuX1Gop5AomjWONGs-yfaKk)

3. **Unpack Aimbot**
   - Extract the contents of the Aimbot.

4. **First Launch and Model Export**
	- Run `ai.exe` and wait until the standard `.onnx` model is exported, usually taking no more than five minutes.
	- To export another model, simply place it in `.onnx` format in the `models` folder. Then, in the AI tab (press `HOME` to open overlay), select this model, and it will be exported automatically.

5. **Settings**
	- After successfully exporting the model, you can configure the program.
	- All settings are available in the overlay (default key is `Home`).
	- A list of settings can be found in the [config documentation](https://github.com/SunOner/sunone_aimbot_docs/blob/main/config/config_cpp.md).

6. **Controls**
	- **Right Mouse Button:** Aim at the detected target.
	- **F2:** Exit the program.
	- **F3:** Activate pause for aiming.
	- **F4:** Reload config.
	- **Home:** Show overlay.

## ✨ Features

### Core Functionality
- **🎯 AI-Powered Target Detection**: Advanced neural networks for precise target identification
- **🔄 Real-Time Tracking**: Kalman filtering and predictive algorithms for smooth target following
- **⚡ GPU Acceleration**: CUDA and TensorRT optimization for minimal latency
- **🎮 Multiple Input Methods**: Support for various mouse drivers and hardware interfaces
- **📊 Optical Flow Integration**: Advanced motion detection and compensation
- **🎛️ Live Configuration**: Real-time parameter adjustment through overlay interface

### AI & Computer Vision
- **TensorRT Model Optimization**: Automatic conversion and optimization of ONNX models
- **Dynamic Shape Support**: Flexible input dimensions for various model architectures
- **Multi-Model Support**: Easy switching between different AI models
- **GPU Memory Management**: Efficient VRAM usage and allocation
- **Post-Processing Pipeline**: Advanced filtering and scoring algorithms

### Input & Control Systems
- **Logitech G-Hub Integration**: Direct communication with Logitech gaming peripherals
- **Razer Device Support**: Native Razer device compatibility
- **KMBox Hardware Support**: Professional-grade hardware input simulation
- **Serial Communication**: Custom hardware integration capabilities
- **PID Control System**: Advanced proportional-integral-derivative control for smooth aiming

### Capture & Processing
- **Desktop Duplication API**: High-performance screen capture
- **Game-Specific Capture**: Optimized capture methods for gaming applications
- **Optical Flow Processing**: Motion-based target tracking and prediction
- **Multi-Monitor Support**: Full multi-display configuration support
- **HSV Color Filtering**: Advanced color-based target filtering

### User Interface
- **ImGui Overlay**: Modern, responsive configuration interface
- **Real-Time Statistics**: Live performance metrics and debugging information
- **Profile Management**: Save and load different configuration profiles
- **Visual Debugging**: Target visualization and tracking display
- **Hotkey Support**: Customizable keyboard shortcuts for all functions

## 🏗️ Architecture & Technical Details

### System Requirements
- **Operating System**: Windows 10/11 (x64)
- **GPU**: NVIDIA RTX 20 series or newer (CUDA Compute Capability 7.5+)
- **CUDA**: Version 12.8
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB free space for models and dependencies

### Core Technologies
- **C++17**: Modern C++ standard with performance optimizations
- **CUDA 12.8**: GPU acceleration and parallel computing
- **TensorRT 10.8**: Deep learning inference optimization
- **OpenCV 4.10**: Computer vision and image processing
- **ImGui**: Immediate mode GUI framework
- **Eigen**: Linear algebra and mathematical operations

### Performance Characteristics
- **Inference Time**: <5ms per frame (RTX 3080)
- **End-to-End Latency**: <10ms total system latency
- **Memory Usage**: ~2GB VRAM, ~500MB RAM
- **CPU Usage**: <10% on modern processors
- **Frame Rate**: Supports up to 240 FPS processing

## 🛠 Build the Project from Source

> **ℹ️ NOTE:** This guide is intended for advanced users. If you encounter errors while building the modules, please report them on the [Discord server](https://discord.gg/sunone).

1. **Install Visual Studio 2022 Community**  
   Download and install from the [official website](https://visualstudio.microsoft.com/vs/community/).

2. **Install Windows SDK**  
   Ensure you have Windows SDK version **10.0.26100.0** installed.

3. **Install CUDA and cuDNN**  
   - **CUDA 12.8**  
     Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).
   - **cuDNN 9.7.1**  
     Available on the [NVIDIA cuDNN archive](https://developer.nvidia.com/cudnn-downloads) website.

4. **Set Up Project Structure**  
   Create a folder named `modules` in the directory `needaimbot/needaimbot/modules`.

5. **Build OpenCV with CUDA Support (Maximum Performance)**
	- Download and install [CMake](https://cmake.org/) and [CUDA 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive).
	- Download [OpenCV 4.12](https://github.com/opencv/opencv) (latest version for enhanced performance).
	- Download [OpenCV Contrib 4.12](https://github.com/opencv/opencv_contrib/tags).
	- Create new directories: `needaimbot/needaimbot/modules/opencv/` and `needaimbot/modules/opencv/build`.
	- Extract `opencv-4.12.0` to `needaimbot/needaimbot/modules/opencv/opencv-4.12.0` and `opencv_contrib-4.12.0` to `needaimbot/modules/opencv/opencv_contrib-4.12.0`.
	- Extract cuDNN to `needaimbot/needaimbot/modules/cudnn`.
	- Open CMake and set the source code location to `needaimbot/modules/opencv/opencv-4.12.0`.
	- Set the build directory to `needaimbot/needaimbot/modules/opencv/build`.
	- Click `Configure`.
	- (Some options will appear after the next configuration application. For example, to configure the CUDNN_LIBRARY paths, you first need to activate the WITH_CUDA option and click configure.)
	
	**Core Performance Settings:**
		- `CMAKE_BUILD_TYPE` = `Release`
		- `CMAKE_CXX_FLAGS_RELEASE` = `/Ox /Ob2 /Ot /Oy /GT /GL /Gw /Gy /fp:fast /arch:AVX2 /favor:INTEL64 /MP /bigobj`
		- `CMAKE_C_FLAGS_RELEASE` = `/Ox /Ob2 /Ot /Oy /GT /GL /Gw /Gy /fp:fast /arch:AVX2 /favor:INTEL64 /MP`
		- `CMAKE_EXE_LINKER_FLAGS_RELEASE` = `/LTCG /OPT:REF /OPT:ICF /INCREMENTAL:NO`
		- `CMAKE_SHARED_LINKER_FLAGS_RELEASE` = `/LTCG /OPT:REF /OPT:ICF /INCREMENTAL:NO`
	
	**CUDA & GPU Acceleration:**
		- `WITH_CUDA` = `ON`
		- `WITH_CUBLAS` = `ON`
		- `ENABLE_FAST_MATH` = `ON`
		- `CUDA_FAST_MATH` = `ON`
		- `WITH_CUDNN` = `ON`
		- `CUDNN_LIBRARY` = `<full path>needaimbot/needaimbot/modules/cudnn/lib/x64/cudnn.lib`
		- `CUDNN_INCLUDE_DIR` = `<full path>needaimbot/needaimbot/modules/cudnn/include`
		- `CUDA_ARCH_BIN` = Visit the [CUDA Wiki](https://en.wikipedia.org/wiki/CUDA) to find your Nvidia GPU architecture. For example, for `RTX 3080-TI`, enter `8.6`.
		- `OPENCV_DNN_CUDA` = `ON`
	
	**CPU Optimization:**
		- `CPU_BASELINE` = `AVX2`
		- `CPU_DISPATCH` = `AVX2,FP16,AVX512_SKX`
		- `WITH_IPP` = `ON` (Intel Performance Primitives)
		- `WITH_TBB` = `ON` (Threading Building Blocks)
		- `WITH_OPENMP` = `ON`
		- `WITH_EIGEN` = `ON`
		- `WITH_LAPACK` = `ON`
	
	**Module Configuration:**
		- `OPENCV_EXTRA_MODULES_PATH` = `<full path>needaimbot/needaimbot/modules/opencv/opencv_contrib-4.12.0/modules`
		- `BUILD_opencv_world` = `ON`
		- `BUILD_SHARED_LIBS` = `ON`
		- `ENABLE_PRECOMPILED_HEADERS` = `ON`
		
	**Disable Unnecessary Features:**
		- `BUILD_opencv_apps` = `OFF`
		- `BUILD_opencv_java` = `OFF`
		- `BUILD_opencv_js` = `OFF`
		- `BUILD_opencv_python2` = `OFF`
		- `BUILD_TESTS` = `OFF`
		- `BUILD_PERF_TESTS` = `OFF`
		- `BUILD_EXAMPLES` = `OFF`
		- `BUILD_DOCS` = `OFF`
		- `WITH_NVCUVENC` = `OFF`
		- `WITH_NVCUVID` = `OFF`
		- `WITH_GSTREAMER` = `OFF`
		- `WITH_GTK` = `OFF`
		- `WITH_QT` = `OFF`
	
   - Click `Configure` again and ensure that all optimization flags are properly set.
   - Click `Generate` to build the C++ solution.
   - Close CMake and open `needaimbot/modules/opencv/build/OpenCV.sln`, or click `Open Project` in cmake.
   - Switch the build configuration to `x64` and `Release`.
   - Open the `CMakeTargets` folder in the solution.
   - Right-click on `ALL_BUILD` and select `Build`. (Building with maximum optimization can take 2-4 hours.)
   - After building, right-click on `INSTALL` and select `Build`.
   - Verify the built files exist in the following folders:
     - `needaimbot/needaimbot/modules/opencv/build/install/include/opencv2` - Contains `.hpp` and `.h` files.
     - `needaimbot/needaimbot/modules/opencv/build/install/x64/vc16/bin` - Contains `.dll` files.
     - `needaimbot/needaimbot/modules/opencv/build/install/x64/vc16/lib` - Contains `.lib` files.
	
	**Performance Notes:**
	- Use `/arch:AVX512` instead of `/arch:AVX2` if your CPU supports AVX-512 (Intel Skylake-X or newer)
	- The `/Ox /Ot /GL /LTCG` combination provides maximum speed optimization
	- Building with these settings will result in larger binaries but significantly faster execution

6. **Download Required Libraries**  
	- [simpleIni](https://github.com/brofield/simpleini/blob/master/SimpleIni.h)
	- [TensorRT-10.8.0.43](https://developer.nvidia.com/tensorrt/download/10x)
	- [GLWF Windows pre-compiled binaries](https://www.glfw.org/download.html)
	- [Eigen](https://gitlab.com/libeigen/eigen/-/releases) (Download the latest stable release)
	
7. **Extract Libraries**  
	Place the downloaded libraries into the respective directories:
	- `SimpleIni.h` -> `needaimbot/needaimbot/modules/SimpleIni.h`
	- `TensorRT-10.8.0.43` -> `needaimbot/needaimbot/modules/TensorRT-10.8.0.43`
	- `GLWF` -> `needaimbot/needaimbot/modules/glfw-3.4.bin.WIN64`
	- `Eigen` -> `needaimbot/needaimbot/modules/eigen`
	  - Extract the Eigen archive and rename the folder to `include`
	  - Verify that the header files are located at `needaimbot/needaimbot/modules/eigen/include/Eigen/`

	**Note**: Serial communication for Arduino input method now uses Windows Native Serial API (no external library required)
   
8. **Configure Project Settings**
	- Open the project in Visual Studio.
	- Ensure all library paths are correctly set in **Project Properties** under **Library Directories**.
	- Go to NuGet packages and install `Microsoft.Windows.CppWinRT`.

9. **Verify CUDA Integration**
	- Right-click on the project in Visual Studio.
	- Navigate to **Build Dependencies** > **Build Customizations**.
	- Ensure that **CUDA 12.8** (.targets, .props) is included.

10. **Build the Project**
    - Switch the build configuration to **Release**.
    - Build the project by selecting **Build** > **Build Solution**.

## 🐛 Troubleshooting

### Common Issues

#### Installation Problems
- **CUDA Not Found**: Ensure CUDA 12.8 is properly installed and added to PATH
- **TensorRT Installation**: Verify TensorRT 10.8 is extracted to the correct modules directory
- **Visual Studio Build Errors**: Check that Windows SDK 10.0.26100.0 is installed
- **OpenCV Build Failures**: Ensure all paths in CMake configuration are absolute and correct

#### Runtime Issues
- **Model Loading Errors**: 
  - Check that .onnx models are in the `models` folder
  - Verify model compatibility with TensorRT 10
  - Ensure sufficient VRAM is available
- **Low Performance**: 
  - Enable GPU acceleration in overlay settings
  - Check CUDA and TensorRT installation
  - Monitor GPU utilization and temperature
- **Capture Issues**:
  - Run as administrator for desktop capture
  - Check display scaling settings
  - Verify capture region configuration

#### Device Connectivity
- **Mouse Input Not Working**:
  - Install appropriate device drivers (G-Hub, Razer Synapse)
  - Check USB connection and device recognition
  - Verify driver selection in overlay settings
- **Serial Device Issues**:
  - Check COM port settings and availability
  - Verify baud rate and communication parameters
  - Test with device manager

### Performance Optimization
- **Reduce Input Lag**: Lower capture resolution, optimize model size
- **Improve Accuracy**: Adjust confidence thresholds, retrain models
- **Memory Management**: Monitor VRAM usage, close unnecessary applications

### Getting Help
- **Discord Community**: [Join our Discord](https://discord.gg/sunone) for real-time support
- **GitHub Issues**: Report bugs and request features
- **Documentation**: Check the [config documentation](https://github.com/SunOner/sunone_aimbot_docs/blob/main/config/config_cpp.md)

## 🧠 AI Model Management

### Export PyTorch models from Python with dynamic shapes
- .pt -> .onnx
	```bash
	pip install ultralytics -U
	yolo export model=sunxds_0.5.6.pt format=onnx dynamic=true simplify=true
	```
- For .onnx -> .engine model export use overlay export tab in overlay.

### Model Optimization Tips
- **Dynamic Shapes**: Enable for flexible input sizes but may reduce performance
- **Precision**: Use FP16 for better performance on compatible GPUs
- **Batch Size**: Optimize batch size based on available VRAM
- **Model Pruning**: Remove unnecessary layers to reduce model size

## 📚 FAQ

### General Questions
**Q: Is this software safe to use?**
A: Yes, the software is open-source and can be built from source. All code is available for inspection.

**Q: What games are supported?**
A: The software works with any game that can be captured via screen capture APIs.

**Q: Do I need a specific GPU?**
A: Yes, NVIDIA RTX 20 series or newer is required due to TensorRT dependencies.

### Technical Questions
**Q: Can I use my own AI models?**
A: Yes, place your .onnx model in the models folder and select it in the overlay.

**Q: How do I improve detection accuracy?**
A: Adjust confidence thresholds, train models on game-specific data, or use higher resolution models.

**Q: Why is my performance low?**
A: Check GPU utilization, reduce capture resolution, or optimize model complexity.

## 🤝 Contributing

We welcome contributions to improve needaimbot! Here's how you can help:

### Development Guidelines
- **Code Style**: Follow existing C++ conventions and naming patterns
- **Testing**: Test your changes thoroughly across different configurations
- **Documentation**: Update documentation for any new features or changes
- **Performance**: Maintain or improve existing performance characteristics

### How to Contribute
1. **Fork the Repository**: Create your own fork of the project
2. **Create a Branch**: Make your changes in a feature branch
3. **Test Thoroughly**: Ensure your changes work across different systems
4. **Submit a Pull Request**: Describe your changes and their benefits
5. **Code Review**: Participate in the review process

### Areas for Contribution
- **Model Optimization**: Improve AI model accuracy and performance
- **Device Support**: Add support for new input devices
- **UI Improvements**: Enhance the overlay interface and user experience
- **Documentation**: Improve guides, tutorials, and code documentation
- **Testing**: Create comprehensive test suites and performance benchmarks

### Community Support
- **Discord**: Join our community for discussions and support
- **Issues**: Report bugs and request features on GitHub
- **Sponsorship**: Support development through [Boosty](https://boosty.to/sunone) or [Patreon](https://www.patreon.com/sunone)

## 📦 Release History

### Current Release
- Latest stable release available on [Mega.nz](https://mega.nz/file/0PVxDRLL#b62nQhHkjm4iOIf0i8_yLuX1Gop5AomjWONGs-yfaKk)

### Old releases
- Stored [here](https://disk.yandex.ru/d/m0jbkiLEFvnZKg).
	
## 📋 Config Documentation
- The config documentation is available in a separate [repository](https://github.com/SunOner/sunone_aimbot_docs/blob/main/config/config_cpp.md).

## 📚 References and modules

- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [OpenCV Documentation](https://docs.opencv.org/4.x/d1/dfb/intro.html)
- [Windows SDK](https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/)
- [simpleIni](https://github.com/brofield/simpleini/)
- [serial](https://github.com/wjwwood/serial)
- [ImGui](https://github.com/ocornut/imgui)
- [CppWinRT](https://github.com/microsoft/cppwinrt)
- [Python AI AIMBOT](https://github.com/SunOner/sunone_aimbot)
- [GLFW](https://www.glfw.org/)

## 📄 Licenses

### Boost
- **License:** [Boost Software License 1.0](https://www.boost.org/LICENSE_1_0.txt)

### OpenCV
- **License:** [Apache License 2.0](https://opencv.org/license.html)

### ImGui
- **License:** [MIT License](https://github.com/ocornut/imgui/blob/master/LICENSE)