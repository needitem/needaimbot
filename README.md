<div align="center">

# needaimbot C++

[![CUDA](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![TensorRT](https://img.shields.io/badge/TensorRT-10.8-blue.svg)](https://developer.nvidia.com/tensorrt)
[![C++](https://img.shields.io/badge/C++-17-orange.svg)](https://en.cppreference.com/w/cpp/17)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

</div>

## 🎯 Overview

**needaimbot** is a high-performance C++ AI-powered aiming assistance tool that utilizes deep learning models and advanced computer vision techniques for real-time target detection and tracking. Built with cutting-edge technologies including TensorRT for GPU acceleration, CUDA for parallel computing, and enhanced Kalman filtering with frame-based prediction.

### Key Highlights
- **Real-time AI target detection** with TensorRT optimization
- **Advanced predictive tracking** using GPU-accelerated Kalman filters with frame-based prediction
- **Multiple capture methods** with hardware acceleration and optimized memory management
- **Extensive input device support** (Logitech G-Hub, Razer, KMBox, Serial, Arduino)
- **Customizable overlay interface** with real-time configuration and visual debugging
- **GPU-accelerated processing** with CUDA kernels for minimal latency
- **Precise frame prediction control** with unlimited range support
- **Optimized color conversion** with pinned memory and frame buffer pooling

> **⚠️ WARNING:** TensorRT version 10 does not support the Pascal architecture (10 series graphics cards). Use only with GPUs of at least the 20 series (Turing architecture or newer).

## 📥 Quick Start

### Prerequisites
1. **Download and Install CUDA 12.8**
   - [CUDA 12.8 Download](https://developer.nvidia.com/cuda-12-8-0-download-archive)
   - Ensure CUDA is added to your system PATH

2. **Download the Latest Release**  
   - [Download from Mega.nz](https://mega.nz/file/dekngIhD#lYudr_T6ob1dmPKiU3SWgFQUGE654E1vefyRlfSULy8)

### Installation Steps

1. **Extract the Archive**
   - Extract all contents to your desired location
   - Ensure the folder structure is preserved

2. **First Launch and Model Export**
   - Run `ai.exe` as administrator
   - Wait for the standard `.onnx` model to be exported to TensorRT format (typically 2-5 minutes)
   - The exported `.engine` file will be cached for faster subsequent launches

3. **Configuration**
   - Press `HOME` to open the overlay interface
   - Configure your preferences in real-time
   - Settings are automatically saved to `config.ini`

### Default Controls
- **Right Mouse Button:** Aim at detected target
- **F2:** Exit program
- **F3:** Toggle aiming pause
- **F4:** Reload configuration
- **HOME:** Show/hide overlay

## ✨ Features

### Core Functionality
- **🎯 AI-Powered Target Detection**: Advanced neural networks with TensorRT optimization for precise identification
- **🔄 Real-Time Tracking**: GPU-accelerated Kalman filtering with frame-based prediction and motion compensation
- **⚡ GPU Acceleration**: CUDA kernels and TensorRT optimization for sub-5ms inference
- **🎮 Multiple Input Methods**: Native support for gaming peripherals and custom hardware
- **📊 Optical Flow Integration**: Advanced motion detection with Lucas-Kanade algorithm
- **🎛️ Live Configuration**: Real-time parameter adjustment without restart
- **🎨 Visual Debugging**: Enhanced tracking visualization with prediction indicators and performance metrics

### Technical Features

#### AI & Deep Learning
- TensorRT 10.8 with dynamic shape support
- Automatic ONNX to TensorRT conversion
- Multi-model hot-swapping capability
- Optimized post-processing pipeline
- GPU memory pooling and efficient allocation
- Support for YOLOv8/v9 architectures

#### Advanced Tracking System
- **GPU Kalman Filter**: CUDA-accelerated state estimation
- **Frame-based Prediction**: Precise control over prediction frames
- **SORT Tracker**: Multi-target tracking with Hungarian algorithm
- **Simple Kalman Tracker**: Lightweight single-target tracking
- **Motion Compensation**: Optical flow integration for movement prediction
- **Target Lock**: Intelligent target switching prevention

#### Input Device Support
- **Logitech G-Hub**: Direct driver communication
- **Razer Devices**: Native Razer peripheral support
- **KMBox Net**: Network-based hardware control
- **Serial/Arduino**: Custom hardware integration
- **Win32 API**: Fallback software input
- **Makcu Connection**: Professional hardware support

#### Capture & Processing
- **Desktop Duplication API**: Hardware-accelerated screen capture
- **Pinned Memory**: Zero-copy transfers between CPU and GPU
- **Frame Buffer Pool**: Efficient memory reuse
- **Color Space Conversion**: Optimized BGRA to BGR conversion
- **Multi-Monitor Support**: Automatic display detection
- **Region of Interest**: Configurable capture areas

## 🏗️ Architecture

### System Requirements

#### Minimum Requirements
- **OS**: Windows 10/11 (x64)
- **GPU**: NVIDIA RTX 2060 or better
- **RAM**: 8GB
- **CUDA**: 12.8
- **Storage**: 2GB free space

#### Recommended Requirements
- **GPU**: NVIDIA RTX 3070 or better
- **RAM**: 16GB
- **CPU**: Intel i5-10400 / AMD Ryzen 5 3600 or better
- **Storage**: SSD with 5GB free space

### Performance Metrics
| Component | Performance |
|-----------|------------|
| Inference Time | <5ms (RTX 3080) |
| End-to-End Latency | <10ms total |
| VRAM Usage | ~2GB |
| RAM Usage | ~500MB |
| CPU Usage | <10% |
| Max Frame Rate | 240 FPS |

## 🛠️ Building from Source

### Required Tools
1. **Visual Studio 2022 Community**
   - [Download](https://visualstudio.microsoft.com/vs/community/)
   - Install with C++ desktop development workload

2. **Windows SDK**
   - Version 10.0.26100.0 or newer

3. **CUDA Toolkit 12.8**
   - [Download](https://developer.nvidia.com/cuda-12-8-0-download-archive)

4. **cuDNN 9.7.1**
   - [Download](https://developer.nvidia.com/cudnn-downloads)

### Build Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/needaimbot.git
   cd needaimbot
   ```

2. **Set Up Dependencies**
   Create `needaimbot/needaimbot/modules/` directory and add:
   - [SimpleIni.h](https://github.com/brofield/simpleini/blob/master/SimpleIni.h)
   - [TensorRT-10.8.0.43](https://developer.nvidia.com/tensorrt/download/10x)
   - [GLFW 3.4](https://www.glfw.org/download.html) (Windows pre-compiled binaries)
   - [Eigen](https://gitlab.com/libeigen/eigen/-/releases) (latest stable)

3. **Configure Visual Studio**
   - Open `needaimbot.sln`
   - Install NuGet package: `Microsoft.Windows.CppWinRT`
   - Verify CUDA 12.8 build customizations are enabled
   - Set configuration to **Release**

4. **Build**
   - Build → Build Solution (Ctrl+Shift+B)
   - Output will be in `x64/Release/`

## 🤝 Contributing & Collaboration

### How to Contribute

We welcome contributions from the community! Here's how you can help:

#### 1. Fork and Clone
```bash
git clone https://github.com/yourusername/needaimbot.git
cd needaimbot
git remote add upstream https://github.com/originalrepo/needaimbot.git
```

#### 2. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

#### 3. Development Guidelines

**Code Standards:**
- Follow existing C++17 conventions
- Use RAII and smart pointers where appropriate
- Maintain consistent naming (camelCase for functions, PascalCase for classes)
- Add comments for complex algorithms

**Performance Considerations:**
- Profile before optimizing
- Prefer CUDA kernels for parallel operations
- Use pinned memory for CPU-GPU transfers
- Minimize memory allocations in hot paths

**Testing Requirements:**
- Test on multiple GPU architectures (20, 30, 40 series)
- Verify memory leak-free operation
- Check performance regression
- Test all input device drivers

#### 4. Commit Guidelines
```bash
# Use conventional commits
git commit -m "feat: add new tracking algorithm"
git commit -m "fix: resolve memory leak in detector"
git commit -m "perf: optimize color conversion kernel"
git commit -m "docs: update build instructions"
```

#### 5. Submit Pull Request
- Push to your fork
- Create PR with detailed description
- Include performance metrics if applicable
- Reference any related issues

### Priority Areas for Contribution

- **🚀 Performance Optimization**
  - CUDA kernel improvements
  - Memory management enhancements
  - Algorithm optimization

- **🎯 AI Models**
  - Model architecture improvements
  - Training pipeline enhancements
  - Dataset expansion

- **🎮 Device Support**
  - New input device drivers
  - Hardware compatibility improvements
  - Cross-platform support research

- **📚 Documentation**
  - Tutorial videos
  - Configuration guides
  - API documentation

### Development Environment Setup

1. **IDE Configuration**
   - Use Visual Studio 2022 with IntelliSense
   - Install CUDA syntax highlighting
   - Configure code formatting (clang-format)

2. **Debugging Tools**
   - NVIDIA Nsight for CUDA debugging
   - Visual Studio Performance Profiler
   - GPU-Z for monitoring

3. **Version Control**
   - Use Git LFS for large binary files
   - Keep commits atomic and focused
   - Write meaningful commit messages

## 🐛 Troubleshooting

### Common Issues

#### Build Errors
| Error | Solution |
|-------|----------|
| CUDA not found | Reinstall CUDA 12.8, verify PATH |
| LNK4098 conflicts | Ensure all libs use `/MT` flag |
| NVCC fatal error | Reinstall CUDA after VS2022 |
| Missing headers | Verify modules directory structure |

#### Runtime Issues
| Issue | Solution |
|-------|----------|
| Model load failure | Check .onnx file location and VRAM |
| Low FPS | Enable GPU acceleration, check temps |
| Capture black screen | Run as administrator |
| Input lag | Reduce capture resolution |

#### Device Issues
| Device | Common Fix |
|--------|------------|
| Logitech | Install G-Hub, run as admin |
| Razer | Update Synapse, check USB |
| Serial | Verify COM port settings |
| KMBox | Check network connectivity |

## 🧠 AI Model Management

### Converting Models
```bash
# PyTorch to ONNX with dynamic shapes
pip install ultralytics
yolo export model=your_model.pt format=onnx dynamic=true simplify=true
```

### Model Optimization
- Use FP16 precision for RTX 30/40 series
- Enable INT8 quantization for inference speed
- Optimize batch size based on VRAM
- Profile with TensorRT's trtexec tool

## 📚 Resources

### Documentation
- [Configuration Guide](https://github.com/SunOner/sunone_aimbot_docs/blob/main/config/config_cpp.md)
- [API Reference](docs/api.md)
- [Model Training Guide](docs/training.md)

### Community
- [Discord Server](https://discord.gg/sunone) - Get help and share ideas
- [GitHub Issues](https://github.com/yourusername/needaimbot/issues) - Report bugs
- [Discussions](https://github.com/yourusername/needaimbot/discussions) - Feature requests

### Support Development
- [Boosty](https://boosty.to/sunone) - Support and get enhanced models
- [Patreon](https://www.patreon.com/sunone) - Monthly support

## 📋 Changelog

### Latest Updates
- ✨ GPU-accelerated Kalman filter with frame-based prediction
- 🎯 Precise frame prediction control with unlimited range
- 🔧 Fixed ImGui ID conflicts in tracking tab
- 🚀 Optimized color conversion with pinned memory
- 📊 Enhanced visual debugging with prediction indicators

### Previous Releases
- Available at [Yandex Disk Archive](https://disk.yandex.ru/d/m0jbkiLEFvnZKg)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

### Libraries and Tools
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)
- [Dear ImGui](https://github.com/ocornut/imgui)
- [Eigen](https://eigen.tuxfamily.org/)
- [GLFW](https://www.glfw.org/)
- [STB Libraries](https://github.com/nothings/stb)
- [SimpleIni](https://github.com/brofield/simpleini)

### Community Contributors
Special thanks to all contributors who have helped improve this project through code contributions, bug reports, and feature suggestions.

---

<div align="center">
Made with ❤️ by the needaimbot community
</div>