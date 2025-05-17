<div align="center">

# needaimbot C++

</div>

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

5. **Build OpenCV with CUDA Support**
	- Download and install [CMake](https://cmake.org/) and [CUDA 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive).
	- Download [OpenCV](https://github.com/opencv/opencv).
	- Download [OpenCV Contrib](https://github.com/opencv/opencv_contrib/tags).
	- Create new directories: `needaimbot/needaimbot/modules/opencv/` and `needaimbot/modules/opencv/build`.
	- Extract `opencv-4.10.0` to `needaimbot/needaimbot/modules/opencv/opencv-4.10.0` and `opencv_contrib-4.10.0` to `needaimbot/modules/opencv/opencv_contrib-4.10.0`.
	- Extract cuDNN to `needaimbot/needaimbot/modules/cudnn`.
	- Open CMake and set the source code location to `needaimbot/modules/opencv/opencv-4.10.0`.
	- Set the build directory to `needaimbot/needaimbot/modules/opencv/build`.
	- Click `Configure`.
	- (Some options will appear after the next configuration application. For example, to configure the CUDNN_LIBRARY paths, you first need to activate the WITH_CUDA option and click configure.)
	- Check or configure:
		- `WITH_CUDA`
		- `WITH_CUBLAS`
		- `ENABLE_FAST_MATH`
		- `CUDA_FAST_MATH`
		- `WITH_CUDNN`
		- `CUDNN_LIBRARY` = `<full path>needaimbot/needaimbot/modules/cudnn/lib/x64/cudnn.lib`
		- `CUDNN_INCLUDE_DIR` = `<full path>needaimbot/needaimbot/modules/cudnn/include`
		- `CUDA_ARCH_BIN` = Visit the [CUDA Wiki](https://en.wikipedia.org/wiki/CUDA) to find your Nvidia GPU architecture. For example, for `RTX 3080-TI`, enter `8.6`.
		- `OPENCV_DNN_CUDA`
		- `OPENCV_EXTRA_MODULES_PATH` = `<full path>needaimbot/needaimbot/modules/opencv/opencv_contrib-4.10.0/modules`
		- `BUILD_opencv_world`
		
		- Uncheck:
			- `WITH_NVCUVENC`
			- `WITH_NVCUVID`
   - Click `Configure` again and ensure that the flags for `CUDA_FAST_MATH` and `ENABLE_FAST_MATH` are not reset.
   - Click `Generate` to build the C++ solution.
   - Close CMake and open `needaimbot/modules/opencv/build/OpenCV.sln`, or click `Open Project` in cmake.
   - Switch the build configuration to `x64` and `Release`.
   - Open the `CMakeTargets` folder in the solution.
   - Right-click on `ALL_BUILD` and select `Build`. (Building the project can take up to two hours.)
   - After building, right-click on `INSTALL` and select `Build`.
   - Verify the built files exist in the following folders:
     - `needaimbot/needaimbot/modules/opencv/build/install/include/opencv2` - Contains `.hpp` and `.h` files.
     - `needaimbot/needaimbot/modules/opencv/build/install/x64/vc16/bin` - Contains `.dll` files.
     - `needaimbot/needaimbot/modules/opencv/build/install/x64/vc16/lib` - Contains `.lib` files.

6. **Download Required Libraries**  
	- [simpleIni](https://github.com/brofield/simpleini/blob/master/SimpleIni.h)
	- [serial](https://github.com/wjwwood/serial)
	- [TensorRT-10.8.0.43](https://developer.nvidia.com/tensorrt/download/10x)
	- [GLWF Windows pre-compiled binaries](https://www.glfw.org/download.html)
	- [Eigen](https://gitlab.com/libeigen/eigen/-/releases) (Download the latest stable release)
	
7. **Extract Libraries**  
	Place the downloaded libraries into the respective directories:
	- `SimpleIni.h` -> `needaimbot/needaimbot/modules/SimpleIni.h`
	- `serial` -> `needaimbot/needaimbot/modules/serial`
	- `TensorRT-10.8.0.43` -> `needaimbot/needaimbot/modules/TensorRT-10.8.0.43`
	- `GLWF` -> `needaimbot/needaimbot/modules/glfw-3.4.bin.WIN64`
	- `Eigen` -> `needaimbot/needaimbot/modules/eigen`
	  - Extract the Eigen archive and rename the folder to `include`
	  - Verify that the header files are located at `needaimbot/needaimbot/modules/eigen/include/Eigen/`
	
8. **Compile serial**
	- Navigate to the serial directory `needaimbot/needaimbot/modules/serial/visual_studio/`
	- Open `visual_studio.sln`
	- Go to project settings -> C/C++ -> Code Generation -> and change `Runtime Library` to `Multi-threaded (/MT)`
	- Build in `Release x64`
   
9. **Configure Project Settings**
	- Open the project in Visual Studio.
	- Ensure all library paths are correctly set in **Project Properties** under **Library Directories**.
	- Go to NuGet packages and install `Microsoft.Windows.CppWinRT`.

10. **Verify CUDA Integration**
	- Right-click on the project in Visual Studio.
	- Navigate to **Build Dependencies** > **Build Customizations**.
	- Ensure that **CUDA 12.8** (.targets, .props) is included.

11. **Build the Project**
    - Switch the build configuration to **Release**.
    - Build the project by selecting **Build** > **Build Solution**.

## Export PyTorch models from Python with dynamic shapes
- .pt -> .onnx
	```bash
	pip install ultralytics -U
	yolo export model=sunxds_0.5.6.pt format=onnx dynamic=true simplify=true
	```
- For .onnx -> .engine model export use overlay export tab in overlay.

## Old releases
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