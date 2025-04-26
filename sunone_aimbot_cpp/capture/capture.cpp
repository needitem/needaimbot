#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <iostream>
#include <atomic>
#include <thread>
#include <mutex>
#include <chrono>
#include <timeapi.h>
#include <condition_variable>
#include <memory>

#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda.hpp>

#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.System.h>
#include <winrt/Windows.System.Threading.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>
#include <winrt/base.h>
#include <comdef.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>

#include "capture.h"
#include "detector.h"
#include "sunone_aimbot_cpp.h"
#include "keycodes.h"
#include "keyboard_listener.h"
#include "other_tools.h"

#include "duplication_api_capture.h"

extern Detector detector;
extern std::mutex configMutex;

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "windowsapp.lib")

cv::cuda::GpuMat latestFrameGpu;
std::mutex frameMutex;
cv::Mat latestFrameCpu;
std::atomic<bool> newFrameAvailable = false;

int screenWidth = 0;
int screenHeight = 0;
std::atomic<int> captureFrameCount(0);
std::atomic<int> captureFps(0);
std::chrono::time_point<std::chrono::high_resolution_clock> captureFpsStartTime;

void captureThread(int CAPTURE_WIDTH, int CAPTURE_HEIGHT)
{
    try
    {
        if (config.verbose)
        {
            std::cout << "[Capture] OpenCV version: " << CV_VERSION << std::endl;
            std::cout << "[Capture] CUDA Support: " << cv::cuda::getCudaEnabledDeviceCount() << " devices found." << std::endl;
        }

        std::unique_ptr<DuplicationAPIScreenCapture> capturer = std::make_unique<DuplicationAPIScreenCapture>(CAPTURE_WIDTH, CAPTURE_HEIGHT);
        if (!capturer) {
             std::cerr << "[Capture] Failed to initialize DuplicationAPIScreenCapture!" << std::endl;
             return;
        }

        if (capturer) {
            detector.setCaptureEvent(capturer->GetCaptureDoneEvent());
        }

        bool buttonPreviouslyPressed = false;

        auto lastSaveTime = std::chrono::steady_clock::now();

        std::optional<std::chrono::duration<double, std::milli>> frame_duration;
        bool frameLimitingEnabled = false;

        if (config.capture_fps > 0.0)
        {
            timeBeginPeriod(1);
            frame_duration = std::chrono::duration<double, std::milli>(1000.0 / config.capture_fps);
            frameLimitingEnabled = true;
        }

        captureFpsStartTime = std::chrono::high_resolution_clock::now();
        auto start_time = std::chrono::high_resolution_clock::now();

        while (!shouldExit)
        {
            if (capture_fps_changed.load())
            {
                if (config.capture_fps > 0.0)
                {
                    if (!frameLimitingEnabled)
                    {
                        timeBeginPeriod(1);
                        frameLimitingEnabled = true;
                    }
                    frame_duration = std::chrono::duration<double, std::milli>(1000.0 / config.capture_fps);
                }
                else
                {
                    if (frameLimitingEnabled)
                    {
                        timeEndPeriod(1);
                        frameLimitingEnabled = false;
                    }
                    frame_duration = std::nullopt;
                }
                capture_fps_changed.store(false);
            }

            if (detection_resolution_changed.load() ||
                capture_cursor_changed.load() ||
                capture_borders_changed.load())
            {
                capturer.reset();

                int new_CAPTURE_WIDTH = config.detection_resolution;
                int new_CAPTURE_HEIGHT = config.detection_resolution;

                capturer = std::make_unique<DuplicationAPIScreenCapture>(new_CAPTURE_WIDTH, new_CAPTURE_HEIGHT);

                if (!capturer || !capturer->IsInitialized()) {
                    std::cerr << "[Capture] Failed to create or initialize new DuplicationAPIScreenCapture after config change!" << std::endl;
                    capturer.reset();
                    break;
                }

                detector.setCaptureEvent(capturer->GetCaptureDoneEvent());

                screenWidth = new_CAPTURE_WIDTH;
                screenHeight = new_CAPTURE_HEIGHT;

                detection_resolution_changed.store(false);
                capture_cursor_changed.store(false);
                capture_borders_changed.store(false);
            }

            cv::cuda::GpuMat screenshotGpu;
            cv::Mat screenshotCpu;

            if (config.capture_use_cuda) {
                screenshotGpu = capturer->GetNextFrameGpu();

                if (!screenshotGpu.empty())
                {
                    detector.processFrame(screenshotGpu);

                    {
                        std::lock_guard<std::mutex> lock(frameMutex);
                        latestFrameGpu = screenshotGpu.clone();
                        {
                           std::lock_guard<std::mutex> config_lock(configMutex);
                            // The download for the display window is handled in the display thread.
                            // No download needed here if the window is shown.
                            // if (config.show_window)
                            // {
                            //    screenshotGpu.download(latestFrameCpu); // Removed this line
                            // }
                        }
                        newFrameAvailable = true;
                        frameCV.notify_one();
                    }
                }
                else
                {

                }
            } else {
                screenshotCpu = capturer->GetNextFrameCpu();

                if (!screenshotCpu.empty())
                {
                    detector.processFrame(screenshotCpu);

                    {
                        std::lock_guard<std::mutex> lock(frameMutex);

                        latestFrameCpu = screenshotCpu.clone();
                        newFrameAvailable = true;

                    }
                    frameCV.notify_one();
                }
                else
                {

                }
            }

            captureFrameCount++;
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed_fps = std::chrono::duration_cast<std::chrono::seconds>(now - captureFpsStartTime).count();
            if (elapsed_fps >= 1)
            {
                captureFps = captureFrameCount.load();
                captureFrameCount = 0;
                captureFpsStartTime = now;
            }

            if (!config.screenshot_button.empty() && config.screenshot_button[0] != "None")
            {
                bool buttonPressed = isAnyKeyPressed(config.screenshot_button);
                auto now_ss = std::chrono::steady_clock::now();
                auto elapsed_ss = std::chrono::duration_cast<std::chrono::milliseconds>(now_ss - lastSaveTime).count();

                if (buttonPressed && !buttonPreviouslyPressed && elapsed_ss > 1000)
                {
                    std::lock_guard<std::mutex> lock(frameMutex);

                    if (config.capture_use_cuda && !latestFrameGpu.empty()) {
                        latestFrameGpu.download(latestFrameCpu);
                    }

                    if (!latestFrameCpu.empty())
                    {
                        saveScreenshot(latestFrameCpu);
                        lastSaveTime = now_ss;
                    }
                }
                buttonPreviouslyPressed = buttonPressed;
            }

            if (frameLimitingEnabled && frame_duration.has_value())
            {
                auto end_time = std::chrono::high_resolution_clock::now();
                auto elapsed_time = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end_time - start_time);
                auto sleep_duration = *frame_duration - elapsed_time;

                if (sleep_duration.count() > 0)
                {
                     std::this_thread::sleep_for(sleep_duration);
                }
                 start_time = std::chrono::high_resolution_clock::now();
            } else {
                start_time = std::chrono::high_resolution_clock::now();
            }
        }

        if (frameLimitingEnabled)
        {
            timeEndPeriod(1);
        }

    }
    catch (const std::exception& e)
    {
        std::cerr << "[Capture] Unhandled exception in capture thread: " << e.what() << std::endl;
    }
}