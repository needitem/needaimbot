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
#include <array>

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

#ifndef __INTELLISENSE__
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#endif

#include "capture.h"
#include "detector.h"
#include "needaimbot.h"
#include "keycodes.h"
#include "keyboard_listener.h"
#include "other_tools.h"

#include "duplication_api_capture.h"

extern Detector detector;
extern std::mutex configMutex;

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "windowsapp.lib")

// Legacy single-buffer variables (required by visuals and draw_debug modules)
cv::cuda::GpuMat latestFrameGpu;
cv::Mat latestFrameCpu;
// Ring buffer to avoid locks
std::array<cv::cuda::GpuMat, FRAME_BUFFER_COUNT> captureGpuBuffer;
std::array<cv::Mat, FRAME_BUFFER_COUNT> captureCpuBuffer;
std::atomic<int> captureGpuWriteIdx{0};
std::atomic<int> captureCpuWriteIdx{0};
std::atomic<bool> newFrameAvailable = false;
// Mutex for condition_variable synchronization (defined for ring buffer signaling)
std::mutex frameMutex;

int g_captureRegionWidth = 0;
int g_captureRegionHeight = 0;
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

            if (detection_resolution_changed.load())
            {
                if (config.verbose) {
                    std::cout << "[Capture] Detection resolution changed. Re-initializing capturer." << std::endl;
                }
                capturer.reset(); // Release old capturer first

                int new_CAPTURE_WIDTH = config.detection_resolution;
                int new_CAPTURE_HEIGHT = config.detection_resolution;

                auto tempCapturer = std::make_unique<DuplicationAPIScreenCapture>(new_CAPTURE_WIDTH, new_CAPTURE_HEIGHT);

                if (!tempCapturer || !tempCapturer->IsInitialized()) {
                    std::cerr << "[Capture] Failed to create or initialize new DuplicationAPIScreenCapture after resolution change!" << std::endl;
                    shouldExit = true; // Signal loop to terminate
                    break;             // Exit the while loop
                } else {
                    capturer = std::move(tempCapturer); // Move ownership to the main capturer
                }

                // This part will only be reached if initialization was successful
                if (capturer) { // Redundant check given the break, but good practice
                    detector.setCaptureEvent(capturer->GetCaptureDoneEvent());
                }

                g_captureRegionWidth = new_CAPTURE_WIDTH;
                g_captureRegionHeight = new_CAPTURE_HEIGHT;

                detection_resolution_changed.store(false);
            }

            // Handle other flags that don't require full re-initialization
            if (capture_cursor_changed.load())
            {
                if (config.verbose) {
                    std::cout << "[Capture] Cursor capture setting changed (no capturer re-init needed)." << std::endl;
                }
                capture_cursor_changed.store(false);
            }

            if (capture_borders_changed.load())
            {
                if (config.verbose) {
                    std::cout << "[Capture] Border capture setting changed (no capturer re-init needed)." << std::endl;
                }
                capture_borders_changed.store(false);
            }

            if (capture_timeout_changed.load())
            {
                if (capturer) {
                    capturer->SetAcquireTimeout(config.capture_timeout_ms);
                }
                if (config.verbose) {
                    std::cout << "[Capture] AcquireFrame timeout changed to: " << config.capture_timeout_ms << "ms" << std::endl;
                }
                capture_timeout_changed.store(false);
            }

            cv::cuda::GpuMat screenshotGpu;
            cv::Mat screenshotCpu;

            auto frame_acq_start_time = std::chrono::high_resolution_clock::now();

            if (config.capture_use_cuda) {
                screenshotGpu = capturer->GetNextFrameGpu();
            } else {
                screenshotCpu = capturer->GetNextFrameCpu();
            }

            auto frame_acq_end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> frame_acq_duration_ms = frame_acq_end_time - frame_acq_start_time;
            g_current_frame_acquisition_time_ms.store(frame_acq_duration_ms.count());
            add_to_history(g_frame_acquisition_time_history, frame_acq_duration_ms.count(), g_frame_acquisition_history_mutex);

            if (config.capture_use_cuda) {
                if (!screenshotGpu.empty())
                {
                    detector.processFrame(screenshotGpu);
                    // Write to ring buffer without lock
                    int idx = (captureGpuWriteIdx.load(std::memory_order_relaxed) + 1) % FRAME_BUFFER_COUNT;
                    captureGpuBuffer[idx] = screenshotGpu;
                    captureGpuWriteIdx.store(idx, std::memory_order_release);
                    newFrameAvailable.store(true, std::memory_order_release);
                    frameCV.notify_one(); // Wake display if needed
                }
                else
                {

                }
            } else {
                if (!screenshotCpu.empty())
                {
                    detector.processFrame(screenshotCpu);
                    // Write to CPU ring buffer without lock
                    int idx = (captureCpuWriteIdx.load(std::memory_order_relaxed) + 1) % FRAME_BUFFER_COUNT;
                    captureCpuBuffer[idx] = screenshotCpu;
                    captureCpuWriteIdx.store(idx, std::memory_order_release);
                    newFrameAvailable.store(true, std::memory_order_release);
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
                float current_fps_val = static_cast<float>(captureFrameCount.load()) / static_cast<float>(elapsed_fps > 0 ? elapsed_fps : 1); 
                g_current_capture_fps.store(current_fps_val);
                add_to_history(g_capture_fps_history, current_fps_val, g_capture_history_mutex);

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
                    // Save the most recent frame from ring buffer
                    if (config.capture_use_cuda)
                    {
                        int idx = captureGpuWriteIdx.load(std::memory_order_acquire);
                        const cv::cuda::GpuMat& gpuFrame = captureGpuBuffer[idx];
                        if (!gpuFrame.empty())
                        {
                            cv::Mat tmp;
                            gpuFrame.download(tmp);
                            saveScreenshot(tmp);
                            lastSaveTime = now_ss;
                        }
                    }
                    else
                    {
                        int idx = captureCpuWriteIdx.load(std::memory_order_acquire);
                        const cv::Mat& cpuFrame = captureCpuBuffer[idx];
                        if (!cpuFrame.empty())
                        {
                            saveScreenshot(cpuFrame);
                            lastSaveTime = now_ss;
                        }
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