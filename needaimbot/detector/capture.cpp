#include "AppContext.h"

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

#include "simple_capture.h"
#include "duplication_api_capture.h"
#include "game_capture.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "windowsapp.lib")


cv::cuda::GpuMat latestFrameGpu;
cv::Mat latestFrameCpu;

std::array<cv::cuda::GpuMat, FRAME_BUFFER_COUNT> captureGpuBuffer;
std::array<cv::Mat, FRAME_BUFFER_COUNT> captureCpuBuffer;
std::atomic<int> captureGpuWriteIdx{0};
std::atomic<int> captureCpuWriteIdx{0};




int g_captureRegionWidth = 0;
int g_captureRegionHeight = 0;
std::atomic<int> captureFrameCount(0);
std::atomic<int> captureFps(0);
std::chrono::time_point<std::chrono::steady_clock> captureFpsStartTime;

void captureThread(int CAPTURE_WIDTH, int CAPTURE_HEIGHT)
{
    auto& ctx = AppContext::getInstance();
    try
    {
        if (ctx.config.verbose)
        {
            std::cout << "[Capture] OpenCV version: " << CV_VERSION << std::endl;
            std::cout << "[Capture] CUDA Support: " << cv::cuda::getCudaEnabledDeviceCount() << " devices found." << std::endl;
        }

        // Initialize capture method based on config
        std::unique_ptr<SimpleScreenCapture> simple_capturer;
        std::unique_ptr<DuplicationAPIScreenCapture> duplication_capturer;
        std::unique_ptr<GameCapture> game_capturer;
        
        // Create the appropriate capturer based on config
        if (ctx.config.capture_method == "duplication") {
            duplication_capturer = std::make_unique<DuplicationAPIScreenCapture>(CAPTURE_WIDTH, CAPTURE_HEIGHT);
        } else if (ctx.config.capture_method == "game_capture") {
            game_capturer = std::make_unique<GameCapture>(CAPTURE_WIDTH, CAPTURE_HEIGHT, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN), ctx.config.target_game_name);
        } else {
            simple_capturer = std::make_unique<SimpleScreenCapture>(CAPTURE_WIDTH, CAPTURE_HEIGHT);
        }
        
        timeBeginPeriod(1);
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
        SetThreadAffinityMask(GetCurrentThread(), 1 << 0);
        // Check if the selected capturer is initialized
        bool is_initialized = false;
        if (simple_capturer) {
            is_initialized = simple_capturer->IsInitialized();
        } else if (duplication_capturer) {
            is_initialized = duplication_capturer->IsInitialized();
        } else if (game_capturer) {
            is_initialized = game_capturer->initialize(); // GameCapture uses initialize() method
        }
        
        if (!is_initialized) {
             std::cerr << "[Capture] Failed to initialize " << ctx.config.capture_method << " capturer!" << std::endl;
             return;
        }

        bool buttonPreviouslyPressed = false;

        auto lastSaveTime = std::chrono::steady_clock::now();

        std::optional<std::chrono::duration<double, std::milli>> frame_duration;
        
        constexpr int PREFETCH_COUNT = 3;
        for (int i = 0; i < PREFETCH_COUNT; ++i) {
            captureGpuBuffer[i].create(CAPTURE_HEIGHT, CAPTURE_WIDTH, CV_8UC3);
        }
        
        // Pre-allocate a reusable GPU mat for uploads
        cv::cuda::GpuMat reusableGpuMat;
        
        // Create dedicated CUDA stream for capture operations
        cv::cuda::Stream captureStream;
        
        HANDLE capture_event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        auto target_interval = std::chrono::nanoseconds(1000000000 / ctx.config.capture_fps);
        auto next_capture_time = std::chrono::high_resolution_clock::now();
        bool frameLimitingEnabled = false;

        if (ctx.config.capture_fps > 0.0)
        {
            timeBeginPeriod(1);
            frame_duration = std::chrono::duration<double, std::milli>(1000.0 / ctx.config.capture_fps);
            frameLimitingEnabled = true;
        }

        captureFpsStartTime = std::chrono::steady_clock::now();
        auto start_time = std::chrono::high_resolution_clock::now();

        cv::cuda::GpuMat screenshotGpu;
        

        while (!shouldExit)
        {
            
            if (AppContext::getInstance().shouldExit) {
                std::cout << "[CaptureThread] shouldExit is true, breaking loop." << std::endl;
                break; 
            }
            if (ctx.capture_fps_changed.load())
            {
                if (ctx.config.capture_fps > 0.0)
                {
                    if (!frameLimitingEnabled)
                    {
                        timeBeginPeriod(1);
                        frameLimitingEnabled = true;
                    }
                    frame_duration = std::chrono::duration<double, std::milli>(1000.0 / ctx.config.capture_fps);
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
                ctx.capture_fps_changed.store(false);
            }

            if (ctx.capture_method_changed.load())
            {
                if (ctx.config.verbose) {
                    std::cout << "[Capture] Capture method changed to: " << ctx.config.capture_method << std::endl;
                }
                
                // Reset all capturers
                simple_capturer.reset();
                duplication_capturer.reset();
                game_capturer.reset();
                
                // Create new capturer based on method
                if (ctx.config.capture_method == "duplication") {
                    duplication_capturer = std::make_unique<DuplicationAPIScreenCapture>(CAPTURE_WIDTH, CAPTURE_HEIGHT);
                } else if (ctx.config.capture_method == "game_capture") {
                    game_capturer = std::make_unique<GameCapture>(CAPTURE_WIDTH, CAPTURE_HEIGHT, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN), ctx.config.target_game_name);
                    game_capturer->initialize();
                } else {
                    simple_capturer = std::make_unique<SimpleScreenCapture>(CAPTURE_WIDTH, CAPTURE_HEIGHT);
                }
                
                ctx.capture_method_changed.store(false);
            }

            if (ctx.detection_resolution_changed.load())
            {
                if (ctx.config.verbose) {
                    std::cout << "[Capture] Detection resolution changed. Re-initializing capturer." << std::endl;
                }
                
                // Reset all capturers
                simple_capturer.reset();
                duplication_capturer.reset();
                game_capturer.reset();

                int new_CAPTURE_WIDTH = ctx.config.detection_resolution;
                int new_CAPTURE_HEIGHT = ctx.config.detection_resolution;

                // Recreate the appropriate capturer
                if (ctx.config.capture_method == "duplication") {
                    duplication_capturer = std::make_unique<DuplicationAPIScreenCapture>(new_CAPTURE_WIDTH, new_CAPTURE_HEIGHT);
                    if (!duplication_capturer || !duplication_capturer->IsInitialized()) {
                        std::cerr << "[Capture] Failed to create or initialize DuplicationAPIScreenCapture after resolution change!" << std::endl;
                        shouldExit = true; 
                        break;             
                    }
                } else if (ctx.config.capture_method == "game_capture") {
                    game_capturer = std::make_unique<GameCapture>(new_CAPTURE_WIDTH, new_CAPTURE_HEIGHT, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN), ctx.config.target_game_name);
                    if (!game_capturer || !game_capturer->initialize()) {
                        std::cerr << "[Capture] Failed to create or initialize GameCapture after resolution change!" << std::endl;
                        shouldExit = true; 
                        break;             
                    }
                } else {
                    simple_capturer = std::make_unique<SimpleScreenCapture>(new_CAPTURE_WIDTH, new_CAPTURE_HEIGHT);
                    if (!simple_capturer || !simple_capturer->IsInitialized()) {
                        std::cerr << "[Capture] Failed to create or initialize SimpleScreenCapture after resolution change!" << std::endl;
                        shouldExit = true; 
                        break;             
                    }
                }

                g_captureRegionWidth = new_CAPTURE_WIDTH;
                g_captureRegionHeight = new_CAPTURE_HEIGHT;

                ctx.detection_resolution_changed.store(false);
            }

            
            if (ctx.capture_cursor_changed.load())
            {
                if (ctx.config.verbose) {
                    std::cout << "[Capture] Cursor capture setting changed (no capturer re-init needed)." << std::endl;
                }
                ctx.capture_cursor_changed.store(false);
            }

            if (ctx.capture_borders_changed.load())
            {
                if (ctx.config.verbose) {
                    std::cout << "[Capture] Border capture setting changed (no capturer re-init needed)." << std::endl;
                }
                ctx.capture_borders_changed.store(false);
            }


            // Measure only the actual capture API call time
            auto frame_acq_start_time = std::chrono::high_resolution_clock::now();
            cv::Mat screenshotCpu;
            if (simple_capturer) {
                screenshotCpu = simple_capturer->GetNextFrameCpu();
            } else if (duplication_capturer) {
                screenshotCpu = duplication_capturer->GetNextFrameCpu();
            } else if (game_capturer) {
                screenshotCpu = game_capturer->get_frame(); // GameCapture uses get_frame() method
            }
            auto frame_acq_end_time = std::chrono::high_resolution_clock::now();
            
            std::chrono::duration<float, std::milli> frame_acq_duration_ms = frame_acq_end_time - frame_acq_start_time;
            ctx.g_current_frame_acquisition_time_ms.store(frame_acq_duration_ms.count());
            ctx.add_to_history(ctx.g_frame_acquisition_time_history, frame_acq_duration_ms.count(), ctx.g_frame_acquisition_history_mutex);

            if (!screenshotCpu.empty()) {
                // Measure GPU upload time separately
                auto gpu_upload_start = std::chrono::high_resolution_clock::now();
                
                // Use dedicated stream to avoid legacy stream conflicts
                if (reusableGpuMat.empty()) {
                    reusableGpuMat.create(CAPTURE_HEIGHT, CAPTURE_WIDTH, CV_8UC3);
                }
                reusableGpuMat.upload(screenshotCpu, captureStream);
                screenshotGpu = reusableGpuMat;
                
                auto gpu_upload_end = std::chrono::high_resolution_clock::now();
                float gpu_upload_ms = std::chrono::duration<float, std::milli>(gpu_upload_end - gpu_upload_start).count();
                
                
                // Update latest frame for debug display (avoid clone for performance)
                if (latestFrameGpu.size() != screenshotGpu.size()) {
                    latestFrameGpu.create(screenshotGpu.size(), screenshotGpu.type());
                }
                screenshotGpu.copyTo(latestFrameGpu);
            } else {
                screenshotGpu = cv::cuda::GpuMat();
            }

            if (!screenshotGpu.empty())
                {
                    captureFrameCount++;
                    if (ctx.detector) {
                        ctx.detector->processFrame(screenshotGpu);
                    }
                }
                else
                {

                }

            auto now = std::chrono::steady_clock::now();
            auto elapsed_fps = std::chrono::duration<double>(now - captureFpsStartTime).count();
            if (elapsed_fps >= 1.0)
            {
                float current_fps_val = static_cast<float>(captureFrameCount.load()) / static_cast<float>(elapsed_fps);
                ctx.g_current_capture_fps.store(current_fps_val);
                ctx.add_to_history(ctx.g_capture_fps_history, current_fps_val, ctx.g_capture_history_mutex);

                captureFps = captureFrameCount.load();
                captureFrameCount = 0;
                captureFpsStartTime = now;
            }

            if (!ctx.config.screenshot_button.empty() && ctx.config.screenshot_button[0] != "None")
            {
                bool buttonPressed = isAnyKeyPressed(ctx.config.screenshot_button);
                auto now_ss = std::chrono::steady_clock::now();
                auto elapsed_ss = std::chrono::duration_cast<std::chrono::milliseconds>(now_ss - lastSaveTime).count();

                if (buttonPressed && !buttonPreviouslyPressed && elapsed_ss > 1000)
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
                buttonPreviouslyPressed = buttonPressed;
            }

            if (frameLimitingEnabled && frame_duration.has_value())
            {
                auto end_time = std::chrono::high_resolution_clock::now();
                auto elapsed_time = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end_time - start_time);
                auto sleep_duration = *frame_duration - elapsed_time;


                if (sleep_duration.count() > 0)
                {
                     auto sleep_start = std::chrono::high_resolution_clock::now();
                     std::this_thread::sleep_for(sleep_duration);
                     auto sleep_end = std::chrono::high_resolution_clock::now();
                     
                     // Record actual additional sleep time
                     float actual_sleep_ms = std::chrono::duration<float, std::milli>(sleep_end - sleep_start).count();
                     ctx.g_current_fps_delay_time_ms.store(actual_sleep_ms, std::memory_order_relaxed);
                     ctx.add_to_history(ctx.g_fps_delay_time_history, actual_sleep_ms, ctx.g_fps_delay_history_mutex);
                } else {
                     ctx.g_current_fps_delay_time_ms.store(0.0f, std::memory_order_relaxed);
                }
                 start_time = std::chrono::high_resolution_clock::now();
            } else {
                // No FPS limiting, no delay
                ctx.g_current_fps_delay_time_ms.store(0.0f, std::memory_order_relaxed);
                start_time = std::chrono::high_resolution_clock::now();
            }
        }

        if (frameLimitingEnabled)
        {
            timeEndPeriod(1);
        }
        
        // Clean up GPU resources before thread exit
        reusableGpuMat.release();
        screenshotGpu.release();
        
        std::cout << "[Capture] Capture thread exiting." << std::endl;

    }
    catch (const std::exception& e)
    {
        std::cerr << "[Capture] Unhandled exception in capture thread: " << e.what() << std::endl;
    }
}
