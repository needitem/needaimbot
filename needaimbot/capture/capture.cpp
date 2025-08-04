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
#include <queue>

#include "../cuda/cuda_image_processing.h"
#include "../cuda/cuda_error_check.h"
#include "../cuda/color_conversion.h"
#include "../utils/image_io.h"
#include "frame_buffer_pool.h"

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
#include "../include/other_tools.h"

#include "simple_capture.h"
#include "windows_graphics_capture.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "windowsapp.lib")


SimpleCudaMat latestFrameGpu;

std::array<SimpleCudaMat, FRAME_BUFFER_COUNT> captureGpuBuffer;
std::atomic<int> captureGpuWriteIdx{0};




int g_captureRegionWidth = 0;
int g_captureRegionHeight = 0;
std::atomic<int> captureFrameCount(0);
std::atomic<int> captureFps(0);
std::chrono::time_point<std::chrono::steady_clock> captureFpsStartTime;

// 비동기 파이프라인을 위한 구조체
struct PipelineFrame {
    SimpleCudaMat gpuFrame;
    cudaEvent_t event;
    bool ready;
};

void captureThread(int CAPTURE_WIDTH, int CAPTURE_HEIGHT)
{
    std::cout << "[CaptureThread] Starting capture thread with resolution: " << CAPTURE_WIDTH << "x" << CAPTURE_HEIGHT << std::endl;
    auto& ctx = AppContext::getInstance();
    
    // RAII guard for CUDA cleanup
    CudaResourceGuard cudaGuard;
    
    // 프레임 버퍼 풀 초기화
    if (!g_frameBufferPool) {
        g_frameBufferPool = std::make_unique<FrameBufferPool>(10);
    }
    
    try
    {

        // Initialize capture method based on config
        std::unique_ptr<SimpleScreenCapture> simple_capturer;
        std::unique_ptr<WindowsGraphicsCapture> wingraphics_capturer;
        
        std::cout << "[CaptureThread] Initializing capture method: " << ctx.config.capture_method << std::endl;
        
        // Create the appropriate capturer based on config
 if (ctx.config.capture_method == "wingraphics") {
            wingraphics_capturer = std::make_unique<WindowsGraphicsCapture>(CAPTURE_WIDTH, CAPTURE_HEIGHT);
            std::cout << "[CaptureThread] Created WindowsGraphicsCapture" << std::endl;
            // Apply initial offset (check if aim+shoot is active)
            float initialOffsetX = ctx.config.crosshair_offset_x;
            float initialOffsetY = ctx.config.crosshair_offset_y;
            if (ctx.config.enable_aim_shoot_offset && ctx.aiming.load() && ctx.shooting.load()) {
                initialOffsetX = ctx.config.aim_shoot_offset_x;
                initialOffsetY = ctx.config.aim_shoot_offset_y;
            }
            wingraphics_capturer->UpdateCaptureRegion(initialOffsetX, initialOffsetY);
        } else {
            simple_capturer = std::make_unique<SimpleScreenCapture>(CAPTURE_WIDTH, CAPTURE_HEIGHT);
            std::cout << "[CaptureThread] Created SimpleScreenCapture" << std::endl;
            // Apply initial offset (check if aim+shoot is active)
            float initialOffsetX = ctx.config.crosshair_offset_x;
            float initialOffsetY = ctx.config.crosshair_offset_y;
            if (ctx.config.enable_aim_shoot_offset && ctx.aiming.load() && ctx.shooting.load()) {
                initialOffsetX = ctx.config.aim_shoot_offset_x;
                initialOffsetY = ctx.config.aim_shoot_offset_y;
            }
            simple_capturer->UpdateCaptureRegion(initialOffsetX, initialOffsetY);
        }
        
        timeBeginPeriod(1);
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
        
        // 개선된 스레드 친화도 설정 - NUMA 인식
        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);
        DWORD_PTR affinityMask = 0;
        
        // 물리 코어 선호 (하이퍼스레딩 회피)
        for (DWORD i = 0; i < sysInfo.dwNumberOfProcessors; i += 2) {
            affinityMask |= (1ULL << i);
        }
        
        if (affinityMask == 0) affinityMask = 1; // 최소한 하나의 코어
        SetThreadAffinityMask(GetCurrentThread(), affinityMask);
        // Check if the selected capturer is initialized
        bool is_initialized = false;
        if (simple_capturer) {
            is_initialized = simple_capturer->IsInitialized();
        } else if (wingraphics_capturer) {
            is_initialized = wingraphics_capturer->IsInitialized();
        }
        
        if (!is_initialized) {
             std::cerr << "[Capture] Failed to initialize " << ctx.config.capture_method << " capturer!" << std::endl;
             return;
        }

        bool buttonPreviouslyPressed = false;

        auto lastSaveTime = std::chrono::steady_clock::now();

        std::optional<std::chrono::duration<double, std::milli>> frame_duration;
        
        // 3단계 비동기 파이프라인을 위한 스트림 생성
        cudaStream_t captureStream = nullptr;
        cudaStream_t processStream = nullptr;
        cudaStream_t uploadStream = nullptr;
        
        CUDA_CHECK_WARN(cudaStreamCreateWithFlags(&captureStream, cudaStreamNonBlocking));
        CUDA_CHECK_WARN(cudaStreamCreateWithFlags(&processStream, cudaStreamNonBlocking));
        CUDA_CHECK_WARN(cudaStreamCreateWithFlags(&uploadStream, cudaStreamNonBlocking));
        
        // 파이프라인 버퍼 준비
        constexpr int PIPELINE_DEPTH = 3;
        std::array<PipelineFrame, PIPELINE_DEPTH> pipeline;
        for (int i = 0; i < PIPELINE_DEPTH; ++i) {
            pipeline[i].ready = true;
            CUDA_CHECK_WARN(cudaEventCreateWithFlags(&pipeline[i].event, cudaEventDisableTiming));
        }
        int pipelineIdx = 0;
        
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

        SimpleCudaMat screenshotGpu;
        

        while (!should_exit)
        {
            
            if (AppContext::getInstance().should_exit) {
                std::cout << "[CaptureThread] should_exit is true, breaking loop." << std::endl;
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
                
                // Reset all capturers
                simple_capturer.reset();
                wingraphics_capturer.reset();
                
                // Determine which offset to use (check aim+shoot state)
                bool use_aim_shoot_offset = ctx.config.enable_aim_shoot_offset && ctx.aiming.load() && ctx.shooting.load();
                float offsetX = use_aim_shoot_offset ? ctx.config.aim_shoot_offset_x : ctx.config.crosshair_offset_x;
                float offsetY = use_aim_shoot_offset ? ctx.config.aim_shoot_offset_y : ctx.config.crosshair_offset_y;
                
                // Create new capturer based on method
                if (ctx.config.capture_method == "wingraphics") {
                    wingraphics_capturer = std::make_unique<WindowsGraphicsCapture>(CAPTURE_WIDTH, CAPTURE_HEIGHT);
                    if (wingraphics_capturer) {
                        wingraphics_capturer->UpdateCaptureRegion(offsetX, offsetY);
                    }
                } else {
                    simple_capturer = std::make_unique<SimpleScreenCapture>(CAPTURE_WIDTH, CAPTURE_HEIGHT);
                    if (simple_capturer) {
                        simple_capturer->UpdateCaptureRegion(offsetX, offsetY);
                    }
                }
                
                ctx.capture_method_changed.store(false);
            }

            if (ctx.detection_resolution_changed.load())
            {
                
                // Reset all capturers
                simple_capturer.reset();
                wingraphics_capturer.reset();

                int new_CAPTURE_WIDTH = ctx.config.detection_resolution;
                int new_CAPTURE_HEIGHT = ctx.config.detection_resolution;

                // Recreate the appropriate capturer
                // Determine which offset to use (check aim+shoot state)
                bool use_aim_shoot_offset = ctx.config.enable_aim_shoot_offset && ctx.aiming.load() && ctx.shooting.load();
                float offsetX = use_aim_shoot_offset ? ctx.config.aim_shoot_offset_x : ctx.config.crosshair_offset_x;
                float offsetY = use_aim_shoot_offset ? ctx.config.aim_shoot_offset_y : ctx.config.crosshair_offset_y;
                
                if (ctx.config.capture_method == "wingraphics") {
                    wingraphics_capturer = std::make_unique<WindowsGraphicsCapture>(new_CAPTURE_WIDTH, new_CAPTURE_HEIGHT);
                    if (!wingraphics_capturer || !wingraphics_capturer->IsInitialized()) {
                        std::cerr << "[Capture] Failed to create or initialize WindowsGraphicsCapture after resolution change!" << std::endl;
                        should_exit = true; 
                        break;             
                    }
                    // Apply correct offset based on current state
                    wingraphics_capturer->UpdateCaptureRegion(offsetX, offsetY);
                } else {
                    simple_capturer = std::make_unique<SimpleScreenCapture>(new_CAPTURE_WIDTH, new_CAPTURE_HEIGHT);
                    if (!simple_capturer || !simple_capturer->IsInitialized()) {
                        std::cerr << "[Capture] Failed to create or initialize SimpleScreenCapture after resolution change!" << std::endl;
                        should_exit = true; 
                        break;             
                    }
                    // Apply correct offset based on current state
                    simple_capturer->UpdateCaptureRegion(offsetX, offsetY);
                }

                g_captureRegionWidth = new_CAPTURE_WIDTH;
                g_captureRegionHeight = new_CAPTURE_HEIGHT;
                
                // Force offset recheck on next frame
                ctx.crosshair_offset_changed.store(true);

                ctx.detection_resolution_changed.store(false);
            }

            
            if (ctx.capture_cursor_changed.load())
            {
                ctx.capture_cursor_changed.store(false);
            }

            if (ctx.capture_borders_changed.load())
            {
                ctx.capture_borders_changed.store(false);
            }

            // Handle crosshair offset changes or aim+shoot state changes
            static bool was_aim_shoot = false;
            bool is_aim_shoot = ctx.config.enable_aim_shoot_offset && ctx.aiming.load() && ctx.shooting.load();
            
            // Check if aim+shoot state changed, which requires offset update
            bool state_changed = (is_aim_shoot != was_aim_shoot);
            if (state_changed) {
                ctx.crosshair_offset_changed.store(true);
            }
            
            // Always check and apply offset changes
            static float last_offsetX = -999999.0f;
            static float last_offsetY = -999999.0f;
            
            float current_offsetX, current_offsetY;
            
            // Determine which offset to use
            if (is_aim_shoot) {
                current_offsetX = ctx.config.aim_shoot_offset_x;
                current_offsetY = ctx.config.aim_shoot_offset_y;
            } else {
                current_offsetX = ctx.config.crosshair_offset_x;
                current_offsetY = ctx.config.crosshair_offset_y;
            }
            
            // Check if offset actually changed
            bool offset_actually_changed = (current_offsetX != last_offsetX || current_offsetY != last_offsetY);
            
            if (ctx.crosshair_offset_changed.load() || offset_actually_changed)
            {
                // Apply offset to the active capturer
                if (wingraphics_capturer) {
                    wingraphics_capturer->UpdateCaptureRegion(current_offsetX, current_offsetY);
                } else if (simple_capturer) {
                    simple_capturer->UpdateCaptureRegion(current_offsetX, current_offsetY);
                }
                
                last_offsetX = current_offsetX;
                last_offsetY = current_offsetY;
                ctx.crosshair_offset_changed.store(false);
            }
            
            // IMPORTANT: Update was_aim_shoot OUTSIDE the if block
            was_aim_shoot = is_aim_shoot;

            // Measure only the actual capture API call time
            auto frame_acq_start_time = std::chrono::high_resolution_clock::now();
            static int captureCount = 0;
            ++captureCount;
            
            // GPU-only capture
            if (simple_capturer) {
                screenshotGpu = simple_capturer->GetNextFrameGpu();
            } else if (wingraphics_capturer) {
                screenshotGpu = wingraphics_capturer->GetNextFrameGpu();
            }
            auto frame_acq_end_time = std::chrono::high_resolution_clock::now();
            
            std::chrono::duration<float, std::milli> frame_acq_duration_ms = frame_acq_end_time - frame_acq_start_time;
            ctx.g_current_frame_acquisition_time_ms.store(frame_acq_duration_ms.count());
            ctx.add_to_history(ctx.g_frame_acquisition_time_history, frame_acq_duration_ms.count(), ctx.g_frame_acquisition_history_mutex);

            if (!screenshotGpu.empty()) {
                static int successCount = 0;
                ++successCount;
                
                // Update latest frame for debug display
                if (latestFrameGpu.rows() != screenshotGpu.rows() || latestFrameGpu.cols() != screenshotGpu.cols()) {
                    latestFrameGpu.create(screenshotGpu.rows(), screenshotGpu.cols(), screenshotGpu.channels());
                }
                latestFrameGpu.copyFrom(screenshotGpu);
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
                    const SimpleCudaMat& gpuFrame = captureGpuBuffer[idx];
                    if (!gpuFrame.empty())
                    {
                        SimpleMat tmp(gpuFrame.rows(), gpuFrame.cols(), gpuFrame.channels());
                            gpuFrame.download(tmp.data(), tmp.step());
                            ImageIO::saveScreenshot(tmp);
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
        
        // 파이프라인 정리
        for (auto& pipe : pipeline) {
            if (!pipe.ready && pipe.event) {
                CUDA_CHECK_WARN(cudaEventSynchronize(pipe.event));
            }
            if (pipe.event) {
                CUDA_CHECK_WARN(cudaEventDestroy(pipe.event));
            }
            if (!pipe.gpuFrame.empty()) {
                g_frameBufferPool->releaseGpuBuffer(std::move(pipe.gpuFrame));
            }
        }
        
        // 스트림 정리
        if (captureStream) CUDA_CHECK_WARN(cudaStreamDestroy(captureStream));
        if (processStream) CUDA_CHECK_WARN(cudaStreamDestroy(processStream));
        if (uploadStream) CUDA_CHECK_WARN(cudaStreamDestroy(uploadStream));
        
        // 버퍼 풀 정리 완료
        
        std::cout << "[Capture] Capture thread exiting." << std::endl;

    }
    catch (const std::exception& e)
    {
        std::cerr << "[Capture] Unhandled exception in capture thread: " << e.what() << std::endl;
    }
}