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
// #include "winrt_capture.h" // Removed
// #include "virtual_camera.h" // Removed for now, can be added back if needed

// Assume detector is globally accessible or passed to captureThread
extern Detector detector;
extern std::mutex configMutex; // Declare configMutex as external

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "windowsapp.lib")

// Define global variables used across threads
cv::cuda::GpuMat latestFrameGpu;
std::mutex frameMutex;
cv::Mat latestFrameCpu;
std::atomic<bool> newFrameAvailable = false; // Definition for frame notification flag

// Define other global variables as needed
int screenWidth = 0;
int screenHeight = 0;
std::atomic<int> captureFrameCount(0);
std::atomic<int> captureFps(0);
std::chrono::time_point<std::chrono::high_resolution_clock> captureFpsStartTime;

// Helper function to create a capturer instance based on config - REMOVED
/*
IScreenCapture* createCapturer(int width, int height) {
    IScreenCapture* capturer = nullptr;
    const std::string& method = config.capture_method; // Cache config value for efficiency

    if (method == "duplication_api") {
        capturer = new DuplicationAPIScreenCapture(width, height);
        if (config.verbose) std::cout << "[Capture] Using Duplication API." << std::endl;
    } else if (method == "winrt") {
        // WinRT apartment should be initialized by the calling thread (captureThread)
        capturer = new WinRTScreenCapture(width, height);
        if (config.verbose) std::cout << "[Capture] Using WinRT." << std::endl;
    } else if (method == "virtual_camera") {
        capturer = new VirtualCameraCapture(width, height);
        if (config.verbose) std::cout << "[Capture] Using virtual camera input." << std::endl;
    } else {
        // Handle unknown capture method by setting and saving a default
        std::cout << "[Capture] Unknown capture method '" << method << "'. Setting default: duplication_api." << std::endl;
        config.capture_method = "duplication_api"; // Update config object
        config.saveConfig(); // Persist the change
        capturer = new DuplicationAPIScreenCapture(width, height); // Create the default capturer
        if (config.verbose) std::cout << "[Capture] Using Duplication API." << std::endl;
    }
    return capturer;
}
*/

void captureThread(int CAPTURE_WIDTH, int CAPTURE_HEIGHT)
{
    // Initialize WinRT apartment once at the start of the thread. - REMOVED
    // winrt::init_apartment(winrt::apartment_type::multi_threaded);
    // Use RAII to ensure uninitialization even if exceptions occur. - REMOVED
    // struct WinRTUninitializer { ~WinRTUninitializer() { winrt::uninit_apartment(); } } winrtUninitializer;

    try
    {
        if (config.verbose)
        {
            std::cout << "[Capture] OpenCV version: " << CV_VERSION << std::endl;
            std::cout << "[Capture] CUDA Support: " << cv::cuda::getCudaEnabledDeviceCount() << " devices found." << std::endl;
        }

        // Use the helper function for initial capturer creation - REPLACED
        // Use std::unique_ptr for automatic memory management
        // std::unique_ptr<IScreenCapture> capturer(createCapturer(CAPTURE_WIDTH, CAPTURE_HEIGHT));
        std::unique_ptr<DuplicationAPIScreenCapture> capturer = std::make_unique<DuplicationAPIScreenCapture>(CAPTURE_WIDTH, CAPTURE_HEIGHT);
        if (!capturer) {
             std::cerr << "[Capture] Failed to initialize DuplicationAPIScreenCapture!" << std::endl;
             return; // Exit thread if initialization fails
        }

        // --- Initial setting of capture event for the detector ---
        if (capturer) { // Check if capturer is valid
            detector.setCaptureEvent(capturer->GetCaptureDoneEvent());
        }
        // --- End initial setting ---

        // cv::cuda::GpuMat latestFrameGpu; // This is a global variable now, remove local declaration
        bool buttonPreviouslyPressed = false;

        auto lastSaveTime = std::chrono::steady_clock::now();

        std::optional<std::chrono::duration<double, std::milli>> frame_duration;
        bool frameLimitingEnabled = false;

        if (config.capture_fps > 0.0)
        {
            timeBeginPeriod(1); // Request higher timer resolution
            frame_duration = std::chrono::duration<double, std::milli>(1000.0 / config.capture_fps);
            frameLimitingEnabled = true;
        }

        captureFpsStartTime = std::chrono::high_resolution_clock::now();
        auto start_time = std::chrono::high_resolution_clock::now(); // For FPS limiting

        while (!shouldExit)
        {
            // --- FPS Limiting Configuration Update ---
            if (capture_fps_changed.load())
            {
                if (config.capture_fps > 0.0)
                {
                    if (!frameLimitingEnabled)
                    {
                        timeBeginPeriod(1); // Enable high-resolution timer if not already
                        frameLimitingEnabled = true;
                    }
                    frame_duration = std::chrono::duration<double, std::milli>(1000.0 / config.capture_fps);
                }
                else // capture_fps <= 0 means no limit
                {
                    if (frameLimitingEnabled)
                    {
                        timeEndPeriod(1); // Release high-resolution timer if no longer needed
                        frameLimitingEnabled = false;
                    }
                    frame_duration = std::nullopt; // Remove frame duration limit
                }
                capture_fps_changed.store(false); // Reset the flag
            }

            // --- Capturer Re-creation on Configuration Change ---
            // Check if any relevant configuration affecting the capturer has changed.
            // Note: Some changes like cursor/border visibility might be handled internally
            // by specific capturer implementations without needing full recreation.
            // This check assumes recreation is necessary for these changes.
            if (detection_resolution_changed.load() ||
                // capture_method_changed.load() || // Removed
                capture_cursor_changed.load() ||
                capture_borders_changed.load())
            {
                // Clean up the old capturer instance (unique_ptr does this automatically via reset)

                // Get new dimensions from config
                int new_CAPTURE_WIDTH = config.detection_resolution;
                int new_CAPTURE_HEIGHT = config.detection_resolution;

                // Use the helper function to create the new capturer - REPLACED
                // capturer.reset(createCapturer(new_CAPTURE_WIDTH, new_CAPTURE_HEIGHT)); // unique_ptr handles deleting the old one
                capturer.reset(new DuplicationAPIScreenCapture(new_CAPTURE_WIDTH, new_CAPTURE_HEIGHT));
                if (!capturer) {
                    std::cerr << "[Capture] Failed to re-initialize DuplicationAPIScreenCapture after config change!" << std::endl;
                    break; // Exit the loop if re-initialization fails
                }

                // --- Update capture event for the detector after recreation ---
                if (capturer) { // Check if new capturer is valid
                    detector.setCaptureEvent(capturer->GetCaptureDoneEvent());
                }
                // --- End update ---

                // Update global screen dimensions (Consider getting actual dimensions from capturer if possible)
                screenWidth = new_CAPTURE_WIDTH;
                screenHeight = new_CAPTURE_HEIGHT;

                // Reset the change flags
                detection_resolution_changed.store(false);
                // capture_method_changed.store(false); // Removed
                capture_cursor_changed.store(false);
                capture_borders_changed.store(false);
            }

            // --- Frame Capture and Processing ---
            cv::cuda::GpuMat screenshotGpu;
            cv::Mat screenshotCpu;

            if (config.capture_use_cuda) { // Assumes config.capture_use_cuda exists
                screenshotGpu = capturer->GetNextFrameGpu();

                if (!screenshotGpu.empty())
                {
                    // REMOVED Preprocessing (Resizing/Masking) - Moved to Detector
                    // cv::cuda::GpuMat processedFrameGpu;
                    // // Apply circle mask if enabled (GPU version)
                    // if (config.circle_mask)
                    // {
                    //     cv::Mat mask = cv::Mat::zeros(screenshotGpu.size(), CV_8UC1);
                    //     cv::Point center(mask.cols / 2, mask.rows / 2);
                    //     int radius = std::min(mask.cols, mask.rows) / 2;
                    //     cv::circle(mask, center, radius, cv::Scalar(255), -1);
                    //     cv::cuda::GpuMat maskGpu;
                    //     maskGpu.upload(mask);
                    //     cv::cuda::GpuMat maskedImageGpu;
                    //     screenshotGpu.copyTo(maskedImageGpu, maskGpu);
                    //     cv::cuda::resize(maskedImageGpu, processedFrameGpu, cv::Size(640, 640), 0, 0, cv::INTER_LINEAR);
                    // }
                    // else
                    // {
                    //     cv::cuda::resize(screenshotGpu, processedFrameGpu, cv::Size(640, 640));
                    // }

                    // Send the RAW captured frame to the detector thread
                    // The detector will handle preprocessing (resize, mask, etc.)
                    detector.processFrame(screenshotGpu); // Pass raw frame

                    // Update Shared Data and Notify Other Threads
                    // WARNING: latestFrameGpu/Cpu now hold the RAW captured frame,
                    // not the resized/processed one, unless detector updates them.
                    // Consider if this download is still necessary or should be removed.
                    {
                        std::lock_guard<std::mutex> lock(frameMutex);
                        latestFrameGpu = screenshotGpu.clone(); // Update global GPU frame (RAW)
                        {
                           std::lock_guard<std::mutex> config_lock(configMutex); // Lock config mutex before reading config
                            if (config.show_window) // Only download to CPU if the debug window is shown
                            {
                                screenshotGpu.download(latestFrameCpu);
                            }
                        }
                        newFrameAvailable = true; // Set flag indicating a frame (GPU or maybe CPU via download) is ready
                        frameCV.notify_one(); // Notify display thread
                    }
                }
                else
                {
                    // Handle empty GPU frame if necessary (e.g., log a warning)
                }
            } else { // Use CPU capture path
                screenshotCpu = capturer->GetNextFrameCpu();

                if (!screenshotCpu.empty())
                {
                    // REMOVED CPU Preprocessing - Will be handled by detector
                    /*
                    cv::Mat processedFrameCpu;

                    // Apply circle mask if enabled (CPU version)
                    if (config.circle_mask)
                    {
                        cv::Mat mask = cv::Mat::zeros(screenshotCpu.size(), CV_8UC1);
                        cv::Point center(mask.cols / 2, mask.rows / 2);
                        int radius = std::min(mask.cols, mask.rows) / 2;
                        cv::circle(mask, center, radius, cv::Scalar(255), -1);
                        cv::Mat maskedImageCpu;
                        screenshotCpu.copyTo(maskedImageCpu, mask);
                        cv::resize(maskedImageCpu, processedFrameCpu, cv::Size(640, 640), 0, 0, cv::INTER_LINEAR);
                    }
                    else
                    {
                        cv::resize(screenshotCpu, processedFrameCpu, cv::Size(640, 640));
                    }

                    // Send the processed frame to the detector thread (needs upload)
                    cv::cuda::GpuMat processedFrameGpuForDetector;
                    processedFrameGpuForDetector.upload(processedFrameCpu);
                    detector.processFrame(processedFrameGpuForDetector);
                    */

                    // Send the RAW CPU frame to the detector thread
                    // Assumes a new detector method like processFrameCpu exists or processFrame is overloaded
                    detector.processFrame(screenshotCpu); // Pass raw CPU frame

                    // Update Shared Data and Notify Other Threads
                    {
                        std::lock_guard<std::mutex> lock(frameMutex);
                        // Update global CPU frame with the raw captured frame
                        latestFrameCpu = screenshotCpu.clone();
                        newFrameAvailable = true; // Set flag indicating a CPU frame is ready
                        // Remove GPU frame update here, detector will handle GpuMat creation
                        // latestFrameGpu = processedFrameGpuForDetector.clone();
                    }
                    frameCV.notify_one(); // Notify display thread
                }
                else
                {
                     // Handle empty CPU frame if necessary (e.g., log a warning)
                }
            }

            // --- FPS Calculation ---
            captureFrameCount++;
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed_fps = std::chrono::duration_cast<std::chrono::seconds>(now - captureFpsStartTime).count();
            if (elapsed_fps >= 1)
            {
                captureFps = captureFrameCount.load();
                captureFrameCount = 0;
                captureFpsStartTime = now;
            }

            // --- Screenshot Logic (Uses latestFrameCpu, which is updated in both paths) ---
            if (!config.screenshot_button.empty() && config.screenshot_button[0] != "None")
            {
                bool buttonPressed = isAnyKeyPressed(config.screenshot_button);
                auto now_ss = std::chrono::steady_clock::now(); // Use a different 'now' variable
                auto elapsed_ss = std::chrono::duration_cast<std::chrono::milliseconds>(now_ss - lastSaveTime).count();

                if (buttonPressed && !buttonPreviouslyPressed && elapsed_ss > 1000) // 1-second cooldown
                {
                    std::lock_guard<std::mutex> lock(frameMutex); // Lock mutex before accessing frames
                    
                    // FIX: Conditionally download the latest GPU frame to CPU just for the screenshot
                    //      if using the GPU capture path and the GPU frame is available.
                    if (config.capture_use_cuda && !latestFrameGpu.empty()) {
                        latestFrameGpu.download(latestFrameCpu); 
                    }
                    
                    // Now save latestFrameCpu (which is now up-to-date even in GPU mode)
                    if (!latestFrameCpu.empty())
                    {
                        saveScreenshot(latestFrameCpu);
                        lastSaveTime = now_ss; // Update last save time
                    }
                }
                buttonPreviouslyPressed = buttonPressed;
            }

            // --- FPS Limiting ---
            if (frameLimitingEnabled && frame_duration.has_value())
            {
                auto end_time = std::chrono::high_resolution_clock::now();
                auto elapsed_time = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end_time - start_time);
                auto sleep_duration = *frame_duration - elapsed_time;

                if (sleep_duration.count() > 0)
                {
                     // Use std::this_thread::sleep_for for better precision if needed
                     std::this_thread::sleep_for(sleep_duration);
                }
                 start_time = std::chrono::high_resolution_clock::now(); // Reset start time for next frame
            } else {
                // Reset start time even if not limiting, to avoid drift if limiting is re-enabled
                start_time = std::chrono::high_resolution_clock::now();
            }
        } // End of while (!shouldExit) loop

        // --- Cleanup Before Exiting Thread ---
        if (frameLimitingEnabled)
        {
            timeEndPeriod(1); // Release high-resolution timer
        }
        // delete capturer; // Delete the final capturer instance - Not needed with unique_ptr

        // WinRT apartment is automatically uninitialized by winrtUninitializer destructor when the thread exits.
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Capture] Unhandled exception in capture thread: " << e.what() << std::endl;
        // Consider adding cleanup logic here as well (e.g., delete capturer if allocated) - Not needed with unique_ptr
        // The RAII WinRTUninitializer will still handle apartment uninitialization.
    }
    // winrt::uninit_apartment(); // Not needed if using RAII wrapper
}