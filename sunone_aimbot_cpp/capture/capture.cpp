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

#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

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
#include "winrt_capture.h"
#include "virtual_camera.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "windowsapp.lib")

// Helper function to create a capturer instance based on config
// This function encapsulates the logic for selecting and initializing the appropriate capturer.
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

cv::cuda::GpuMat latestFrameGpu;
cv::Mat latestFrameCpu;

std::mutex frameMutex;

int screenWidth = 0;
int screenHeight = 0;

std::atomic<int> captureFrameCount(0);
std::atomic<int> captureFps(0);
std::chrono::time_point<std::chrono::high_resolution_clock> captureFpsStartTime;

void captureThread(int CAPTURE_WIDTH, int CAPTURE_HEIGHT)
{
    // Initialize WinRT apartment once at the start of the thread.
    // This is necessary for the WinRT capture method and safe to call even if not used.
    winrt::init_apartment(winrt::apartment_type::multi_threaded);
    // Use RAII to ensure uninitialization even if exceptions occur.
    struct WinRTUninitializer { ~WinRTUninitializer() { winrt::uninit_apartment(); } } winrtUninitializer;

    try
    {
        if (config.verbose)
        {
            std::cout << "[Capture] OpenCV version: " << CV_VERSION << std::endl;
            std::cout << "[Capture] CUDA Support: " << cv::cuda::getCudaEnabledDeviceCount() << " devices found." << std::endl;
        }

        // Use the helper function for initial capturer creation
        IScreenCapture* capturer = createCapturer(CAPTURE_WIDTH, CAPTURE_HEIGHT);
        if (!capturer) {
             std::cerr << "[Capture] Failed to initialize capturer!" << std::endl;
             return; // Exit thread if initialization fails
        }

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
                capture_method_changed.load() ||
                capture_cursor_changed.load() ||
                capture_borders_changed.load())
            {
                delete capturer; // Clean up the old capturer instance

                // Get new dimensions from config
                int new_CAPTURE_WIDTH = config.detection_resolution;
                int new_CAPTURE_HEIGHT = config.detection_resolution;

                // Use the helper function to create the new capturer
                capturer = createCapturer(new_CAPTURE_WIDTH, new_CAPTURE_HEIGHT);
                if (!capturer) {
                    std::cerr << "[Capture] Failed to re-initialize capturer after config change!" << std::endl;
                    break; // Exit the loop if re-initialization fails
                }

                // Update global screen dimensions (Consider getting actual dimensions from capturer if possible)
                screenWidth = new_CAPTURE_WIDTH;
                screenHeight = new_CAPTURE_HEIGHT;

                // Reset the change flags
                detection_resolution_changed.store(false);
                capture_method_changed.store(false);
                capture_cursor_changed.store(false);
                capture_borders_changed.store(false);
            }

            // --- Frame Capture and Processing ---
            cv::cuda::GpuMat screenshotGpu = capturer->GetNextFrame();

            if (!screenshotGpu.empty())
            {
                cv::cuda::GpuMat processedFrame;

                // Apply circle mask if enabled
                if (config.circle_mask)
                {
                    // Create a circular mask
                    cv::Mat mask = cv::Mat::zeros(screenshotGpu.size(), CV_8UC1);
                    cv::Point center(mask.cols / 2, mask.rows / 2);
                    int radius = std::min(mask.cols, mask.rows) / 2;
                    cv::circle(mask, center, radius, cv::Scalar(255), -1);

                    // Upload mask to GPU and apply it
                    cv::cuda::GpuMat maskGpu;
                    maskGpu.upload(mask);
                    cv::cuda::GpuMat maskedImageGpu;
                    screenshotGpu.copyTo(maskedImageGpu, maskGpu);

                    // Resize the masked image
                    // Using INTER_LINEAR for resizing is a good balance between speed and quality.
                    cv::cuda::resize(maskedImageGpu, processedFrame, cv::Size(640, 640), 0, 0, cv::INTER_LINEAR);
                }
                else
                {
                    // Resize the original screenshot directly
                    cv::cuda::resize(screenshotGpu, processedFrame, cv::Size(640, 640));
                }

                // --- Update Shared Data and Notify Other Threads ---
                {
                    // Update the global GPU frame under mutex protection
                    std::lock_guard<std::mutex> lock(frameMutex);
                    latestFrameGpu = processedFrame.clone(); // Store a clone for the display thread
                }

                // Send the processed frame to the detector thread
                detector.processFrame(processedFrame);

                // Download the processed frame to CPU for display/overlay thread
                // download() copies data, so latestFrameCpu will hold the new frame data.
                processedFrame.download(latestFrameCpu);
                {
                    // Lock mutex to ensure consistency when display thread reads latestFrameCpu
                    std::lock_guard<std::mutex> lock(frameMutex);
                    // No need to clone latestFrameCpu again, download already updated it.
                }
                // Notify the display thread that a new CPU frame is available
                frameCV.notify_one();

                // --- Screenshot Logic ---
                if (!config.screenshot_button.empty() && config.screenshot_button[0] != "None")
                {
                    bool buttonPressed = isAnyKeyPressed(config.screenshot_button);
                    auto now = std::chrono::steady_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastSaveTime).count();

                    // Check if the button is pressed and enough time has passed since the last save
                    if (buttonPressed && elapsed >= config.screenshot_delay)
                    {
                        cv::Mat resizedCpu; // Use a temporary Mat for the screenshot
                        processedFrame.download(resizedCpu); // Download the frame to save
                        auto epoch_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch()).count();
                        std::string filename = std::to_string(epoch_time) + ".jpg";
                        // Ensure the "screenshots" directory exists (checked at program start)
                        cv::imwrite("screenshots/" + filename, resizedCpu);
                        lastSaveTime = now; // Update the last save time
                    }
                }

                // --- FPS Calculation ---
                captureFrameCount++;
                auto currentTime = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsedTime = currentTime - captureFpsStartTime;
                if (elapsedTime.count() >= 1.0) // Update FPS counter every second
                {
                    captureFps = static_cast<int>(captureFrameCount / elapsedTime.count());
                    captureFrameCount = 0; // Reset frame count for the next second
                    captureFpsStartTime = currentTime; // Reset start time
                }
            } // End if (!screenshotGpu.empty())

            // --- FPS Limiting Sleep ---
            if (frame_duration.has_value()) // Check if frame limiting is enabled
            {
                auto end_time = std::chrono::high_resolution_clock::now();
                auto work_duration = end_time - start_time;
                auto sleep_duration = frame_duration.value() - work_duration;

                // Sleep only if there's remaining time in the frame budget
                if (sleep_duration > std::chrono::duration<double, std::milli>(0))
                {
                    // Use sleep_for for accurate waiting
                    std::this_thread::sleep_for(sleep_duration);
                }
                // Update start_time for the next iteration's measurement
                start_time = std::chrono::high_resolution_clock::now();
            }
        } // End while (!shouldExit)

        // --- Cleanup Before Exiting Thread ---
        if (frameLimitingEnabled)
        {
            timeEndPeriod(1); // Release high-resolution timer
        }
        delete capturer; // Delete the final capturer instance

        // WinRT apartment is automatically uninitialized by winrtUninitializer destructor when the thread exits.
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Capture] Unhandled exception in capture thread: " << e.what() << std::endl;
        // Consider adding cleanup logic here as well (e.g., delete capturer if allocated)
        // The RAII WinRTUninitializer will still handle apartment uninitialization.
    }
    // winrt::uninit_apartment(); // Not needed if using RAII wrapper
}