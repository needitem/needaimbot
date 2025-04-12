#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <iostream>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <cuda_runtime_api.h>

#include "visuals.h"
#include "config.h"
#include "sunone_aimbot_cpp.h"
#include "capture.h"

extern std::atomic<bool> show_window_changed;
extern std::mutex configMutex; // Declare configMutex as external

// Assume these are defined elsewhere and updated (e.g., in main or capture)
extern cv::cuda::GpuMat latestFrameGpu;
extern std::mutex frameMutex;
extern std::condition_variable frameCV;
// ---

void displayThread()
{
    int currentSize = 0;
    // We need both GPU and CPU Mats for different paths
    cv::cuda::GpuMat frameGpu;
    cv::cuda::GpuMat displayFrameGpu;
    cv::Mat frameCpu;
    cv::Mat displayFrameCpu; // This will hold the final image for imshow
    cv::cuda::Stream visStream;
    bool window_created = false;

    while (!shouldExit)
    {
        bool show_window_flag_changed = show_window_changed.load();
        bool should_show_window = false;
        bool should_be_always_on_top = false;
        bool use_cuda = false; // Capture mode
        int window_size = 100;
        bool show_fps = false;

        {
            std::lock_guard<std::mutex> lock(configMutex);
            should_show_window = config.show_window;
            should_be_always_on_top = config.always_on_top;
            use_cuda = config.capture_use_cuda;
            window_size = config.window_size;
            show_fps = config.show_fps;
        }

        if (show_window_flag_changed) {
            show_window_changed.store(false);
        }

        // --- Window Lifecycle Management ---
        if (should_show_window) {
             if (!window_created) {
                 cv::namedWindow(config.window_name, cv::WINDOW_NORMAL);
                 cv::setWindowProperty(config.window_name, cv::WND_PROP_TOPMOST, should_be_always_on_top ? 1 : 0);
                 window_created = true;
                 currentSize = 0;
                 // Clear buffers when window is created
                 displayFrameGpu = cv::cuda::GpuMat();
                 displayFrameCpu = cv::Mat();
             }
             else if (show_window_flag_changed) {
                  cv::setWindowProperty(config.window_name, cv::WND_PROP_TOPMOST, should_be_always_on_top ? 1 : 0);
             }

            // --- Frame Acquisition ---
            bool frameValid = false;
            int originalWidth = 0;
            int originalHeight = 0;
            {
                std::unique_lock<std::mutex> lock(frameMutex);
                frameCV.wait(lock, [&] { return newFrameAvailable || shouldExit; });

                if (shouldExit) break;

                if (newFrameAvailable) {
                    if (use_cuda) {
                        if (!latestFrameGpu.empty()) {
                            frameGpu = latestFrameGpu.clone(); // Clone GPU frame
                            originalWidth = frameGpu.cols;
                            originalHeight = frameGpu.rows;
                            frameValid = true;
                        }
                    } else { // CPU Mode
                        if (!latestFrameCpu.empty()) {
                            frameCpu = latestFrameCpu.clone(); // Clone CPU frame
                            originalWidth = frameCpu.cols;
                            originalHeight = frameCpu.rows;
                            frameValid = true;
                        }
                    }
                    newFrameAvailable = false; // Reset flag
                }
            } // frameMutex unlocked

            if (!frameValid || originalWidth == 0 || originalHeight == 0) continue;

            // --- Resizing Logic (GPU or CPU path) ---
            int displayWidth = originalWidth;
            int displayHeight = originalHeight;
            bool needsResize = false;

            if (window_size != 100) {
                displayWidth = static_cast<int>((originalWidth * window_size) / 100.0f);
                displayHeight = static_cast<int>((originalHeight * window_size) / 100.0f);
            }

            // Check if resize is needed OR if target buffer size is wrong
            if (currentSize != displayWidth || // Window size changed
                (use_cuda && (displayFrameGpu.cols != displayWidth || displayFrameGpu.rows != displayHeight)) ||
                (!use_cuda && (displayFrameCpu.cols != displayWidth || displayFrameCpu.rows != displayHeight)) ||
                (use_cuda && displayFrameGpu.empty()) || // Buffer empty
                (!use_cuda && displayFrameCpu.empty()) )
            {
                 needsResize = true; // Need resize or initial copy
                 if (window_created && displayWidth > 0 && displayHeight > 0) {
                     cv::resizeWindow(config.window_name, displayWidth, displayHeight);
                 }
                 currentSize = displayWidth;
            }

            // Perform Resize/Copy
            if (use_cuda) {
                bool resize_performed = false;
                // Check if window/buffer size mismatch requires action
                if (needsResize) {
                    if (window_size != 100 && displayWidth > 0 && displayHeight > 0) {
                         // Perform GPU resize
                         cv::cuda::resize(frameGpu, displayFrameGpu, cv::Size(displayWidth, displayHeight), 0, 0, cv::INTER_LINEAR, visStream);
                         resize_performed = true;
                    }
                    // If window_size is 100, needsResize might be true only because displayFrameGpu was empty/wrong size.
                    // The copy will happen below if resize wasn't performed.
                }

                // If resize was NOT performed (either needsResize was false, or window_size was 100),
                // ensure displayFrameGpu is updated with the current frame's data using copyTo.
                if (!resize_performed) {
                    frameGpu.copyTo(displayFrameGpu, visStream);
                }

                 // Now download the final displayFrameGpu (it's guaranteed to be updated now)
                 if (!displayFrameGpu.empty()){
                     visStream.waitForCompletion();
                     displayFrameGpu.download(displayFrameCpu, visStream);
                     visStream.waitForCompletion();
                 } else {
                     std::cerr << "[Visuals] Error: displayFrameGpu is empty before download!" << std::endl;
                     continue;
                 }

            } else { // CPU Mode
                if (needsResize) {
                     if (window_size != 100 && displayWidth > 0 && displayHeight > 0) {
                          cv::resize(frameCpu, displayFrameCpu, cv::Size(displayWidth, displayHeight), 0, 0, cv::INTER_LINEAR);
                     } else {
                          displayFrameCpu = frameCpu; // Use directly if no resize or size 100
                     }
                } else if (displayFrameCpu.empty()) { // Initial frame case without resize
                    displayFrameCpu = frameCpu;
                }
                 // displayFrameCpu now holds the correct CPU image
                 if (displayFrameCpu.empty()) {
                     std::cerr << "[Visuals] Error: displayFrameCpu is empty in CPU mode!" << std::endl;
                     continue;
                 }
            }


            // --- Detection Overlay Logic (Draws on displayFrameCpu) ---
             Detection bestTargetHost;
             bool hasTarget = false;
             cv::Rect scaledBox;
             std::string labelText = "";
             {
                 std::lock_guard<std::mutex> lock(detector.detectionMutex);
                 if (detector.m_hasBestTarget)
                 {
                     hasTarget = true;
                     bestTargetHost = detector.m_bestTargetHost;
                     // Scaling calculations (From detection res to display size: displayWidth/displayHeight)
                     cv::Rect box = bestTargetHost.box;
                     float scale_x = static_cast<float>(displayWidth) / config.detection_resolution;
                     float scale_y = static_cast<float>(displayHeight) / config.detection_resolution;

                     scaledBox.x = static_cast<int>(box.x * scale_x);
                     scaledBox.y = static_cast<int>(box.y * scale_y);
                     scaledBox.width = static_cast<int>(box.width * scale_x);
                     scaledBox.height = static_cast<int>(box.height * scale_y);

                     // Clamp box
                     scaledBox.x = std::max(0, scaledBox.x);
                     scaledBox.y = std::max(0, scaledBox.y);
                     scaledBox.width = std::min(displayWidth - scaledBox.x, scaledBox.width);
                     scaledBox.height = std::min(displayHeight - scaledBox.y, scaledBox.height);

                     labelText = "ID:" + std::to_string(bestTargetHost.classId);
                 }
             }

            // --- Draw on CPU (displayFrameCpu) ---
            if (!displayFrameCpu.empty()) {
                // Draw detection box
                if (hasTarget && scaledBox.width > 0 && scaledBox.height > 0) {
                    cv::rectangle(displayFrameCpu, scaledBox, cv::Scalar(0, 255, 0), 2);
                    cv::Point textOrg(scaledBox.x, std::max(0, scaledBox.y - 5));
                    cv::putText(displayFrameCpu, labelText, textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
                }
                // Draw FPS (Use value read earlier under lock)
                if (show_fps) {
                     cv::putText(displayFrameCpu, "FPS: " + std::to_string(static_cast<int>(captureFps)), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 0), 2);
                }
            }

            // --- Show Frame (displayFrameCpu) ---
            try {
                 if (!displayFrameCpu.empty()) {
                     cv::imshow(config.window_name, displayFrameCpu);
                 }
            } catch (const cv::Exception& e) {
                std::cerr << "[Visuals]: OpenCV display error: " << e.what() << std::endl;
                 if(window_created) {
                      if (cv::getWindowProperty(config.window_name, cv::WND_PROP_VISIBLE) >= 0) {
                         cv::destroyWindow(config.window_name);
                      }
                      window_created = false;
                 }
            }

            // --- Handle OpenCV events ---
            if (cv::waitKey(1) == 27) { shouldExit = true; }

        } else { // should_show_window is false
            if (window_created) {
                if (cv::getWindowProperty(config.window_name, cv::WND_PROP_VISIBLE) >= 0) {
                    cv::destroyWindow(config.window_name);
                }
                window_created = false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    } // End while loop

    if (window_created) {
        if (cv::getWindowProperty(config.window_name, cv::WND_PROP_VISIBLE) >= 0) {
            cv::destroyWindow(config.window_name);
        }
    }
}