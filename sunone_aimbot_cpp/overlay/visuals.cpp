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

// Assume these are defined elsewhere and updated (e.g., in main or capture)
extern cv::cuda::GpuMat latestFrameGpu;
extern std::mutex frameMutex;
extern std::condition_variable frameCV;
// ---

void displayThread()
{
    // Removed CPU vectors: std::vector<cv::Rect> boxes; std::vector<int> classes;

    if (config.show_window)
    {
        cv::namedWindow(config.window_name, cv::WINDOW_NORMAL);

        if (config.always_on_top)
        {
            cv::setWindowProperty(config.window_name, cv::WND_PROP_TOPMOST, 1);
        }
        else
        {
            cv::setWindowProperty(config.window_name, cv::WND_PROP_TOPMOST, 0);
        }
    }

    int currentSize = 0;
    cv::cuda::GpuMat frameGpu;       // Use GpuMat for the frame
    cv::cuda::GpuMat displayFrameGpu; // Use GpuMat for resizing and drawing
    cv::Mat displayFrameCpu;      // CPU Mat for imshow and text drawing

    // Use a dedicated CUDA stream for visualization tasks if desired, or use default stream
    cv::cuda::Stream visStream; // Optional: Create a stream for visualization

    while (!shouldExit)
    {
        if (show_window_changed.load())
        {
            if (config.show_window)
            {
                cv::namedWindow(config.window_name, cv::WINDOW_NORMAL);
                if (config.always_on_top)
                {
                    cv::setWindowProperty(config.window_name, cv::WND_PROP_TOPMOST, 1);
                }
                else
                {
                    cv::setWindowProperty(config.window_name, cv::WND_PROP_TOPMOST, 0);
                }
            }
            else
            {
                if (cv::getWindowProperty(config.window_name, cv::WND_PROP_VISIBLE) >= 0)
                {
                   cv::destroyWindow(config.window_name);
                }
            }
            show_window_changed.store(false);
        }

        if (config.show_window)
        {
            {
                std::unique_lock<std::mutex> lock(frameMutex);
                frameCV.wait(lock, [] { return !latestFrameGpu.empty() || shouldExit; });
                if (shouldExit) break;
                frameGpu = latestFrameGpu.clone();
            }

            if (frameGpu.empty()) continue; // Skip if frame is empty

            int desiredWidth = frameGpu.cols; // Base width
            int desiredHeight = frameGpu.rows; // Base height
            int displayWidth = desiredWidth;
            int displayHeight = desiredHeight;

            // Assume displayFrameGpu needs update if size changes or it's empty
            bool needsResizeOrCopy = false;

            if (config.window_size != 100)
            {
                displayWidth = static_cast<int>((desiredWidth * config.window_size) / 100.0f);
                displayHeight = static_cast<int>((desiredHeight * config.window_size) / 100.0f);
                if (displayWidth != currentSize || displayFrameGpu.cols != displayWidth || displayFrameGpu.rows != displayHeight || displayFrameGpu.empty()) {
                    cv::resizeWindow(config.window_name, displayWidth, displayHeight);
                    currentSize = displayWidth;
                    needsResizeOrCopy = true;
                }
            }
            else // config.window_size == 100
            {
                 displayWidth = desiredWidth;
                 displayHeight = desiredHeight;
                 if (currentSize != displayWidth || displayFrameGpu.cols != displayWidth || displayFrameGpu.rows != displayHeight || displayFrameGpu.empty()) {
                    cv::resizeWindow(config.window_name, displayWidth, displayHeight);
                    currentSize = displayWidth;
                    needsResizeOrCopy = true; // Need to copy/clone even if size is same but buffer is invalid/empty
                 }
            }

            // Perform resize or copy to displayFrameGpu
            if (needsResizeOrCopy) {
                if (config.window_size != 100) {
                    cv::cuda::resize(frameGpu, displayFrameGpu, cv::Size(displayWidth, displayHeight), 0, 0, cv::INTER_LINEAR, visStream);
                } else {
                    // If size is 100, displayFrameGpu should be a copy of frameGpu
                    frameGpu.copyTo(displayFrameGpu, visStream); // Use copyTo if no modification needed before download
                }
            } else {
                 // If no resize/copy needed, make sure displayFrameGpu still references the correct data for download
                 // This case might be tricky if frameGpu is modified elsewhere. Safest might be copy always if unsure.
                 // Let's assume if needsResizeOrCopy is false, displayFrameGpu is already correct from previous iteration or not needed.
                 // If drawing happened *on* displayFrameGpu, it would need cloning. Since drawing is moved to CPU, copyTo is ok.
                  if (displayFrameGpu.empty()) { // Ensure it's initialized if this is the first frame and size is 100
                      frameGpu.copyTo(displayFrameGpu, visStream);
                  }
            }

            std::string labelText = ""; // Store text label to draw later on CPU

            Detection bestTargetHost; // Local struct to hold the best target copied from Host (already copied in detector)
            bool hasTarget = false;
            cv::Rect scaledBox;       // Scaled box coordinates for drawing

            { // Mutex lock scope for accessing detector results (reading host variables)
                std::lock_guard<std::mutex> lock(detector.detectionMutex);
                if (detector.m_hasBestTarget) // Use the flag set after successful DtoH copy in Detector
                {
                    hasTarget = true;
                    bestTargetHost = detector.m_bestTargetHost; // Copy from the host member

                    // --- Scaling calculations (can stay on CPU) ---
                    cv::Rect box = bestTargetHost.box; // Original box from detection resolution

                    // 1. Scale box from detection resolution to original frame resolution (frameGpu size)
                    float box_scale_x = static_cast<float>(frameGpu.cols) / config.detection_resolution;
                    float box_scale_y = static_cast<float>(frameGpu.rows) / config.detection_resolution;

                    scaledBox.x = static_cast<int>(box.x * box_scale_x);
                    scaledBox.y = static_cast<int>(box.y * box_scale_y);
                    scaledBox.width = static_cast<int>(box.width * box_scale_x);
                    scaledBox.height = static_cast<int>(box.height * box_scale_y);

                    // 2. Scale box from original frame resolution to display frame resolution (displayFrameGpu size)
                    float resize_scale_x = static_cast<float>(displayFrameGpu.cols) / frameGpu.cols;
                    float resize_scale_y = static_cast<float>(displayFrameGpu.rows) / frameGpu.rows;

                    scaledBox.x = static_cast<int>(scaledBox.x * resize_scale_x);
                    scaledBox.y = static_cast<int>(scaledBox.y * resize_scale_y);
                    scaledBox.width = static_cast<int>(scaledBox.width * resize_scale_x);
                    scaledBox.height = static_cast<int>(scaledBox.height * resize_scale_y);

                    // 3. Clamp box to display frame boundaries
                    scaledBox.x = std::max(0, scaledBox.x);
                    scaledBox.y = std::max(0, scaledBox.y);
                    scaledBox.width = std::min(displayFrameGpu.cols - scaledBox.x, scaledBox.width);
                    scaledBox.height = std::min(displayFrameGpu.rows - scaledBox.y, scaledBox.height);

                    // Prepare label text (on CPU)
                    labelText = "ID:" + std::to_string(bestTargetHost.classId); // + " C:" + std::to_string(bestTargetHost.confidence).substr(0,4);
                }
            } // Mutex released here

            // --- Download the frame to CPU for text overlay and display ---
             if (!displayFrameGpu.empty()) {
                visStream.waitForCompletion(); // Ensure drawing/resizing is done before download
                displayFrameGpu.download(displayFrameCpu, visStream);
                visStream.waitForCompletion(); // Ensure download is done before CPU access
             } else if (!frameGpu.empty()) {
                 // Fallback: If displayFrameGpu wasn't created (e.g., size 100, no resize), download original frameGpu
                 frameGpu.download(displayFrameCpu, visStream);
                 visStream.waitForCompletion();
             } else {
                 // If both are empty, skip display
                 continue;
             }

            // --- Draw overlays on CPU Mat ---
            if (!displayFrameCpu.empty()) {
                // Draw rectangle on CPU if a target exists
                if (hasTarget && scaledBox.width > 0 && scaledBox.height > 0)
                {
                    cv::rectangle(displayFrameCpu, scaledBox, cv::Scalar(0, 255, 0), 2); // Draw on CPU Mat
                }

                // Draw label text on CPU
                if (hasTarget) {
                    // Draw label text near the (scaled) box
                    cv::Point textOrg(scaledBox.x, std::max(0, scaledBox.y - 5)); // Position above box, clamp y
                    cv::putText(displayFrameCpu, labelText, textOrg,
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
                }

                if (config.show_fps && !displayFrameCpu.empty())
                {
                    // Draw FPS text
                    cv::putText(displayFrameCpu, "FPS: " + std::to_string(static_cast<int>(captureFps)), cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 0), 2);
                }
            }

            // --- Display the final CPU frame ---
            try
            {
                 if (!displayFrameCpu.empty()) {
                     cv::imshow(config.window_name, displayFrameCpu);
                 }
            }
            catch (cv::Exception& e)
            {
                std::cerr << "[Visuals]: OpenCV display error: " << e.what() << std::endl;
                // Consider breaking or handling the error appropriately
                // break;
            }

            if (cv::waitKey(1) == 27) shouldExit = true;
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }

    // Cleanup: Use getWindowProperty again
    if (cv::getWindowProperty(config.window_name, cv::WND_PROP_VISIBLE) >= 0)
    {
        cv::destroyWindow(config.window_name);
    }
}