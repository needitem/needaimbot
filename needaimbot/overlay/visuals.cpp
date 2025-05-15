#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
// #include <opencv2/cudawarping.hpp> // cudawarping for resize might not be needed if we don't resize here
#include <iostream>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <cuda_runtime_api.h>
#include <cstdio> // For printf

#include "visuals.h"
#include "config.h"
#include "needaimbot.h" // For detector.detectionMutex, detector.m_hasBestTarget etc. if any debug drawing were to be kept (it's removed)
#include "capture.h"   // For latestFrameGpu, latestFrameCpu, frameMutex, frameCV etc.

// extern std::atomic<bool> show_window_changed; // Not used anymore by this thread
extern std::mutex configMutex; 

// Assume these are defined elsewhere and updated (e.g., in main or capture)
// extern cv::cuda::GpuMat latestFrameGpu; // Already in capture.h
// extern cv::Mat latestFrameCpu;         // Already in capture.h
// extern std::mutex frameMutex;           // Already in capture.h
// extern std::condition_variable frameCV;// Already in capture.h
// ---

void displayThread()
{
    // printf("[DisplayThread] Thread started.\n");
    cv::cuda::GpuMat acquiredGpuFrame;
    cv::Mat acquiredCpuFrame;
    cv::cuda::Stream processingStream; // Optional stream for GPU operations

    int loop_count = 0;

    while (!shouldExit)
    {
        loop_count++;
        // printf("[DisplayThread L%d] Loop iteration start.\n", loop_count);

        bool use_cuda_capture = false;
        {
            // printf("[DisplayThread L%d] Attempting to lock configMutex...\n", loop_count);
            std::lock_guard<std::mutex> lock(configMutex);
            use_cuda_capture = config.capture_use_cuda;
            // printf("[DisplayThread L%d] configMutex locked and released. use_cuda_capture: %d\n", loop_count, use_cuda_capture);
        }

        // --- Frame Acquisition ---
        bool new_frame_was_available = false;
        // printf("[DisplayThread L%d] Attempting to lock frameMutex for frame acquisition...\n", loop_count);
        {
            std::unique_lock<std::mutex> lock(frameMutex);
            // printf("[DisplayThread L%d] frameMutex locked for frame acquisition.\n", loop_count);
            // printf("[DisplayThread L%d] Waiting on frameCV. newFrameAvailable: %d, shouldExit: %d\n", loop_count, newFrameAvailable.load(std::memory_order_relaxed), shouldExit.load(std::memory_order_relaxed));
            
            frameCV.wait(lock, [&] { 
                // This lambda can be noisy, only log if something changes or for a few iterations
                // if (loop_count < 5) printf("[DisplayThread L%d] frameCV predicate check. newFrameAvailable: %d, shouldExit: %d\n", loop_count, newFrameAvailable.load(std::memory_order_relaxed), shouldExit.load(std::memory_order_relaxed));
                return newFrameAvailable.load(std::memory_order_relaxed) || shouldExit.load(std::memory_order_relaxed); 
            });
            // printf("[DisplayThread L%d] Woke up from frameCV. newFrameAvailable: %d, shouldExit: %d\n", loop_count, newFrameAvailable.load(std::memory_order_relaxed), shouldExit.load(std::memory_order_relaxed));

            if (shouldExit.load(std::memory_order_relaxed)) {
                // printf("[DisplayThread L%d] shouldExit is true, breaking loop.\n", loop_count);
                // Mutex will be released by lock_guard at scope end
                break;
            }

            if (newFrameAvailable.load(std::memory_order_relaxed)) {
                // printf("[DisplayThread L%d] newFrameAvailable is true.\n", loop_count);
                if (use_cuda_capture) {
                    // printf("[DisplayThread L%d] CUDA capture mode. latestFrameGpu empty: %d\n", loop_count, latestFrameGpu.empty());
                    if (!latestFrameGpu.empty()) {
                        acquiredGpuFrame = latestFrameGpu.clone(); 
                        // printf("[DisplayThread L%d] Cloned latestFrameGpu.\n", loop_count);
                    } else {
                        // printf("[DisplayThread L%d] Warning: CUDA mode but latestFrameGpu is empty.\n", loop_count);
                    }
                } else { // CPU Capture Mode
                    // printf("[DisplayThread L%d] CPU capture mode. latestFrameCpu empty: %d\n", loop_count, latestFrameCpu.empty());
                    if (!latestFrameCpu.empty()) {
                        acquiredCpuFrame = latestFrameCpu.clone();
                        // printf("[DisplayThread L%d] Cloned latestFrameCpu.\n", loop_count);
                    } else {
                        // printf("[DisplayThread L%d] Warning: CPU mode but latestFrameCpu is empty.\n", loop_count);
                    }
                }
                newFrameAvailable.store(false, std::memory_order_relaxed); // Reset flag
                new_frame_was_available = true;
                // printf("[DisplayThread L%d] newFrameAvailable reset, new_frame_was_available = true.\n", loop_count);
            }
            // printf("[DisplayThread L%d] Releasing frameMutex (end of scope) for frame acquisition.\n", loop_count);
        } // frameMutex unlocked

        if (!new_frame_was_available) {
            // printf("[DisplayThread L%d] No new frame was available after wake-up. Sleeping briefly.\n", loop_count);
            std::this_thread::sleep_for(std::chrono::milliseconds(5)); 
            continue;
        }

        // --- Process acquired frame to update the shared latestFrameCpu ---
        // printf("[DisplayThread L%d] Processing acquired frame. use_cuda_capture: %d\n", loop_count, use_cuda_capture);
        if (use_cuda_capture) {
            // printf("[DisplayThread L%d] CUDA mode processing. acquiredGpuFrame empty: %d\n", loop_count, acquiredGpuFrame.empty());
            if (!acquiredGpuFrame.empty()) {
                cv::Mat tempCpuFrame;
                // printf("[DisplayThread L%d] Downloading acquiredGpuFrame...\n", loop_count);
                try {
                    acquiredGpuFrame.download(tempCpuFrame, processingStream);
                    processingStream.waitForCompletion(); // Ensure download is complete
                    // printf("[DisplayThread L%d] Download complete. tempCpuFrame empty: %d\n", loop_count, tempCpuFrame.empty());
                } catch (const cv::Exception& e) {
                    // printf("[DisplayThread L%d] cv::Exception during GpuMat::download: %s\n", loop_count, e.what());
                    // Continue to allow thread to possibly recover or exit gracefully
                }

                if (!tempCpuFrame.empty()) {
                    // printf("[DisplayThread L%d] Attempting to lock frameMutex to update latestFrameCpu...\n", loop_count);
                    {
                        std::lock_guard<std::mutex> lock(frameMutex); 
                        // printf("[DisplayThread L%d] frameMutex locked to update latestFrameCpu.\n", loop_count);
                        latestFrameCpu = tempCpuFrame.clone(); 
                        // printf("[DisplayThread L%d] Updated global latestFrameCpu from Gpu download.\n", loop_count);
                        // printf("[DisplayThread L%d] Releasing frameMutex (end of scope) after updating latestFrameCpu.\n", loop_count);
                    }
                } else {
                    if (config.verbose) {} // printf("[DisplayThread L%d] Error: Downloaded GpuFrame is empty!\n", loop_count);
                    if (config.verbose) OutputDebugStringA("[Visuals] Error: Downloaded GpuFrame is empty!\n"); // Kept original ODS for consistency if it's used elsewhere
                }
            } else {
                 if (config.verbose) {} // printf("[DisplayThread L%d] Error: Acquired GpuFrame is empty in CUDA mode!\n", loop_count);
                 if (config.verbose) OutputDebugStringA("[Visuals] Error: Acquired GpuFrame is empty in CUDA mode!\n");
            }
        } else { // CPU Capture Mode
            // printf("[DisplayThread L%d] CPU mode processing. acquiredCpuFrame empty: %d\n", loop_count, acquiredCpuFrame.empty());
            if (!acquiredCpuFrame.empty()) {
                // In CPU capture mode, captureThread already updated latestFrameCpu.
                // acquiredCpuFrame is a clone of that. No need to update latestFrameCpu again from here.
                // printf("[DisplayThread L%d] CPU mode: latestFrameCpu already up-to-date by captureThread. No action needed with acquiredCpuFrame.\n", loop_count);
            } else {
                 if (config.verbose) {} // printf("[DisplayThread L%d] Error: Acquired CpuFrame is empty in CPU mode!\n", loop_count);
                 if (config.verbose) OutputDebugStringA("[Visuals] Error: Acquired CpuFrame is empty in CPU mode!\n");
            }
        }
        // printf("[DisplayThread L%d] Loop iteration end.\n\n", loop_count);
    }
    // printf("[DisplayThread] Thread exiting.\n");
}