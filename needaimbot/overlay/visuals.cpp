#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>

#include <iostream>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <cuda_runtime_api.h>
#include <cstdio> 

#include "visuals.h"
#include "config.h"
#include "needaimbot.h" 
#include "capture.h"   


extern std::mutex configMutex; 








void displayThread()
{
    
    cv::cuda::GpuMat acquiredGpuFrame;
    cv::Mat acquiredCpuFrame;
    cv::cuda::Stream processingStream; 

    int loop_count = 0;

    while (!shouldExit)
    {
        loop_count++;
        

        bool use_cuda_capture = false;
        {
            
            std::lock_guard<std::mutex> lock(configMutex);
            use_cuda_capture = config.capture_use_cuda;
            
        }

        
        bool new_frame_was_available = false;
        
        {
            std::unique_lock<std::mutex> lock(frameMutex);
            
            
            
            frameCV.wait(lock, [&] { 
                
                
                return newFrameAvailable.load(std::memory_order_relaxed) || shouldExit.load(std::memory_order_relaxed); 
            });
            

            if (shouldExit.load(std::memory_order_relaxed)) {
                
                
                break;
            }

            if (newFrameAvailable.load(std::memory_order_relaxed)) {
                
                if (use_cuda_capture) {
                    
                    if (!latestFrameGpu.empty()) {
                        acquiredGpuFrame = latestFrameGpu.clone(); 
                        
                    } else {
                        
                    }
                } else { 
                    
                    if (!latestFrameCpu.empty()) {
                        acquiredCpuFrame = latestFrameCpu.clone();
                        
                    } else {
                        
                    }
                }
                newFrameAvailable.store(false, std::memory_order_relaxed); 
                new_frame_was_available = true;
                
            }
            
        } 

        if (!new_frame_was_available) {
            
            std::this_thread::sleep_for(std::chrono::milliseconds(5)); 
            continue;
        }

        
        
        if (use_cuda_capture) {
            
            if (!acquiredGpuFrame.empty()) {
                cv::Mat tempCpuFrame;
                
                try {
                    acquiredGpuFrame.download(tempCpuFrame, processingStream);
                    processingStream.waitForCompletion(); 
                    
                } catch (const cv::Exception& e) {
                    
                    
                }

                if (!tempCpuFrame.empty()) {
                    
                    {
                        std::lock_guard<std::mutex> lock(frameMutex); 
                        
                        latestFrameCpu = tempCpuFrame.clone(); 
                        
                        
                    }
                } else {
                    if (config.verbose) {} 
                    if (config.verbose) OutputDebugStringA("[Visuals] Error: Downloaded GpuFrame is empty!\n"); 
                }
            } else {
                 if (config.verbose) {} 
                 if (config.verbose) OutputDebugStringA("[Visuals] Error: Acquired GpuFrame is empty in CUDA mode!\n");
            }
        } else { 
            
            if (!acquiredCpuFrame.empty()) {
                
                
                
            } else {
                 if (config.verbose) {} 
                 if (config.verbose) OutputDebugStringA("[Visuals] Error: Acquired CpuFrame is empty in CPU mode!\n");
            }
        }
        
    }
    
}
