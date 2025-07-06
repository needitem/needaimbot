#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <windows.h>
#include <cuda_runtime.h>

class SimpleScreenCapture {
public:
    SimpleScreenCapture(int width, int height);
    ~SimpleScreenCapture();
    
    cv::cuda::GpuMat GetNextFrameGpu();
    cv::Mat GetNextFrameCpu();
    
    bool IsInitialized() const { return m_initialized; }
    void SetAcquireTimeout(UINT timeout) {} // No-op for compatibility
    
private:
    bool m_initialized;
    int m_width;
    int m_height;
    int m_screenWidth;
    int m_screenHeight;
    
    HDC m_screenDC;
    HDC m_memoryDC;
    HBITMAP m_bitmap;
    BITMAPINFO m_bmpInfo;
    unsigned char* m_bitmapData;
    
    cv::Mat m_hostFrame;
    cv::cuda::GpuMat m_deviceFrame;
    cv::cuda::GpuMat m_tempBgrFrame;
    
    // Raw CUDA memory for direct access
    void* m_cudaPtr;
    size_t m_cudaPitch;
};