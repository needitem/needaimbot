#pragma once

#include "../cuda/simple_cuda_mat.h"
#include "capture.h"
#include <windows.h>
#include <cuda_runtime.h>
#include <thread>

class SimpleScreenCapture : public IScreenCapture {
public:
    SimpleScreenCapture(int width, int height);
    ~SimpleScreenCapture();
    
    SimpleCudaMat GetNextFrameGpu() override;
    cudaEvent_t GetCaptureDoneEvent() const override { return nullptr; } // Simple capture doesn't use events
    bool IsInitialized() const override { return m_initialized; }
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
    
    SimpleCudaMat m_deviceFrame;
    SimpleCudaMat m_tempBgrFrame;
    
    // 에러 복구를 위한 카운터
    int m_retryCount;
    const int m_maxRetries;
};