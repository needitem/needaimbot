#pragma once

#include "../cuda/simple_cuda_mat.h"
#include <windows.h>
#include <cuda_runtime.h>

class SimpleScreenCapture {
public:
    SimpleScreenCapture(int width, int height);
    ~SimpleScreenCapture();
    
    SimpleCudaMat GetNextFrameGpu();
    SimpleMat GetNextFrameCpu();
    
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
    
    SimpleMat m_hostFrame;
    SimpleCudaMat m_deviceFrame;
    SimpleCudaMat m_tempBgrFrame;
    
    // Raw CUDA memory for direct access
    void* m_cudaPtr;
    size_t m_cudaPitch;
};