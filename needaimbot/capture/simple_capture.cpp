#include "simple_capture.h"
#include "AppContext.h"
#include "../cuda/cuda_image_processing.h"
#include "../cuda/color_conversion.h"
#include "frame_buffer_pool.h"
#include <iostream>
#include <chrono>
#include <algorithm>  // For std::min
#include <ppl.h>  // For parallel_for

SimpleScreenCapture::SimpleScreenCapture(int width, int height) 
    : m_initialized(false), m_width(width), m_height(height),
      m_screenDC(nullptr), m_memoryDC(nullptr), m_bitmap(nullptr), m_bitmapData(nullptr),
      m_retryCount(0), m_maxRetries(3)
{
    
    // Get screen dimensions
    m_screenWidth = GetSystemMetrics(SM_CXSCREEN);
    m_screenHeight = GetSystemMetrics(SM_CYSCREEN);
    
    
    // Get screen DC
    m_screenDC = GetDC(NULL);
    if (!m_screenDC) {
        return;
    }
    
    // Create compatible DC
    m_memoryDC = CreateCompatibleDC(m_screenDC);
    if (!m_memoryDC) {
        ReleaseDC(NULL, m_screenDC);
        return;
    }
    
    // Setup bitmap info
    ZeroMemory(&m_bmpInfo, sizeof(BITMAPINFO));
    m_bmpInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    m_bmpInfo.bmiHeader.biWidth = m_width;
    m_bmpInfo.bmiHeader.biHeight = -m_height; // Negative for top-down bitmap
    m_bmpInfo.bmiHeader.biPlanes = 1;
    m_bmpInfo.bmiHeader.biBitCount = 32; // BGRA
    m_bmpInfo.bmiHeader.biCompression = BI_RGB;
    
    // Create DIB section
    m_bitmap = CreateDIBSection(m_memoryDC, &m_bmpInfo, DIB_RGB_COLORS, 
                               (void**)&m_bitmapData, NULL, 0);
    if (!m_bitmap) {
        DeleteDC(m_memoryDC);
        ReleaseDC(NULL, m_screenDC);
        return;
    }
    
    // Select bitmap into memory DC
    SelectObject(m_memoryDC, m_bitmap);
    
    // Don't use external data wrapper for now - it seems to be causing issues
    // m_hostFrame = SimpleMat(m_height, m_width, 4, m_bitmapData, m_width * 4);
    
    // std::cout << "[SimpleCapture] Host frame created: " << m_hostFrame.cols() << "x" << m_hostFrame.rows() 
    //           << " channels=" << m_hostFrame.channels() << " empty=" << m_hostFrame.empty() << std::endl;
    
    // Pre-allocate GPU memory to avoid allocation during stream capture
    m_deviceFrame.create(m_height, m_width, 4); // BGRA first
    m_tempBgrFrame.create(m_height, m_width, 3); // BGR result
    
    m_initialized = true;
    
}

SimpleScreenCapture::~SimpleScreenCapture()
{
    if (m_bitmap) DeleteObject(m_bitmap);
    if (m_memoryDC) DeleteDC(m_memoryDC);
    if (m_screenDC) ReleaseDC(NULL, m_screenDC);
}

SimpleCudaMat SimpleScreenCapture::GetNextFrameGpu()
{
    if (!m_initialized) {
        return SimpleCudaMat();
    }
    
    // Calculate center region coordinates
    int startX = (m_screenWidth - m_width) / 2;
    int startY = (m_screenHeight - m_height) / 2;
    
    // Get frame buffer pool
    if (!g_frameBufferPool) {
        g_frameBufferPool = std::make_unique<FrameBufferPool>(10);
    }
    
    // Perform screen capture with retry logic
    BOOL result = FALSE;
    int retries = 0;
    
    while (!result && retries < m_maxRetries) {
        result = BitBlt(m_memoryDC, 0, 0, m_width, m_height, 
                        m_screenDC, startX, startY, SRCCOPY | NOMIRRORBITMAP);
        
        if (!result) {
            if (retries < m_maxRetries - 1) {
                // Exponential backoff
                int sleepMs = (1 << retries) * 10; // 10ms, 20ms, 40ms
                std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
                retries++;
            } else {
                // Final failure
                return SimpleCudaMat();
            }
        }
    }
    
    // Verify bitmap data is valid
    if (!m_bitmapData) {
        return SimpleCudaMat();
    }
    
    // Get GPU buffer from pool - keep BGRA format (4 channels)
    SimpleCudaMat gpuFrame = g_frameBufferPool->acquireGpuBuffer(m_height, m_width, 4);
    
    // Upload bitmap data directly to GPU (already in BGRA format)
    cudaError_t err = cudaMemcpy2D(
        gpuFrame.data(), gpuFrame.step(),
        m_bitmapData, m_width * 4,
        m_width * 4, m_height,
        cudaMemcpyHostToDevice
    );
    
    if (err != cudaSuccess) {
        g_frameBufferPool->releaseGpuBuffer(std::move(gpuFrame));
        return SimpleCudaMat();
    }
    
    // No conversion needed - return BGRA directly
    return gpuFrame;
}

