#include "simple_capture.h"
#include "AppContext.h"
#include "../cuda/cuda_image_processing.h"
#include <iostream>
#include <chrono>

SimpleScreenCapture::SimpleScreenCapture(int width, int height) 
    : m_initialized(false), m_width(width), m_height(height),
      m_screenDC(nullptr), m_memoryDC(nullptr), m_bitmap(nullptr), m_bitmapData(nullptr)
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
    // Return CPU frame as GPU frame is problematic during stream capture
    // Let the caller handle GPU upload if needed
    SimpleMat cpuFrame = GetNextFrameCpu();
    if (cpuFrame.empty()) {
        return SimpleCudaMat();
    }
    
    // For now, return empty GpuMat to avoid stream capture conflicts
    // The detector should handle CPU frames or upload manually
    return SimpleCudaMat();
}

SimpleMat SimpleScreenCapture::GetNextFrameCpu()
{
    static int captureAttempts = 0;
    captureAttempts++;
    
    // Only log every 300 frames to reduce spam
    if (captureAttempts % 300 == 1) {
    }
    
    if (!m_initialized) {
        return SimpleMat();
    }
    
    // Initialization check passed (no need to log every frame)
    
    // Calculate center region coordinates
    int startX = (m_screenWidth - m_width) / 2;
    int startY = (m_screenHeight - m_height) / 2;
    
    if (captureAttempts % 60 == 1) {
    }
    
    // Try optimized BitBlt with NOCOPYBITS for better performance on high-res displays
    BOOL result = BitBlt(m_memoryDC, 0, 0, m_width, m_height, 
                        m_screenDC, startX, startY, SRCCOPY | NOMIRRORBITMAP);
    
    if (!result) {
        DWORD error = GetLastError();
        return SimpleMat();
    }
    
    // Verify bitmap data is valid
    if (!m_bitmapData) {
        return SimpleMat();
    }
    
    // Verify bitmap data periodically
    if (captureAttempts % 300 == 1 && m_bitmapData) {
    }
    
    // Convert BGRA to BGR
    SimpleMat bgrFrame(m_height, m_width, 3);
    
    if (captureAttempts % 60 == 1) {
    }
    
    // Verify source data before conversion
    const uint8_t* srcData = m_hostFrame.data();
    if (!srcData) {
        // Use bitmap data directly
        srcData = static_cast<const uint8_t*>(m_bitmapData);
        if (!srcData) {
            return SimpleMat();
        }
    }
    
    // Manual color conversion from BGRA to BGR
    for (int y = 0; y < m_height; ++y) {
        const uint8_t* srcRow = srcData + y * (m_width * 4);  // BGRA stride
        uint8_t* dstRow = bgrFrame.data() + y * bgrFrame.step();
        for (int x = 0; x < m_width; ++x) {
            dstRow[x * 3 + 0] = srcRow[x * 4 + 0]; // B
            dstRow[x * 3 + 1] = srcRow[x * 4 + 1]; // G
            dstRow[x * 3 + 2] = srcRow[x * 4 + 2]; // R
        }
    }
    
    if (captureAttempts % 60 == 1) {
    }
    
    return bgrFrame;
}