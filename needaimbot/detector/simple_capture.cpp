#include "simple_capture.h"
#include "AppContext.h"
#include <iostream>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaimgproc.hpp>

SimpleScreenCapture::SimpleScreenCapture(int width, int height) 
    : m_initialized(false), m_width(width), m_height(height),
      m_screenDC(nullptr), m_memoryDC(nullptr), m_bitmap(nullptr), m_bitmapData(nullptr)
{
    // Get screen dimensions
    m_screenWidth = GetSystemMetrics(SM_CXSCREEN);
    m_screenHeight = GetSystemMetrics(SM_CYSCREEN);
    
    if (AppContext::getInstance().config.verbose) {
        std::cout << "[SimpleCapture] Screen resolution: " << m_screenWidth << "x" << m_screenHeight << std::endl;
        std::cout << "[SimpleCapture] Capture region: " << m_width << "x" << m_height << std::endl;
    }
    
    // Get screen DC
    m_screenDC = GetDC(NULL);
    if (!m_screenDC) {
        std::cerr << "[SimpleCapture] Failed to get screen DC" << std::endl;
        return;
    }
    
    // Create compatible DC
    m_memoryDC = CreateCompatibleDC(m_screenDC);
    if (!m_memoryDC) {
        std::cerr << "[SimpleCapture] Failed to create memory DC" << std::endl;
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
        std::cerr << "[SimpleCapture] Failed to create DIB section" << std::endl;
        DeleteDC(m_memoryDC);
        ReleaseDC(NULL, m_screenDC);
        return;
    }
    
    // Select bitmap into memory DC
    SelectObject(m_memoryDC, m_bitmap);
    
    // Pre-allocate OpenCV matrices
    m_hostFrame = cv::Mat(m_height, m_width, CV_8UC4, m_bitmapData);
    
    // Pre-allocate GPU memory to avoid allocation during stream capture
    try {
        m_deviceFrame.create(m_height, m_width, CV_8UC4); // BGRA first
        m_tempBgrFrame.create(m_height, m_width, CV_8UC3); // BGR result
    } catch (const cv::Exception& e) {
        std::cerr << "[SimpleCapture] Failed to pre-allocate GPU memory: " << e.what() << std::endl;
        DeleteObject(m_bitmap);
        DeleteDC(m_memoryDC);
        ReleaseDC(NULL, m_screenDC);
        return;
    }
    
    m_initialized = true;
    
    if (AppContext::getInstance().config.verbose) {
        std::cout << "[SimpleCapture] Initialized successfully" << std::endl;
    }
}

SimpleScreenCapture::~SimpleScreenCapture()
{
    if (m_bitmap) DeleteObject(m_bitmap);
    if (m_memoryDC) DeleteDC(m_memoryDC);
    if (m_screenDC) ReleaseDC(NULL, m_screenDC);
}

cv::cuda::GpuMat SimpleScreenCapture::GetNextFrameGpu()
{
    // Return CPU frame as GPU frame is problematic during stream capture
    // Let the caller handle GPU upload if needed
    cv::Mat cpuFrame = GetNextFrameCpu();
    if (cpuFrame.empty()) {
        return cv::cuda::GpuMat();
    }
    
    // For now, return empty GpuMat to avoid stream capture conflicts
    // The detector should handle CPU frames or upload manually
    return cv::cuda::GpuMat();
}

cv::Mat SimpleScreenCapture::GetNextFrameCpu()
{
    if (!m_initialized) {
        return cv::Mat();
    }
    
    // Calculate center region coordinates
    int startX = (m_screenWidth - m_width) / 2;
    int startY = (m_screenHeight - m_height) / 2;
    
    // Capture screen region using BitBlt with CAPTUREBLT flag for better performance
    BOOL result = BitBlt(m_memoryDC, 0, 0, m_width, m_height, 
                        m_screenDC, startX, startY, SRCCOPY | CAPTUREBLT);
    
    if (!result) {
        std::cerr << "[SimpleCapture] BitBlt failed" << std::endl;
        return cv::Mat();
    }
    
    // Convert BGRA to BGR
    cv::Mat bgrFrame;
    cv::cvtColor(m_hostFrame, bgrFrame, cv::COLOR_BGRA2BGR);
    
    return bgrFrame.clone(); // Return a copy since m_hostFrame is reused
}